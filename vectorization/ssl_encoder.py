import os
import torch
import numpy as np
import hashlib
from typing import Optional, List
from tqdm import tqdm


def extract_ssl_embeddings(
    codes: List[str],
    model_name: str = "microsoft/codebert-base",
    batch_size: int = 32,
    device: str = "cpu",
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """
    Extracts dense embeddings for a list of code snippets using a HuggingFace SSL model (e.g., CodeBERT).
    Implements chunking for codes > 512 tokens and Mean Pooling for better semantic representation.

    Args:
        codes: List of raw code strings.
        model_name: HuggingFace model identifier.
        batch_size: Batch size for inference.
        device: 'cpu' | 'cuda' | 'xpu'
        cache_path: Optional path to save/load embeddings.
    """
    # ---- Cache Hash Strategy ----
    # Length validation is insufficient (codes can be swapped). Use MD5 hash of all codes.
    codes_concat = "".join(codes).encode('utf-8')
    codes_hash = hashlib.md5(codes_concat).hexdigest()
    
    # If cache_path is provided, we append the hash to ensure invalidation
    if cache_path is not None:
        base, ext = os.path.splitext(cache_path)
        actual_cache_path = f"{base}_{codes_hash}{ext}"
    else:
        actual_cache_path = None

    # ---- Cache Loading ----
    if actual_cache_path is not None and os.path.exists(actual_cache_path):
        print(f"  → Loading SSL embeddings from cache: {actual_cache_path}")
        cached = np.load(actual_cache_path)
        if cached.shape[0] == len(codes):
            print(f"  → Cache hit: {cached.shape}")
            return cached.astype(np.float32)

    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError("Please install 'transformers' and 'torch' to use SSL embeddings.")

    print(f"  → Loading SSL model: {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []

    print(f"  → Extracting SSL embeddings (batch_size={batch_size}, chunking=True, pooling=Mean)...")
    for i in tqdm(range(0, len(codes), batch_size), desc="SSL Embeddings"):
        batch_codes = codes[i:i+batch_size]

        all_chunks_ids = []
        all_chunks_masks = []
        chunk_to_code_idx = []
        
        for c_idx, code in enumerate(batch_codes):
            encoded = tokenizer(code, add_special_tokens=False)
            tokens = encoded["input_ids"]
            if not tokens:
                tokens = [tokenizer.unk_token_id]
                
            # Chunking logic (510 to leave space for CLS and SEP)
            for start in range(0, len(tokens), 510):
                chunk_tokens = tokens[start:start+510]
                chunk_ids = [tokenizer.cls_token_id] + chunk_tokens + [tokenizer.sep_token_id]
                all_chunks_ids.append(chunk_ids)
                all_chunks_masks.append([1] * len(chunk_ids))
                chunk_to_code_idx.append(c_idx)

        # Pad chunks in this batch
        max_len = min(512, max(len(c) for c in all_chunks_ids))
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(all_chunks_ids, all_chunks_masks):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
            padded_masks.append(mask + [0] * pad_len)

        chunk_tensor = torch.tensor(padded_ids, dtype=torch.long, device=device)
        mask_tensor = torch.tensor(padded_masks, dtype=torch.long, device=device)
        
        chunk_embeddings = []
        # Sub-batch processing for chunks to avoid OOM
        sub_batch_size = batch_size
        for j in range(0, len(chunk_tensor), sub_batch_size):
            sub_ids = chunk_tensor[j:j+sub_batch_size]
            sub_mask = mask_tensor[j:j+sub_batch_size]
            
            with torch.no_grad():
                outputs = model(input_ids=sub_ids, attention_mask=sub_mask)
                hidden = outputs.last_hidden_state
                
                # Mean Pooling (excluding padding tokens)
                mask_exp = sub_mask.unsqueeze(-1).expand(hidden.size()).float()
                sum_emb = torch.sum(hidden * mask_exp, 1)
                sum_mask_val = torch.clamp(mask_exp.sum(1), min=1e-9)
                mean_pool = (sum_emb / sum_mask_val).cpu().numpy()
                chunk_embeddings.append(mean_pool)

        chunk_embeddings = np.vstack(chunk_embeddings)

        # Aggregate chunks back to original code representations via mean
        batch_res = []
        chunk_to_code_idx_arr = np.array(chunk_to_code_idx)
        for c_idx in range(len(batch_codes)):
            mask = (chunk_to_code_idx_arr == c_idx)
            if mask.any():
                batch_res.append(np.mean(chunk_embeddings[mask], axis=0))
            else:
                batch_res.append(np.zeros(768, dtype=np.float32))
                
        embeddings.extend(batch_res)

    result = np.vstack(embeddings).astype(np.float32) if embeddings else np.array([], dtype=np.float32)

    # ---- Cache Saving ----
    if actual_cache_path is not None and result.size > 0:
        os.makedirs(os.path.dirname(os.path.abspath(actual_cache_path)), exist_ok=True)
        np.save(actual_cache_path, result)
        print(f"  → SSL embeddings saved to cache: {actual_cache_path}")

    return result


def build_ssl_pipeline(model_name: str = "microsoft/codebert-base", device: str = "cpu"):
    """
    Returns the tokenizer and model for single-item inference.
    """
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model
