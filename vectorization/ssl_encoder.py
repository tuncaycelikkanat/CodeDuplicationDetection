import os
import torch
import numpy as np
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

    Args:
        codes: List of raw code strings.
        model_name: HuggingFace model identifier.
        batch_size: Batch size for inference.
        device: 'cpu' | 'cuda' | 'xpu'
        cache_path: Opsiyonel .npy dosya yolu. Dosya mevcutsa embedding'ler
                    buradan yüklenir (re-extraction atlanır). Dosya yoksa
                    extraction sonrası bu yola kaydedilir.

    Returns:
        np.ndarray of shape (len(codes), hidden_size), dtype=float32
    """
    # ── Cache yükleme ───────────────────────────────────────────────────────
    if cache_path is not None and os.path.exists(cache_path):
        print(f"  → Loading SSL embeddings from cache: {cache_path}")
        cached = np.load(cache_path)
        if cached.shape[0] == len(codes):
            print(f"  → Cache hit: {cached.shape}")
            return cached.astype(np.float32)
        else:
            print(f"  ⚠️  Cache shape mismatch ({cached.shape[0]} vs {len(codes)}), re-extracting...")

    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError("Please install 'transformers' and 'torch' to use SSL embeddings.")

    print(f"  → Loading SSL model: {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []

    print(f"  → Extracting SSL embeddings in batches of {batch_size}...")
    for i in tqdm(range(0, len(codes), batch_size), desc="SSL Embeddings"):
        batch_codes = codes[i:i+batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_codes,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Use the [CLS] token representation (the first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)

    result = np.vstack(embeddings).astype(np.float32) if embeddings else np.array([], dtype=np.float32)

    # ── Cache kaydetme ──────────────────────────────────────────────────────
    if cache_path is not None and result.size > 0:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.save(cache_path, result)
        print(f"  → SSL embeddings saved to cache: {cache_path}")

    return result


def build_ssl_pipeline(model_name: str = "microsoft/codebert-base", device: str = "cpu"):
    """
    Returns the tokenizer and model for single-item inference.
    Model singleton olarak app.py'de bir kez yüklenmeli, her request'te
    yeniden çağrılmamalıdır.
    """
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model

