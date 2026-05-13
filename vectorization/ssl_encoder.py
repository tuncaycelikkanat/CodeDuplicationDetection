import torch
import numpy as np
from tqdm import tqdm

def extract_ssl_embeddings(codes, model_name="microsoft/codebert-base", batch_size=32, device="cpu"):
    """
    Extracts dense embeddings for a list of code snippets using a HuggingFace SSL model (e.g., CodeBERT).
    """
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
            
    if embeddings:
        return np.vstack(embeddings).astype(np.float32)
    return np.array([], dtype=np.float32)

def build_ssl_pipeline(model_name="microsoft/codebert-base", device="cpu"):
    """
    Returns the tokenizer and model for single-item inference.
    """
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model
