import os
import sys
import math
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.logger import Log

from typing import Optional, Tuple
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing.tokenizer import normalize_tokens, tokenize
from preprocessing.code_features import _extract_single
from rapidfuzz.distance import Levenshtein
from utils.similarity_utils import _jaccard_sim, _string_bigram_jaccard, _tuple_bigram_jaccard


# ================= SSL HELPER =================

def _extract_single_ssl_embedding(code: str, ssl_tokenizer, ssl_model) -> np.ndarray:
    """
    Tek bir kod snippet'i için SSL (CodeBERT) embedding'i çıkarır.
    Eğitimdeki ssl_encoder.py ile tutarlı: Mean Pooling + Chunking (510 token).
    
    Bu fonksiyon daha önce CLS token kullanıyordu — eğitim/inference uyumsuzluğuna
    neden oluyordu. Artık eğitimdeki ile aynı yöntemi (Mean Pooling) kullanıyor.
    """
    import torch

    device = next(ssl_model.parameters()).device

    # Tokenize without special tokens (we add them manually per chunk)
    encoded = ssl_tokenizer(code, add_special_tokens=False)
    tokens = encoded["input_ids"]
    if not tokens:
        tokens = [ssl_tokenizer.unk_token_id]

    # Chunking: 510 tokens per chunk (leaving room for CLS + SEP)
    chunk_embeddings = []
    for start in range(0, len(tokens), 510):
        chunk_tokens = tokens[start:start + 510]
        chunk_ids = [ssl_tokenizer.cls_token_id] + chunk_tokens + [ssl_tokenizer.sep_token_id]
        attention_mask = [1] * len(chunk_ids)

        input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=device)
        attn_mask = torch.tensor([attention_mask], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = ssl_model(input_ids=input_ids, attention_mask=attn_mask)
            hidden = outputs.last_hidden_state  # (1, seq_len, 768)

            # Mean Pooling (excluding padding — here there is no padding, but for consistency)
            mask_exp = attn_mask.unsqueeze(-1).expand(hidden.size()).float()
            sum_emb = torch.sum(hidden * mask_exp, dim=1)
            sum_mask_val = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
            mean_pool = (sum_emb / sum_mask_val).cpu().numpy()  # (1, 768)
            chunk_embeddings.append(mean_pool[0])

    # Aggregate all chunks via mean (same as ssl_encoder.py)
    return np.mean(chunk_embeddings, axis=0).astype(np.float32)  # (768,)


# ================= PIPELINE =================

def build_pair_vector(
    raw1: str,
    raw2: str,
    vectorizer,
    svd_model=None,
    ssl_pipeline=None,
    ssl_pca=None,
) -> np.ndarray:
    """
    Dense feature extraction --- bir cift kod snippet icin.
    Web Demo ve Automation scriptleri tarafindan kullanilir.

    Dondurulen array pair_generator.py ile birebir ayni feature sirasi:
        [0]      cos_token         <- CASCADE FILTRESI BURAYA BAKAR
        [1]      length_ratio
        [2]      manhattan_token
        [3]      euclidean_token
        [4..31]  AST ratios + diffs  (STAGE1_FEATURE_COUNT siniri)
        [32]     cf_sim
        [33..39] Semantic Jaccard x6 + abstract CF
        [40]     type_profile_cosine
        [41..90] svd_diff            (sadece svd_model verilmisse, 50 boyut)
        [91..218] ssl_pca_abs_diff   (sadece ssl_pipeline + ssl_pca verilmisse, 128 boyut)
        [219..346] ssl_pca_product   (sadece ssl_pipeline + ssl_pca verilmisse, 128 boyut)

    UYARI — ssl_pca: Egitimde fit edilmis PCA nesnesi.  ssl_pipeline ile birlikte
             verilmezse SSL ozellikleri feature vektorune eklenmez (boyut uyumsuzlugu).
    """
    def preprocess(code):
        tokens = tokenize(code)
        return " ".join(normalize_tokens(tokens))

    code1 = preprocess(raw1)
    code2 = preprocess(raw2)

    X1 = vectorizer.transform([code1])
    X2 = vectorizer.transform([code2])

    diff = abs(X1 - X2)
    cos_token = cosine_similarity(X1, X2)[0][0]

    len1 = len(code1.split())
    len2 = len(code2.split())
    length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0

    manhattan = float(diff.sum())
    euclidean = float(np.sqrt(diff.power(2).sum()))

    # extra başlar: cos_token, length_ratio, manhattan, euclidean
    extra = [cos_token, length_ratio, manhattan, euclidean]

    # AST + CF + Semantic
    feat1, cf1, sem1 = _extract_single(raw1)
    feat2, cf2, sem2 = _extract_single(raw2)

    # AST ratios (20) + AST diffs (20) — pair_generator.py ile aynı sıra
    ast_ratios = []
    ast_diffs = []
    for v1, v2 in zip(feat1, feat2):
        max_val = max(v1, v2)
        ast_ratios.append(min(v1, v2) / max_val if max_val > 0 else 1.0)  # ratio
        ast_diffs.append(abs(v1 - v2))                                     # diff
        
    extra.extend(ast_ratios)
    extra.extend(ast_diffs)

    # CF pattern similarity
    cf_dist = Levenshtein.distance(cf1, cf2)
    cf_max = max(len(cf1), len(cf2))
    extra.append(1.0 - (cf_dist / cf_max) if cf_max > 0 else 1.0)

    # Semantic features (7)
    extra.append(_jaccard_sim(sem1['library_calls'], sem2['library_calls']))
    extra.append(_jaccard_sim(sem1['library_categories'], sem2['library_categories']))
    extra.append(_jaccard_sim(sem1['data_structs'], sem2['data_structs']))
    extra.append(_string_bigram_jaccard(sem1['io_pattern'], sem2['io_pattern']))
    extra.append(_jaccard_sim(sem1['math_ops'], sem2['math_ops']))
    extra.append(_tuple_bigram_jaccard(sem1['skeleton'], sem2['skeleton']))
    
    # Abstract CF Levenshtein
    acf1, acf2 = sem1['abstract_cf'], sem2['abstract_cf']
    if not acf1 and not acf2:
        acf_sim = 1.0
    elif not acf1 or not acf2:
        acf_sim = 0.0
    else:
        acf_dist = Levenshtein.distance(acf1, acf2)
        acf_max = max(len(acf1), len(acf2))
        acf_sim = 1.0 - (acf_dist / acf_max) if acf_max > 0 else 1.0
    extra.append(acf_sim)

    # Type profile cosine
    tp1, tp2 = sem1['type_profile'], sem2['type_profile']
    dot = np.dot(tp1, tp2)
    norm = np.linalg.norm(tp1) * np.linalg.norm(tp2)
    tp_cos = dot / norm if norm > 0 else 1.0
    extra.append(tp_cos)

    # SVD farkları (opsiyonel, pair_generator.py'deki X_svd'ye karşılık gelir)
    if svd_model is not None:
        svd1 = svd_model.transform(X1)[0]
        svd2 = svd_model.transform(X2)[0]
        svd_diff = np.abs(svd1 - svd2)
        extra.extend(svd_diff.tolist())

    # SSL özellikleri (opsiyonel) — Mean Pooling + Chunking + PCA abs diff + element-wise product
    # NOT: Eğitimdeki ssl_encoder.py ile tutarlı Mean Pooling kullanılır.
    # Sentence-BERT / NLI literatüründen: concat(|u-v|, u*v) — hem mesafe hem yönsel etkileşim.
    if ssl_pipeline is not None and ssl_pca is not None:
        ssl_tokenizer, ssl_model = ssl_pipeline
        emb1 = _extract_single_ssl_embedding(raw1, ssl_tokenizer, ssl_model)  # (768,)
        emb2 = _extract_single_ssl_embedding(raw2, ssl_tokenizer, ssl_model)  # (768,)
        # PCA indirgeme (eğitimde fit edilmiş)
        emb1_r = ssl_pca.transform(emb1.reshape(1, -1)).astype(np.float32)  # (1, ssl_dim)
        emb2_r = ssl_pca.transform(emb2.reshape(1, -1)).astype(np.float32)  # (1, ssl_dim)
        ssl_diff = np.abs(emb1_r - emb2_r)[0]                               # (ssl_dim,)
        ssl_product = (emb1_r * emb2_r)[0]                                   # (ssl_dim,)
        extra.extend(ssl_diff.tolist())
        extra.extend(ssl_product.tolist())
    elif ssl_pipeline is not None and ssl_pca is None:
        # Geriye dönük uyumluluk: PCA yoksa 2 skaler (eski davranış)
        ssl_tokenizer, ssl_model = ssl_pipeline
        emb1 = _extract_single_ssl_embedding(raw1, ssl_tokenizer, ssl_model)
        emb2 = _extract_single_ssl_embedding(raw2, ssl_tokenizer, ssl_model)
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        ssl_cos = dot / (norm1 * norm2) if (norm1 * norm2) > 0 else 1.0
        ssl_euclidean = float(np.linalg.norm(emb1 - emb2))
        extra.extend([ssl_cos, ssl_euclidean])

    return np.array([extra], dtype=np.float32)
