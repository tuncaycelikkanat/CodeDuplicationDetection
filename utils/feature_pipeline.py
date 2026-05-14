import os
import sys
import math
import numpy as np
from typing import Optional, Tuple
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Project root for imports
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from preprocessing.tokenizer import normalize_tokens, tokenize
from preprocessing.code_features import _extract_single
from rapidfuzz.distance import Levenshtein
from utils.similarity_utils import _jaccard_sim, _string_bigram_jaccard, _tuple_bigram_jaccard

# ================= PIPELINE =================

def build_pair_vector(
    raw1: str,
    raw2: str,
    vectorizer,
    char_vectorizer=None,
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
        [91..154] ssl_pca_diff       (sadece ssl_pipeline + ssl_pca verilmisse, 64 boyut)

    UYARI — char_vectorizer: Egitimde kullanilmadiysa None gecirin.
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

    char_diff = None
    if char_vectorizer is not None:
        C1 = char_vectorizer.transform([code1])
        C2 = char_vectorizer.transform([code2])
        char_diff = abs(C1 - C2)
        extra.append(cosine_similarity(C1, C2)[0][0])   # cos_char

    # AST + CF + Semantic
    feat1, cf1, sem1 = _extract_single(raw1)
    feat2, cf2, sem2 = _extract_single(raw2)

    # AST ratios (14) + AST diffs (14) — pair_generator.py ile aynı sıra
    for v1, v2 in zip(feat1, feat2):
        max_val = max(v1, v2)
        extra.append(min(v1, v2) / max_val if max_val > 0 else 1.0)  # ratio
        extra.append(abs(v1 - v2))                                     # diff

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

    # SSL ozellikleri (opsiyonel) — PCA ile indirgenmis abs diff vektoru
    if ssl_pipeline is not None and ssl_pca is not None:
        ssl_tokenizer, ssl_model = ssl_pipeline
        import torch
        inputs = ssl_tokenizer(
            [code1, code2], return_tensors="pt",
            max_length=512, truncation=True, padding=True
        )
        device = next(ssl_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = ssl_model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # (2, 768)
        emb1, emb2 = cls_emb[0:1], cls_emb[1:2]  # (1, 768) her biri
        # PCA indirgeme (egitimde fit edilmis)
        emb1_r = ssl_pca.transform(emb1).astype(np.float32)  # (1, ssl_dim)
        emb2_r = ssl_pca.transform(emb2).astype(np.float32)  # (1, ssl_dim)
        ssl_diff = np.abs(emb1_r - emb2_r)[0]                # (ssl_dim,)
        extra.extend(ssl_diff.tolist())
    elif ssl_pipeline is not None and ssl_pca is None:
        # Geriye donuk uyumluluk: PCA yoksa 2 skaler (eski davranis)
        ssl_tokenizer, ssl_model = ssl_pipeline
        import torch
        inputs = ssl_tokenizer(
            [code1, code2], return_tensors="pt",
            max_length=512, truncation=True, padding=True
        )
        device = next(ssl_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = ssl_model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            ssl1, ssl2 = cls_emb[0], cls_emb[1]
            dot = np.dot(ssl1, ssl2)
            norm1 = np.linalg.norm(ssl1)
            norm2 = np.linalg.norm(ssl2)
            ssl_cos = dot / (norm1 * norm2) if (norm1 * norm2) > 0 else 1.0
            ssl_euclidean = float(np.linalg.norm(ssl1 - ssl2))
            extra.extend([ssl_cos, ssl_euclidean])

    return np.array([extra], dtype=np.float32)
