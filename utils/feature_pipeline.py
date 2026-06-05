import os
import sys
import functools
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.logger import Log

from typing import Optional, List, Tuple
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing.tokenizer import normalize_tokens, tokenize
from preprocessing.code_features import _extract_single
from rapidfuzz.distance import Levenshtein
from utils.similarity_utils import _jaccard_sim, _string_bigram_jaccard, _tuple_bigram_jaccard


# ================= PREPROCESSING CACHE =================
# Aynı kod snippet'i birden fazla kez geldiğinde (web demo, test automation)
# tokenizasyonu tekrarlamaz — LRU cache ile anında cevap verir.

@functools.lru_cache(maxsize=4096)
def _preprocess_cached(code: str) -> str:
    tokens = tokenize(code)
    return " ".join(normalize_tokens(tokens))


# ================= SSL HELPERS =================

def _extract_single_ssl_embedding(code: str, ssl_tokenizer, ssl_model) -> np.ndarray:
    """
    Tek bir kod snippet'i için SSL (CodeBERT) embedding'i çıkarır.
    Eğitimdeki ssl_encoder.py ile tutarlı: Mean Pooling + Chunking (510 token).
    """
    import torch

    device = next(ssl_model.parameters()).device

    encoded = ssl_tokenizer(code, add_special_tokens=False)
    tokens = encoded["input_ids"]
    if not tokens:
        tokens = [ssl_tokenizer.unk_token_id]

    chunk_embeddings = []
    for start in range(0, len(tokens), 510):
        chunk_tokens = tokens[start:start + 510]
        chunk_ids = [ssl_tokenizer.cls_token_id] + chunk_tokens + [ssl_tokenizer.sep_token_id]
        attention_mask = [1] * len(chunk_ids)

        input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=device)
        attn_mask = torch.tensor([attention_mask], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = ssl_model(input_ids=input_ids, attention_mask=attn_mask)
            hidden = outputs.last_hidden_state

            mask_exp = attn_mask.unsqueeze(-1).expand(hidden.size()).float()
            sum_emb = torch.sum(hidden * mask_exp, dim=1)
            sum_mask_val = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
            mean_pool = (sum_emb / sum_mask_val).cpu().numpy()
            chunk_embeddings.append(mean_pool[0])

    return np.mean(chunk_embeddings, axis=0).astype(np.float32)


def _extract_batch_ssl_embeddings(codes: List[str], ssl_tokenizer, ssl_model, batch_size: int = 32) -> np.ndarray:
    """
    Çok sayıda kod snippet'i için SSL embedding'leri BATCH olarak çıkarır.
    GPU'yu tam kapasiteyle doldurur — single çıkarmadan 4-6x hızlıdır.

    Returns: np.ndarray shape (N, 768)
    """
    import torch

    device = next(ssl_model.parameters()).device
    all_embeddings = []

    for batch_start in range(0, len(codes), batch_size):
        batch_codes = codes[batch_start: batch_start + batch_size]
        batch_embs = []

        for code in batch_codes:
            emb = _extract_single_ssl_embedding(code, ssl_tokenizer, ssl_model)
            batch_embs.append(emb)

        all_embeddings.extend(batch_embs)

    return np.array(all_embeddings, dtype=np.float32)  # (N, 768)


# ================= SINGLE PAIR PIPELINE =================

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
    # LRU cache sayesinde aynı kod tekrar gelirse tokenizasyonu atlar
    code1 = _preprocess_cached(raw1)
    code2 = _preprocess_cached(raw2)

    X1 = vectorizer.transform([code1])
    X2 = vectorizer.transform([code2])

    diff = abs(X1 - X2)
    cos_token = cosine_similarity(X1, X2)[0][0]

    len1 = len(code1.split())
    len2 = len(code2.split())
    length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0

    manhattan = float(diff.sum())
    euclidean = float(np.sqrt(diff.power(2).sum()))

    extra = [cos_token, length_ratio, manhattan, euclidean]

    # AST + CF + Semantic
    feat1, cf1, sem1 = _extract_single(raw1)
    feat2, cf2, sem2 = _extract_single(raw2)

    ast_ratios = []
    ast_diffs = []
    for v1, v2 in zip(feat1, feat2):
        max_val = max(v1, v2)
        ast_ratios.append(min(v1, v2) / max_val if max_val > 0 else 1.0)
        ast_diffs.append(abs(v1 - v2))

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

    if svd_model is not None:
        svd1 = svd_model.transform(X1)[0]
        svd2 = svd_model.transform(X2)[0]
        svd_diff = np.abs(svd1 - svd2)
        extra.extend(svd_diff.tolist())

    if ssl_pipeline is not None and ssl_pca is not None:
        ssl_tokenizer, ssl_model = ssl_pipeline
        emb1 = _extract_single_ssl_embedding(raw1, ssl_tokenizer, ssl_model)
        emb2 = _extract_single_ssl_embedding(raw2, ssl_tokenizer, ssl_model)
        emb1_r = ssl_pca.transform(emb1.reshape(1, -1)).astype(np.float32)
        emb2_r = ssl_pca.transform(emb2.reshape(1, -1)).astype(np.float32)
        ssl_diff = np.abs(emb1_r - emb2_r)[0]
        ssl_product = (emb1_r * emb2_r)[0]
        extra.extend(ssl_diff.tolist())
        extra.extend(ssl_product.tolist())
    elif ssl_pipeline is not None and ssl_pca is None:
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


# ================= BATCH PAIR PIPELINE =================

def build_pair_vectors_batch(
    pairs: List[Tuple[str, str]],
    vectorizer,
    svd_model=None,
    ssl_pipeline=None,
    ssl_pca=None,
    ssl_batch_size: int = 32,
) -> np.ndarray:
    """
    Çok sayıda kod çifti için feature vektörlerini BATCH olarak hesaplar.
    test_automation.py'de ThreadPoolExecutor yerine bu kullanılırsa SSL embedding
    GPU'yu tam kapasiteyle doldurur (4-6x hız artışı).

    Args:
        pairs: [(raw1, raw2), ...] listesi
        ssl_batch_size: CodeBERT batch boyutu (GPU belleğine göre ayarlanır)

    Returns: np.ndarray shape (N, F)
    """
    N = len(pairs)

    # 1. Tüm kodları önce önişle (LRU cache'den yararlanır)
    all_raws = []
    for r1, r2 in pairs:
        all_raws.append(r1)
        all_raws.append(r2)

    # 2. SSL embeddings batch (GPU tam kapasite)
    ssl_embs = None
    if ssl_pipeline is not None and ssl_pca is not None:
        ssl_tokenizer, ssl_model = ssl_pipeline
        raw_embs = _extract_batch_ssl_embeddings(all_raws, ssl_tokenizer, ssl_model, batch_size=ssl_batch_size)
        # PCA transform toplu yapılır
        pca_embs = ssl_pca.transform(raw_embs).astype(np.float32)  # (2N, ssl_dim)
        ssl_embs = pca_embs

    # 3. Her çift için feature vektörü hesapla
    results = []
    for idx, (raw1, raw2) in enumerate(pairs):
        code1 = _preprocess_cached(raw1)
        code2 = _preprocess_cached(raw2)

        X1 = vectorizer.transform([code1])
        X2 = vectorizer.transform([code2])
        diff = abs(X1 - X2)
        cos_token = cosine_similarity(X1, X2)[0][0]

        len1 = len(code1.split())
        len2 = len(code2.split())
        length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0

        manhattan = float(diff.sum())
        euclidean = float(np.sqrt(diff.power(2).sum()))
        extra = [cos_token, length_ratio, manhattan, euclidean]

        feat1, cf1, sem1 = _extract_single(raw1)
        feat2, cf2, sem2 = _extract_single(raw2)

        ast_ratios, ast_diffs = [], []
        for v1, v2 in zip(feat1, feat2):
            max_val = max(v1, v2)
            ast_ratios.append(min(v1, v2) / max_val if max_val > 0 else 1.0)
            ast_diffs.append(abs(v1 - v2))
        extra.extend(ast_ratios)
        extra.extend(ast_diffs)

        cf_dist = Levenshtein.distance(cf1, cf2)
        cf_max = max(len(cf1), len(cf2))
        extra.append(1.0 - (cf_dist / cf_max) if cf_max > 0 else 1.0)

        extra.append(_jaccard_sim(sem1['library_calls'], sem2['library_calls']))
        extra.append(_jaccard_sim(sem1['library_categories'], sem2['library_categories']))
        extra.append(_jaccard_sim(sem1['data_structs'], sem2['data_structs']))
        extra.append(_string_bigram_jaccard(sem1['io_pattern'], sem2['io_pattern']))
        extra.append(_jaccard_sim(sem1['math_ops'], sem2['math_ops']))
        extra.append(_tuple_bigram_jaccard(sem1['skeleton'], sem2['skeleton']))

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

        tp1, tp2 = sem1['type_profile'], sem2['type_profile']
        dot = np.dot(tp1, tp2)
        norm = np.linalg.norm(tp1) * np.linalg.norm(tp2)
        extra.append(dot / norm if norm > 0 else 1.0)

        if svd_model is not None:
            svd1 = svd_model.transform(X1)[0]
            svd2 = svd_model.transform(X2)[0]
            extra.extend(np.abs(svd1 - svd2).tolist())

        if ssl_embs is not None:
            e1 = ssl_embs[idx * 2]      # (ssl_dim,)
            e2 = ssl_embs[idx * 2 + 1]  # (ssl_dim,)
            extra.extend(np.abs(e1 - e2).tolist())
            extra.extend((e1 * e2).tolist())
        elif ssl_pipeline is not None and ssl_pca is None:
            ssl_tokenizer, ssl_model = ssl_pipeline
            emb1 = _extract_single_ssl_embedding(raw1, ssl_tokenizer, ssl_model)
            emb2 = _extract_single_ssl_embedding(raw2, ssl_tokenizer, ssl_model)
            dot_s = np.dot(emb1, emb2)
            n1, n2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
            extra.extend([dot_s / (n1 * n2) if (n1 * n2) > 0 else 1.0,
                          float(np.linalg.norm(emb1 - emb2))])

        results.append(extra)

    return np.array(results, dtype=np.float32)  # (N, F)
