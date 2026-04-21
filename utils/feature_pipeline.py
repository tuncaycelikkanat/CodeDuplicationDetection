import os
import sys
import math
import numpy as np
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

def build_pair_vector(raw1, raw2, vectorizer, char_vectorizer=None, svd_model=None):
    """
    Standart feature extraction — bir çift kod snippet için.
    Web Demo ve Automation scriptleri tarafından kullanılır.

    Feature sırası (pair_generator.py ile birebir eşleştirilmiştir):
        diff_matrix       : |TF-IDF(token)_i - TF-IDF(token)_j|  (TOKEN_FEAT_COUNT sütun)
        [char_diff]       : |TF-IDF(char)_i  - TF-IDF(char)_j|   (sadece char_vectorizer varsa)
        cos_token         : token cosine similarity                 (1)
        length_ratio      : min/max token uzunluk oranı            (1)
        manhattan_token   : L1 norm of token diff                  (1)
        euclidean_token   : L2 norm of token diff                  (1)
        [cos_char]        : char cosine similarity                  (1, sadece char_vectorizer varsa)
        ast_ratios        : min/max AST feature oranları           (14)
        ast_diffs         : |AST_i - AST_j|                       (14)
        cf_sim            : control-flow pattern similarity         (1)
        lib_call_jaccard  : library call Jaccard                   (1)
        data_struct_jaccard: data struct Jaccard                   (1)
        io_pattern_jaccard: I/O pattern bigram Jaccard             (1)
        math_op_jaccard   : math op Jaccard                        (1)
        skeleton_jaccard  : skeleton bigram Jaccard                (1)
        type_profile_cos  : type profile cosine                    (1)
        [svd_diff]        : |SVD(token)_i - SVD(token)_j|         (SVD_N_COMPONENTS, sadece svd_model varsa)
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

    # Semantic features (5)
    extra.append(_jaccard_sim(sem1['library_calls'], sem2['library_calls']))
    extra.append(_jaccard_sim(sem1['data_structs'], sem2['data_structs']))
    extra.append(_string_bigram_jaccard(sem1['io_pattern'], sem2['io_pattern']))
    extra.append(_jaccard_sim(sem1['math_ops'], sem2['math_ops']))
    extra.append(_tuple_bigram_jaccard(sem1['skeleton'], sem2['skeleton']))

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

    extra_matrix = csr_matrix([extra])
    parts = [diff]
    if char_diff is not None:
        parts.append(char_diff)
    parts.append(extra_matrix)

    return hstack(parts)
