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

# ================= HELPERS =================

def _jaccard_sim(set_a, set_b):
    if not set_a and not set_b: return 1.0
    if not set_a or not set_b: return 0.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union > 0 else 1.0



def _string_bigram_jaccard(s1, s2):
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    def _bg(s):
        if len(s) < 2: return {s}
        return set(s[i:i+2] for i in range(len(s) - 1))
    bg1, bg2 = _bg(s1), _bg(s2)
    union = len(bg1 | bg2)
    return len(bg1 & bg2) / union if union > 0 else 1.0

def _tuple_bigram_jaccard(t1, t2):
    if not t1 and not t2: return 1.0
    if not t1 or not t2: return 0.0
    def _bg(t):
        if len(t) < 2: return {t}
        return set((t[i], t[i+1]) for i in range(len(t) - 1))
    bg1, bg2 = _bg(t1), _bg(t2)
    union = len(bg1 | bg2)
    return len(bg1 & bg2) / union if union > 0 else 1.0

# ================= PIPELINE =================

def build_pair_vector(raw1, raw2, vectorizer, char_vectorizer=None, svd_model=None):
    """
    Standardized feature extraction for a pair of code snippets.
    Used by both Web Demo and Automation scripts.
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

    manhattan = diff.sum(axis=1)[0, 0]
    euclidean = np.sqrt(diff.power(2).sum(axis=1)[0, 0])

    extra = [cos_token, length_ratio, manhattan, euclidean]

    char_diff = None
    if char_vectorizer is not None:
        C1 = char_vectorizer.transform([code1])
        C2 = char_vectorizer.transform([code2])
        extra.append(cosine_similarity(C1, C2)[0][0])
        char_diff = abs(C1 - C2)

    # AST + CF + Semantic
    feat1, cf1, sem1 = _extract_single(raw1)
    feat2, cf2, sem2 = _extract_single(raw2)

    for v1, v2 in zip(feat1, feat2):
        max_val = max(v1, v2)
        extra.append(min(v1, v2) / max_val if max_val > 0 else 1.0)
        extra.append(abs(v1 - v2))

    cf_dist = Levenshtein.distance(cf1, cf2)
    cf_max = max(len(cf1), len(cf2))
    extra.append(1.0 - (cf_dist / cf_max) if cf_max > 0 else 1.0)

    # Semantic additions
    extra.append(_jaccard_sim(sem1['library_calls'], sem2['library_calls']))
    extra.append(_jaccard_sim(sem1['data_structs'], sem2['data_structs']))
    extra.append(_string_bigram_jaccard(sem1['io_pattern'], sem2['io_pattern']))
    extra.append(_jaccard_sim(sem1['math_ops'], sem2['math_ops']))
    extra.append(_tuple_bigram_jaccard(sem1['skeleton'], sem2['skeleton']))
    
    # Cosine Similarity for Type Profile
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

    extra_matrix = csr_matrix([extra])
    parts = [diff]
    if char_diff is not None: parts.append(char_diff)
    parts.append(extra_matrix)
    
    return hstack(parts)
