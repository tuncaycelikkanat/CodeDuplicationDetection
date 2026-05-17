import numpy as np
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple
from scipy.sparse import hstack, csr_matrix
from tqdm import tqdm
from rapidfuzz.distance import Levenshtein
from joblib import Parallel, delayed

from preprocessing.code_features import cf_pattern_similarity
from utils.similarity_utils import _jaccard_sim, _string_bigram_jaccard, _tuple_bigram_jaccard
from config import HARD_MINING_RATIO


# ================= SEMANTIC SIMILARITY HELPERS =================
# _jaccard_sim, _string_bigram_jaccard, _tuple_bigram_jaccard
# → utils/similarity_utils.py'den import edildi (DRY)


def generate_pairs(
    X_token: csr_matrix,
    labels: List[str],
    num_pairs: int,
    processed_codes: List[str],
    code_features: Optional[np.ndarray] = None,
    cf_patterns: Optional[List[str]] = None,
    semantic_features: Optional[Dict] = None,
    X_svd: Optional[np.ndarray] = None,
    ssl_embeddings: Optional[np.ndarray] = None,
    random_state: int = 42,
    positive_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate pairs of code samples for clone detection.

    Memory-optimized: generates all pair indices first, then computes
    features in vectorized batches.

    Returns a dense NumPy float32 array (N x F) — TF-IDF diff is NOT included.
    Feature order:
        [0]  cosine_similarity(token)
        [1]  length_ratio
        [2]  manhattan_token (L1 diff norm)
        [3]  euclidean_token (L2 diff norm)
        [4..31]  ast_ratios (14) + ast_diffs (14)  -- if code_features provided
        [32] cf_pattern_similarity                  -- if cf_patterns provided
        [33..37] semantic Jaccard/cosine features   -- if semantic_features provided
        [38] type_profile_cosine
        [39..88] SVD diff (50 dims)                 -- if X_svd provided
        [89..152] SSL PCA embedding abs diff        -- if ssl_embeddings provided (64 dims)

    Args:
        positive_ratio: Fraction of pairs that should be positive (clone) pairs.
                        Default 0.5. For realistic evaluation use ~0.05-0.10.
    """


    label_to_indices = {}
    for idx, lbl in enumerate(labels):
        label_to_indices.setdefault(lbl, []).append(idx)

    unique_labels = list(label_to_indices.keys())

    # ---- Step 1: Generate pair indices (vectorized with NumPy) ----
    print("  → Generating pair indices (vectorized)...")
    np_rng = np.random.RandomState(random_state)

    all_i = np.empty(num_pairs, dtype=np.int32)
    all_j = np.empty(num_pairs, dtype=np.int32)
    pairs_y = np.empty(num_pairs, dtype=np.int8)

    # Pre-convert label_to_indices values to numpy arrays
    label_indices_np = {lbl: np.array(idxs, dtype=np.int32) for lbl, idxs in label_to_indices.items()}
    n_labels = len(unique_labels)

    # Decide positive/negative for all pairs at once using configurable ratio
    is_positive = np_rng.random(num_pairs) < positive_ratio
    n_pos = int(is_positive.sum())
    n_neg = num_pairs - n_pos

    pos_indices = np.where(is_positive)[0]
    neg_indices_mask = np.where(~is_positive)[0]

    # Positive pairs: batch by label
    pos_labels_chosen = np_rng.randint(0, n_labels, size=n_pos)
    for li in range(n_labels):
        batch_mask = pos_labels_chosen == li
        count = int(batch_mask.sum())
        if count == 0:
            continue
        idxs = label_indices_np[unique_labels[li]]
        if len(idxs) < 2:
            continue
        picks_i = np_rng.randint(0, len(idxs), size=count)
        picks_j = np_rng.randint(0, len(idxs), size=count)
        same = picks_i == picks_j
        while same.any():
            picks_j[same] = np_rng.randint(0, len(idxs), size=int(same.sum()))
            same = picks_i == picks_j
        slots = pos_indices[batch_mask]
        all_i[slots] = idxs[picks_i]
        all_j[slots] = idxs[picks_j]
    pairs_y[pos_indices] = 1

    # Negative pairs: batch by label pair
    neg_lbl1_idx = np_rng.randint(0, n_labels, size=n_neg)
    neg_lbl2_offset = np_rng.randint(1, n_labels, size=n_neg)
    neg_lbl2_idx = (neg_lbl1_idx + neg_lbl2_offset) % n_labels  # ensures different label
    for li in range(n_labels):
        for lj in range(n_labels):
            if li == lj:
                continue
            batch_mask = (neg_lbl1_idx == li) & (neg_lbl2_idx == lj)
            count = int(batch_mask.sum())
            if count == 0:
                continue
            idxs_i = label_indices_np[unique_labels[li]]
            idxs_j = label_indices_np[unique_labels[lj]]
            slots = neg_indices_mask[batch_mask]
            all_i[slots] = idxs_i[np_rng.randint(0, len(idxs_i), size=count)]
            all_j[slots] = idxs_j[np_rng.randint(0, len(idxs_j), size=count)]
    pairs_y[neg_indices_mask] = 0

    # ---- Hard Mining Preparations ----
    num_hard_neg = int(len(neg_indices_mask) * HARD_MINING_RATIO)
    num_hard_pos = int(len(pos_indices) * HARD_MINING_RATIO)

    if num_hard_neg > 0 or num_hard_pos > 0:
        code_lengths = np.array([len(c.split()) for c in processed_codes])
        label_cand_lengths = {}
        label_cand_indices = {}
        # O(1) label lookup — unique_labels.index() önceden O(n) tariyordu
        label_to_idx: Dict[str, int] = {lbl: i for i, lbl in enumerate(unique_labels)}
        for lbl in unique_labels:
            idxs = label_indices_np[lbl]
            label_cand_lengths[lbl] = code_lengths[idxs]
            label_cand_indices[lbl] = idxs

    # ---- Hard Negative Mining: replace easy negatives ----
    if num_hard_neg > 0:
        print(f"  → Hard negative mining (vectorized) - {num_hard_neg} pairs...")
        hard_slots = np_rng.choice(neg_indices_mask, size=min(num_hard_neg, len(neg_indices_mask)), replace=False)
        src_indices = all_i[hard_slots]
        src_labels_arr = np.array([labels[idx] for idx in src_indices])
        src_lengths = code_lengths[src_indices]

        for k, p in enumerate(hard_slots):
            src_lbl = src_labels_arr[k]
            src_lbl_idx = label_to_idx[src_lbl]  # O(1) — önceden O(n_labels)
            other_lbl = unique_labels[(src_lbl_idx + np_rng.randint(1, n_labels)) % n_labels]
            cand_lengths = label_cand_lengths[other_lbl]
            closest = np.argmin(np.abs(cand_lengths - src_lengths[k]))
            all_j[p] = label_cand_indices[other_lbl][closest]

    # ---- Hard Positive Mining: replace easy positives ----
    if num_hard_pos > 0:
        print(f"  → Hard positive mining (TF-IDF Cosine distance) - {num_hard_pos} pairs...")
        hard_pos_slots = np_rng.choice(pos_indices, size=min(num_hard_pos, len(pos_indices)), replace=False)
        src_indices = all_i[hard_pos_slots]
        src_labels_arr = np.array([labels[idx] for idx in src_indices])

        for k, p in enumerate(hard_pos_slots):
            src_lbl = src_labels_arr[k]
            cand_indices = label_cand_indices[src_lbl]
            
            # Find the LEAST SIMILAR (farthest) in TF-IDF space in the SAME class
            # This simulates Type-4 clones much better than just length differences
            if len(cand_indices) > 1:
                src_vec = X_token[src_indices[k]]
                cand_vecs = X_token[cand_indices]
                
                # Compute dot product (since tf-idf vectors aren't L2 normalized now, we need true cosine)
                # Note: The vectorizer has norm=None now, so we compute cosine sim manually
                dot_prods = src_vec.dot(cand_vecs.T).toarray()[0]
                norm_src = np.sqrt((src_vec.data ** 2).sum())
                norm_cands = np.sqrt(cand_vecs.power(2).sum(axis=1)).A1
                denoms = norm_src * norm_cands
                denoms[denoms == 0] = 1.0
                sims = dot_prods / denoms
                
                # Prevent picking itself
                sims[cand_indices == src_indices[k]] = np.inf
                farthest = np.argmin(sims)
                all_j[p] = cand_indices[farthest]

    # ---- Duplicate Pair Filtering ----
    print("  → Filtering duplicate and self-pairs...")
    # Force i < j for symmetry
    swap_mask = all_i > all_j
    all_i[swap_mask], all_j[swap_mask] = all_j[swap_mask], all_i[swap_mask]
    
    # Remove self-pairs (i == j)
    valid_mask = all_i != all_j
    all_i = all_i[valid_mask]
    all_j = all_j[valid_mask]
    pairs_y = pairs_y[valid_mask]
    
    # Remove exact duplicate (i, j) pairs
    # Using np.unique with axis=0 requires combining them into a 2D array
    pair_coords = np.column_stack((all_i, all_j))
    _, unique_idx = np.unique(pair_coords, axis=0, return_index=True)
    
    # Sort indices to preserve original random order somewhat
    unique_idx.sort()
    all_i = all_i[unique_idx]
    all_j = all_j[unique_idx]
    pairs_y = pairs_y[unique_idx]
    print(f"    - Final unique pairs: {len(all_i)} (dropped {num_pairs - len(all_i)} duplicates)")
    num_pairs = len(all_i)

    # ---- Step 2: Batch token TF-IDF diff (sparse, vectorized) ----
    print("  → Computing token TF-IDF diff (batch)...")
    X_i_token = X_token[all_i]
    X_j_token = X_token[all_j]
    diff_matrix = abs(X_i_token - X_j_token)  # sparse matrix subtraction

    # ---- Step 3: Batch cosine similarity (token) ----
    print("  → Computing token cosine similarity (batch)...")
    cos_token = np.array(X_i_token.multiply(X_j_token).sum(axis=1)).ravel()
    norm_i = np.sqrt(np.array(X_i_token.multiply(X_i_token).sum(axis=1)).ravel())
    norm_j = np.sqrt(np.array(X_j_token.multiply(X_j_token).sum(axis=1)).ravel())
    denom = norm_i * norm_j
    denom[denom == 0] = 1.0
    cos_token = cos_token / denom

    del X_i_token, X_j_token

    # ---- Step 4: Batch length ratio ----
    print("  → Computing length ratio...")
    token_lengths = np.array([len(code.split()) for code in processed_codes], dtype=np.float32)

    len_i = token_lengths[all_i]
    len_j = token_lengths[all_j]
    max_len = np.maximum(len_i, len_j)
    max_len[max_len == 0] = 1.0
    length_ratio = np.minimum(len_i, len_j) / max_len
    del len_i, len_j, max_len, token_lengths

    # ---- Step 5: Build extra features array ----
    manhattan_token = np.array(diff_matrix.sum(axis=1)).ravel()
    euclidean_token = np.sqrt(np.array(diff_matrix.power(2).sum(axis=1)).ravel())
    
    extra_cols = [cos_token.reshape(-1, 1),
                  length_ratio.reshape(-1, 1),
                  manhattan_token.reshape(-1, 1).astype(np.float32),
                  euclidean_token.reshape(-1, 1).astype(np.float32)]
    del cos_token, length_ratio, manhattan_token, euclidean_token

    # ---- Step 7: AST feature ratios (if provided) ----
    if code_features is not None:
        print("  → Computing AST feature ratios (batch)...")
        cf_i = code_features[all_i]
        cf_j = code_features[all_j]
        max_cf = np.maximum(cf_i, cf_j)
        min_cf = np.minimum(cf_i, cf_j)
        max_cf[max_cf == 0] = 1.0
        ast_ratios = min_cf / max_cf
        ast_diffs = np.abs(cf_i - cf_j)
        extra_cols.append(ast_ratios.astype(np.float32))
        extra_cols.append(ast_diffs.astype(np.float32))
        del cf_i, cf_j, max_cf, min_cf, ast_ratios, ast_diffs

    # ---- Step 8: Control flow pattern similarity (parallelized) ----
    if cf_patterns is not None:
        print("  → Computing CF pattern similarity (parallel batch)...")

        def _cf_sim_chunk(start, end, idx_i, idx_j, patterns):
            result = np.empty(end - start, dtype=np.float32)
            for k in range(end - start):
                p = start + k
                pi, pj = patterns[idx_i[p]], patterns[idx_j[p]]
                if not pi and not pj:
                    result[k] = 1.0
                elif not pi or not pj:
                    result[k] = 0.0
                else:
                    dist = Levenshtein.distance(pi, pj)
                    max_len = max(len(pi), len(pj))
                    result[k] = 1.0 - (dist / max_len) if max_len > 0 else 1.0
            return result

        CHUNK = 100_000
        chunks = [(s, min(s + CHUNK, num_pairs)) for s in range(0, num_pairs, CHUNK)]
        cf_results = Parallel(n_jobs=-1, backend='threading')(
            delayed(_cf_sim_chunk)(s, e, all_i, all_j, cf_patterns)
            for s, e in chunks
        )
        cf_sim = np.concatenate(cf_results)
        extra_cols.append(cf_sim.reshape(-1, 1))
        del cf_sim, cf_results

    # ---- Step 9: Semantic similarity features (A1, A2, B3) ----
    if semantic_features is not None:
        print("  → Computing semantic similarity features...")

        lib_calls = semantic_features['library_calls']
        lib_categories = semantic_features['library_categories']
        data_structs = semantic_features['data_structs']
        io_patterns = semantic_features['io_patterns']
        math_ops = semantic_features['math_ops']
        skeletons = semantic_features['skeletons']
        abstract_cf_patterns = semantic_features['abstract_cf_patterns']

        # Vectorize Jaccard/cosine computations using chunked parallel processing
        def _semantic_sim_chunk(start, end, idx_i, idx_j,
                                lib_calls, lib_categories, data_structs, io_patterns,
                                math_ops, skeletons, abstract_cf):
            size = end - start
            result = np.empty((size, 7), dtype=np.float32)
            for k in range(size):
                p = start + k
                ii, jj = idx_i[p], idx_j[p]
                result[k, 0] = _jaccard_sim(lib_calls[ii], lib_calls[jj])
                result[k, 1] = _jaccard_sim(lib_categories[ii], lib_categories[jj])
                result[k, 2] = _jaccard_sim(data_structs[ii], data_structs[jj])
                result[k, 3] = _string_bigram_jaccard(io_patterns[ii], io_patterns[jj])
                result[k, 4] = _jaccard_sim(math_ops[ii], math_ops[jj])
                result[k, 5] = _tuple_bigram_jaccard(skeletons[ii], skeletons[jj])
                
                # Abstract CF Levenshtein
                pi, pj = abstract_cf[ii], abstract_cf[jj]
                if not pi and not pj:
                    result[k, 6] = 1.0
                elif not pi or not pj:
                    result[k, 6] = 0.0
                else:
                    dist = Levenshtein.distance(pi, pj)
                    max_len = max(len(pi), len(pj))
                    result[k, 6] = 1.0 - (dist / max_len) if max_len > 0 else 1.0
                    
            return result

        CHUNK = 100_000
        chunks = [(s, min(s + CHUNK, num_pairs)) for s in range(0, num_pairs, CHUNK)]
        sem_results = Parallel(n_jobs=-1, backend='threading')(
            delayed(_semantic_sim_chunk)(s, e, all_i, all_j,
                                        lib_calls, lib_categories, data_structs, io_patterns,
                                        math_ops, skeletons, abstract_cf_patterns)
            for s, e in chunks
        )
        sem_matrix = np.vstack(sem_results)
        extra_cols.append(sem_matrix)
        del sem_results, sem_matrix

        # Type Profile Cosine Similarity (Vectorized)
        print("  → Computing type profile cosine similarity...")
        type_profiles_mat = np.vstack(semantic_features['type_profiles'])
        tp_i = type_profiles_mat[all_i]
        tp_j = type_profiles_mat[all_j]
        dot = np.sum(tp_i * tp_j, axis=1)
        norm_i = np.linalg.norm(tp_i, axis=1)
        norm_j = np.linalg.norm(tp_j, axis=1)
        denom = norm_i * norm_j
        denom[denom == 0] = 1.0
        # For perfectly empty profiles, cosine sim is 1.0
        tp_cos = np.where(denom == 0, 1.0, dot / denom)
        extra_cols.append(tp_cos.reshape(-1, 1).astype(np.float32))
        del type_profiles_mat, tp_i, tp_j, dot, norm_i, norm_j, denom, tp_cos

    # ---- Step 9.5: SVD Differences (if provided) ----
    if X_svd is not None:
        print("  → Computing SVD differences (batch)...")
        svd_i = X_svd[all_i]
        svd_j = X_svd[all_j]
        svd_diff = np.abs(svd_i - svd_j)
        extra_cols.append(svd_diff.astype(np.float32))
        del svd_i, svd_j, svd_diff

    # ---- Step 9.6: SSL Embeddings Abs Diff (if provided) ----
    # ssl_embeddings burada zaten PCA ile indirgenmis (N, SSL_PCA_COMPONENTS) formatindadir.
    # 2 skaler (cos+euclidean) yerine tam abs fark vektoru kullaniliyor:
    # model her boyuttaki farki ayri ayri ogrenir -> Type-4 icin cok daha zengin sinyal.
    if ssl_embeddings is not None:
        print("  → Computing SSL embedding abs diff (batch)...")
        ssl_i = ssl_embeddings[all_i]   # (num_pairs, ssl_dim)
        ssl_j = ssl_embeddings[all_j]   # (num_pairs, ssl_dim)
        ssl_abs_diff = np.abs(ssl_i - ssl_j).astype(np.float32)
        extra_cols.append(ssl_abs_diff)
        del ssl_i, ssl_j, ssl_abs_diff

    # ---- Step 10: Combine all features ----
    print("  → Combining features...")
    # We no longer append the 500 TF-IDF differences. We return only the dense extra features.
    result = np.hstack(extra_cols).astype(np.float32)
    del extra_cols, diff_matrix
    
    return result, pairs_y

