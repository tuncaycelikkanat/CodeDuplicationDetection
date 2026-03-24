import numpy as np
from scipy.sparse import hstack, csr_matrix
from tqdm import tqdm
from rapidfuzz.distance import Levenshtein
from joblib import Parallel, delayed

from preprocessing.code_features import cf_pattern_similarity



def generate_pairs(X_token, labels, num_pairs, processed_codes,
                   code_features=None, cf_patterns=None,
                   random_state=42):
    """
    Generate pairs of code samples for clone detection.

    Memory-optimized: generates all pair indices first, then computes
    features in vectorized batches instead of one-by-one sparse matrices.

    Features per pair:
        - |TF-IDF_token_diff|        (sparse)
        - cosine_similarity(token)   (1 feature)
        - length_ratio               (1 feature)
        - AST feature ratios         (5 features, if code_features provided)
        - CF pattern similarity      (1 feature, if cf_patterns provided)
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

    # Decide positive/negative for all pairs at once
    is_positive = np_rng.random(num_pairs) < 0.5
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
    num_hard_neg = int(len(neg_indices_mask) * 0.30)
    num_hard_pos = int(len(pos_indices) * 0.30)

    if num_hard_neg > 0 or num_hard_pos > 0:
        code_lengths = np.array([len(c.split()) for c in processed_codes])
        label_cand_lengths = {}
        label_cand_indices = {}
        for lbl in unique_labels:
            idxs = label_indices_np[lbl]
            label_cand_lengths[lbl] = code_lengths[idxs]
            label_cand_indices[lbl] = idxs

    # ---- Hard Negative Mining: replace 30% of easy negatives ----
    if num_hard_neg > 0:
        print("  → Hard negative mining (vectorized)...")
        hard_slots = np_rng.choice(neg_indices_mask, size=min(num_hard_neg, len(neg_indices_mask)), replace=False)
        src_indices = all_i[hard_slots]
        src_labels_arr = np.array([labels[idx] for idx in src_indices])
        src_lengths = code_lengths[src_indices]

        for k, p in enumerate(hard_slots):
            src_lbl = src_labels_arr[k]
            other_lbl = unique_labels[(unique_labels.index(src_lbl) + np_rng.randint(1, n_labels)) % n_labels]
            cand_lengths = label_cand_lengths[other_lbl]
            closest = np.argmin(np.abs(cand_lengths - src_lengths[k]))
            all_j[p] = label_cand_indices[other_lbl][closest]

    # ---- Hard Positive Mining: replace 30% of easy positives ----
    if num_hard_pos > 0:
        print("  → Hard positive mining (vectorized)...")
        hard_pos_slots = np_rng.choice(pos_indices, size=min(num_hard_pos, len(pos_indices)), replace=False)
        src_indices = all_i[hard_pos_slots]
        src_labels_arr = np.array([labels[idx] for idx in src_indices])
        src_lengths = code_lengths[src_indices]

        for k, p in enumerate(hard_pos_slots):
            src_lbl = src_labels_arr[k]
            cand_lengths = label_cand_lengths[src_lbl]
            cand_indices = label_cand_indices[src_lbl]
            
            # Find the MOST DIFFERENT length (farthest) in the SAME class
            # to force the model to learn from positive structural anomalies (Type-4 Code Clones)
            if len(cand_lengths) > 1:
                farthest = np.argmax(np.abs(cand_lengths - src_lengths[k]))
                all_j[p] = cand_indices[farthest]

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
    extra_cols = [cos_token.reshape(-1, 1),
                  length_ratio.reshape(-1, 1)]
    del cos_token, length_ratio

    # ---- Step 7: AST feature ratios (if provided) ----
    if code_features is not None:
        print("  → Computing AST feature ratios (batch)...")
        cf_i = code_features[all_i]
        cf_j = code_features[all_j]
        max_cf = np.maximum(cf_i, cf_j)
        min_cf = np.minimum(cf_i, cf_j)
        max_cf[max_cf == 0] = 1.0
        ast_ratios = min_cf / max_cf
        extra_cols.append(ast_ratios.astype(np.float32))
        del cf_i, cf_j, max_cf, min_cf, ast_ratios

    # ---- Step 8: Control flow pattern similarity (parallelized) ----
    if cf_patterns is not None:
        print("  → Computing CF pattern similarity (parallel batch)...")

        # Pre-compute bigram sets for all patterns
        def _bigrams(s):
            if not s:
                return set()
            if len(s) < 2:
                return {s}
            return set(s[i:i+2] for i in range(len(s) - 1))

        all_bigrams = [_bigrams(p) for p in cf_patterns]

        def _cf_sim_chunk(start, end, bigrams_list, idx_i, idx_j, patterns):
            result = np.empty(end - start, dtype=np.float32)
            for k in range(end - start):
                p = start + k
                pi, pj = patterns[idx_i[p]], patterns[idx_j[p]]
                if not pi and not pj:
                    result[k] = 1.0
                elif not pi or not pj:
                    result[k] = 0.0
                else:
                    bg1 = bigrams_list[idx_i[p]]
                    bg2 = bigrams_list[idx_j[p]]
                    union = len(bg1 | bg2)
                    result[k] = len(bg1 & bg2) / union if union > 0 else 1.0
            return result

        CHUNK = 100_000
        chunks = [(s, min(s + CHUNK, num_pairs)) for s in range(0, num_pairs, CHUNK)]
        cf_results = Parallel(n_jobs=-1, backend='loky')(
            delayed(_cf_sim_chunk)(s, e, all_bigrams, all_i, all_j, cf_patterns)
            for s, e in chunks
        )
        cf_sim = np.concatenate(cf_results)
        extra_cols.append(cf_sim.reshape(-1, 1))
        del cf_sim, all_bigrams, cf_results

    # ---- Step 10: Combine all features ----
    print("  → Combining features...")
    extra_matrix = csr_matrix(np.hstack(extra_cols))
    del extra_cols

    sparse_parts = [diff_matrix, extra_matrix]

    result = hstack(sparse_parts, format='csr')
    del diff_matrix, extra_matrix
    
    return result, pairs_y
