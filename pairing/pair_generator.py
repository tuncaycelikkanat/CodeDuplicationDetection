import numpy as np
from scipy.sparse import hstack, csr_matrix
from tqdm import tqdm
from rapidfuzz.distance import Levenshtein
from joblib import Parallel, delayed

from preprocessing.code_features import cf_pattern_similarity



def generate_pairs(X_token, labels, num_pairs, processed_codes,
                   X_char=None, code_features=None, cf_patterns=None,
                   raw_codes=None, random_state=42):
    """
    Generate pairs of code samples for clone detection.

    Memory-optimized: generates all pair indices first, then computes
    features in vectorized batches instead of one-by-one sparse matrices.

    Features per pair:
        - |TF-IDF_token_diff|        (sparse)
        - cosine_similarity(token)   (1 feature)
        - token_overlap (Jaccard)    (1 feature)
        - length_ratio               (1 feature)
        - cosine_similarity(char)    (1 feature, if X_char provided)
        - |TF-IDF_char_diff|         (sparse, if X_char provided)
        - AST feature ratios         (5 features, if code_features provided)
        - CF pattern similarity      (1 feature, if cf_patterns provided)
        - edit_distance_ratio        (1 feature, if raw_codes provided)
        - line_count_ratio           (1 feature, if raw_codes provided)
        - char_length_ratio          (1 feature, if raw_codes provided)
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

    # ---- Hard Negative Mining: replace 30% of easy negatives ----
    print("  → Hard negative mining (vectorized)...")
    neg_indices = neg_indices_mask
    num_hard = int(len(neg_indices) * 0.30)

    if num_hard > 0:
        code_lengths = np.array([len(c.split()) for c in processed_codes])
        hard_slots = np_rng.choice(neg_indices, size=min(num_hard, len(neg_indices)), replace=False)

        # Vectorized hard negative mining
        src_indices = all_i[hard_slots]
        src_labels_arr = np.array([labels[idx] for idx in src_indices])
        src_lengths = code_lengths[src_indices]

        # Pre-compute per-label candidate lengths for fast lookup
        label_cand_lengths = {}
        label_cand_indices = {}
        for lbl in unique_labels:
            idxs = label_indices_np[lbl]
            label_cand_lengths[lbl] = code_lengths[idxs]
            label_cand_indices[lbl] = idxs

        for k, p in enumerate(hard_slots):
            src_lbl = src_labels_arr[k]
            other_lbl = unique_labels[(unique_labels.index(src_lbl) + np_rng.randint(1, n_labels)) % n_labels]
            cand_lengths = label_cand_lengths[other_lbl]
            closest = np.argmin(np.abs(cand_lengths - src_lengths[k]))
            all_j[p] = label_cand_indices[other_lbl][closest]

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

    # ---- Step 4: Batch Jaccard overlap & length ratio (parallelized) ----
    print("  → Computing Jaccard overlap & length ratio (parallel batch)...")
    token_sets = [set(code.split()) for code in processed_codes]
    token_lengths = np.array([len(code.split()) for code in processed_codes], dtype=np.float32)

    def _jaccard_chunk(start, end, sets_list, idx_i, idx_j):
        result = np.empty(end - start, dtype=np.float32)
        for k in range(end - start):
            p = start + k
            si, sj = sets_list[idx_i[p]], sets_list[idx_j[p]]
            union = len(si | sj)
            result[k] = len(si & sj) / union if union > 0 else 0.0
        return result

    CHUNK = 100_000
    chunks = [(s, min(s + CHUNK, num_pairs)) for s in range(0, num_pairs, CHUNK)]
    jaccard_results = Parallel(n_jobs=-1, backend='loky')(
        delayed(_jaccard_chunk)(s, e, token_sets, all_i, all_j)
        for s, e in chunks
    )
    overlap_arr = np.concatenate(jaccard_results)

    del token_sets, jaccard_results

    len_i = token_lengths[all_i]
    len_j = token_lengths[all_j]
    max_len = np.maximum(len_i, len_j)
    max_len[max_len == 0] = 1.0
    length_ratio = np.minimum(len_i, len_j) / max_len
    del len_i, len_j, max_len, token_lengths

    # ---- Step 5: Build extra features array ----
    extra_cols = [cos_token.reshape(-1, 1),
                  overlap_arr.reshape(-1, 1),
                  length_ratio.reshape(-1, 1)]
    del cos_token, overlap_arr, length_ratio

    # ---- Step 6: Batch char TF-IDF cosine + diff (if provided) ----
    if X_char is not None:
        print("  → Computing char cosine similarity + diff (batch)...")
        X_i_char = X_char[all_i]
        X_j_char = X_char[all_j]
        cos_char = np.array(X_i_char.multiply(X_j_char).sum(axis=1)).ravel()
        norm_ci = np.sqrt(np.array(X_i_char.multiply(X_i_char).sum(axis=1)).ravel())
        norm_cj = np.sqrt(np.array(X_j_char.multiply(X_j_char).sum(axis=1)).ravel())
        denom_c = norm_ci * norm_cj
        denom_c[denom_c == 0] = 1.0
        cos_char = cos_char / denom_c
        extra_cols.append(cos_char.reshape(-1, 1))

        # Char TF-IDF diff (sparse) — will be added to final matrix
        char_diff_matrix = abs(X_i_char - X_j_char)

        del X_i_char, X_j_char, cos_char, norm_ci, norm_cj, denom_c
    else:
        char_diff_matrix = None

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

    # ---- Step 9: New features — edit distance, line count, char length ----
    if raw_codes is not None:
        print("  → Computing edit distance ratio (parallel batch)...")
        line_counts = np.array([c.count('\n') + 1 for c in raw_codes], dtype=np.float32)
        char_lengths = np.array([len(c) for c in raw_codes], dtype=np.float32)

        # Pre-truncate codes for edit distance (300 chars for speed)
        truncated_codes = [c[:200] for c in raw_codes]

        def _edit_dist_chunk(start, end, codes, idx_i, idx_j):
            result = np.empty(end - start, dtype=np.float32)
            for k in range(end - start):
                p = start + k
                result[k] = Levenshtein.normalized_similarity(
                    codes[idx_i[p]], codes[idx_j[p]]
                )
            return result

        CHUNK = 100_000
        chunks = [(s, min(s + CHUNK, num_pairs)) for s in range(0, num_pairs, CHUNK)]
        edit_results = Parallel(n_jobs=-1, backend='loky')(
            delayed(_edit_dist_chunk)(s, e, truncated_codes, all_i, all_j)
            for s, e in chunks
        )
        edit_dist = np.concatenate(edit_results)
        extra_cols.append(edit_dist.reshape(-1, 1))
        del edit_dist, truncated_codes, edit_results

        # Line count ratio
        lc_i = line_counts[all_i]
        lc_j = line_counts[all_j]
        max_lc = np.maximum(lc_i, lc_j)
        max_lc[max_lc == 0] = 1.0
        line_ratio = np.minimum(lc_i, lc_j) / max_lc
        extra_cols.append(line_ratio.reshape(-1, 1))
        del lc_i, lc_j, max_lc, line_ratio, line_counts

        # Char length ratio
        cl_i = char_lengths[all_i]
        cl_j = char_lengths[all_j]
        max_cl = np.maximum(cl_i, cl_j)
        max_cl[max_cl == 0] = 1.0
        char_ratio = np.minimum(cl_i, cl_j) / max_cl
        extra_cols.append(char_ratio.reshape(-1, 1))
        del cl_i, cl_j, max_cl, char_ratio, char_lengths

    # ---- Step 10: Combine all features ----
    print("  → Combining features...")
    extra_matrix = csr_matrix(np.hstack(extra_cols))
    del extra_cols

    sparse_parts = [diff_matrix]
    if char_diff_matrix is not None:
        sparse_parts.append(char_diff_matrix)
    sparse_parts.append(extra_matrix)

    result = hstack(sparse_parts, format='csr')
    del diff_matrix, char_diff_matrix, extra_matrix

    return result, pairs_y
