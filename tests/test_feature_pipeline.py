"""
Unit testler: utils/feature_pipeline.py
Güncel dense mimariyle uyumlu — sparse matris artık döndürülmüyor.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.feature_pipeline import build_pair_vector
from preprocessing.code_features import FEATURE_NAMES


CODE_A = """
int sum(int arr[], int n) {
    int total = 0;
    for (int i = 0; i < n; i++)
        total += arr[i];
    return total;
}
"""

CODE_B = """
int sumRec(int arr[], int n) {
    if (n <= 0) return 0;
    return sumRec(arr, n - 1) + arr[n - 1];
}
"""

CODE_C = """
void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n-1; i++)
        for (int j = 0; j < n-i-1; j++)
            if (arr[j] > arr[j+1]) {
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
}
"""


@pytest.fixture(scope="module")
def fitted_vectorizer():
    """Küçük corpus üzerinde fit edilmiş TF-IDF — min_df=1 (test corpus < 3 belge)."""
    from preprocessing.tokenizer import tokenize, normalize_tokens
    corpus = [" ".join(normalize_tokens(tokenize(c))) for c in [CODE_A, CODE_B, CODE_C]]
    vec = TfidfVectorizer(
        min_df=1, max_df=0.99, ngram_range=(1, 3),
        token_pattern=r"[^ ]+", sublinear_tf=True, max_features=500
    )
    vec.fit(corpus)
    return vec


class TestBuildPairVector:
    def test_returns_numpy_array(self, fitted_vectorizer):
        """feature_pipeline artık dense numpy array döndürüyor (sparse değil)."""
        result = build_pair_vector(CODE_A, CODE_B, fitted_vectorizer)
        assert isinstance(result, np.ndarray), (
            f"Expected np.ndarray, got {type(result)}. "
            "build_pair_vector artık dense array döndürüyor."
        )

    def test_single_row(self, fitted_vectorizer):
        result = build_pair_vector(CODE_A, CODE_B, fitted_vectorizer)
        assert result.shape[0] == 1

    def test_column_count_without_char_or_svd(self, fitted_vectorizer):
        """
        Beklenen sütun sayısı (TF-IDF diff artık YOK):
            cos_token + length_ratio + manhattan + euclidean = 4
            ast_ratios(18) + ast_diffs(18) = 36  ← FEATURE_NAMES şimdi 18 feature
            cf_sim = 1
            semantic_jaccard x5 = 5
            type_profile_cosine = 1
            SVD diff = 0 (svd_model=None)
        Toplam: 4 + 36 + 1 + 5 + 1 = 47
        Not: FEATURE_NAMES uzunluğuna bağlıdır — kod_features değişirse bu test güncellenir.
        """
        result = build_pair_vector(CODE_A, CODE_B, fitted_vectorizer)
        n_ast = len(FEATURE_NAMES)  # ratio + diff = 2 * n_ast
        expected_cols = 4 + (2 * n_ast) + 1 + 5 + 1
        assert result.shape[1] == expected_cols, (
            f"Expected {expected_cols} cols, got {result.shape[1]}. "
            f"FEATURE_NAMES count: {n_ast}"
        )

    def test_float32_dtype(self, fitted_vectorizer):
        """Modeller float32 giriş bekliyor."""
        result = build_pair_vector(CODE_A, CODE_B, fitted_vectorizer)
        assert result.dtype == np.float32

    def test_cos_token_at_index_zero(self, fitted_vectorizer):
        """Cascade filtresi cos_token'u her zaman index 0'dan okuyor."""
        result = build_pair_vector(CODE_A, CODE_A, fitted_vectorizer)
        cos_token = float(result[0, 0])
        assert cos_token > 0.99, (
            f"Aynı kod için cos_token > 0.99 bekleniyor, {cos_token:.4f} alındı"
        )

    def test_identical_codes_high_similarity(self, fitted_vectorizer):
        """Aynı kodu kendi kendisiyle karşılaştırınca cos_token ~1.0 olmalı."""
        result = build_pair_vector(CODE_A, CODE_A, fitted_vectorizer)
        cos_token = float(result[0, 0])
        assert cos_token > 0.99

    def test_different_codes_lower_similarity(self, fitted_vectorizer):
        """Farklı kodların cos_token'ı aynı koddan küçük olmalı."""
        result_same = build_pair_vector(CODE_A, CODE_A, fitted_vectorizer)
        result_diff = build_pair_vector(CODE_A, CODE_C, fitted_vectorizer)
        cos_same = float(result_same[0, 0])
        cos_diff = float(result_diff[0, 0])
        assert cos_same > cos_diff

    def test_values_in_valid_range(self, fitted_vectorizer):
        """cos_token ve length_ratio [0, 1] aralığında olmalı."""
        result = build_pair_vector(CODE_A, CODE_B, fitted_vectorizer)
        cos_token = float(result[0, 0])
        length_ratio = float(result[0, 1])
        assert 0.0 <= cos_token <= 1.0
        assert 0.0 <= length_ratio <= 1.0

    def test_symmetry(self, fitted_vectorizer):
        """
        (A,B) ve (B,A) çiftlerinin cos_token ve length_ratio değerleri aynı olmalı.
        (Simetrik metrikler)
        """
        ab = build_pair_vector(CODE_A, CODE_B, fitted_vectorizer)
        ba = build_pair_vector(CODE_B, CODE_A, fitted_vectorizer)
        assert abs(float(ab[0, 0]) - float(ba[0, 0])) < 1e-4  # cos_token
        assert abs(float(ab[0, 1]) - float(ba[0, 1])) < 1e-4  # length_ratio
