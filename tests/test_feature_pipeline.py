"""
Unit testler: utils/feature_pipeline.py
Güncel dense mimariyle uyumlu — sparse matris artık döndürülmüyor.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from utils.logger import Log

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from utils.feature_pipeline import build_pair_vector
from preprocessing.code_features import FEATURE_NAMES
from config import ENSEMBLE_SVD_START_IDX, SVD_N_COMPONENTS


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

CODE_SHORT = "int x;"
CODE_EMPTY = ""


@pytest.fixture(scope="module")
def fitted_vectorizer():
    """Kucuk corpus uzerinde fit edilmis TF-IDF."""
    from preprocessing.tokenizer import tokenize, normalize_tokens
    corpus = [" ".join(normalize_tokens(tokenize(c))) for c in [CODE_A, CODE_B, CODE_C]]
    vec = TfidfVectorizer(
        min_df=1, max_df=0.99, ngram_range=(1, 3),
        token_pattern=r"[^ ]+", sublinear_tf=True, max_features=500
    )
    vec.fit(corpus)
    return vec


@pytest.fixture(scope="module")
def fitted_svd(fitted_vectorizer):
    """TF-IDF uzerine fit edilmis TruncatedSVD (10 bilesen)."""
    from preprocessing.tokenizer import tokenize, normalize_tokens
    corpus = [" ".join(normalize_tokens(tokenize(c))) for c in [CODE_A, CODE_B, CODE_C]]
    X = fitted_vectorizer.transform(corpus)
    svd = TruncatedSVD(n_components=10, random_state=42)
    svd.fit(X)
    return svd


class TestBuildPairVector:
    def test_returns_numpy_array(self, fitted_vectorizer):
        """feature_pipeline artik dense numpy array donduruyor (sparse degil)."""
        result = build_pair_vector(CODE_A, CODE_B, fitted_vectorizer)
        assert isinstance(result, np.ndarray), (
            f"Expected np.ndarray, got {type(result)}."
        )

    def test_single_row(self, fitted_vectorizer):
        result = build_pair_vector(CODE_A, CODE_B, fitted_vectorizer)
        assert result.shape[0] == 1

    def test_column_count_without_char_or_svd(self, fitted_vectorizer):
        """
        Beklenen sutun sayisi (char_vectorizer=None, svd=None):
            cos_token + length_ratio + manhattan + euclidean = 4
            ast_ratios(N) + ast_diffs(N)                    = 2*N
            cf_sim                                          = 1
            semantic (lib, lib_cat, data_struct, io, math, skeleton, abstract_cf) = 7
            type_profile_cosine                             = 1
        Toplam: 13 + 2*N
        """
        result = build_pair_vector(CODE_A, CODE_B, fitted_vectorizer)
        n_ast = len(FEATURE_NAMES)
        expected_cols = 4 + (2 * n_ast) + 1 + 7 + 1
        assert result.shape[1] == expected_cols, (
            f"Expected {expected_cols} cols, got {result.shape[1]}. "
            f"FEATURE_NAMES count: {n_ast}"
        )

    def test_column_count_with_svd(self, fitted_vectorizer, fitted_svd):
        """SVD model verilince sutun sayisi svd transform ciktisi kadar artar."""
        result_no_svd = build_pair_vector(CODE_A, CODE_B, fitted_vectorizer)
        result_svd    = build_pair_vector(CODE_A, CODE_B, fitted_vectorizer, svd_model=fitted_svd)
        actual_diff = result_svd.shape[1] - result_no_svd.shape[1]
        # Kucuk corpus'ta SVD transform n_components'tan kucuk olabilir (rank kisitlamasi)
        from preprocessing.tokenizer import tokenize, normalize_tokens
        corpus = [" ".join(normalize_tokens(tokenize(c))) for c in [CODE_A, CODE_B, CODE_C]]
        X_test = fitted_vectorizer.transform(corpus)
        expected_svd_cols = fitted_svd.transform(X_test).shape[1]
        assert actual_diff == expected_svd_cols, (
            f"SVD eklince {expected_svd_cols} yeni sutun bekleniyor, {actual_diff} alindi."
        )

    def test_float32_dtype(self, fitted_vectorizer):
        """Modeller float32 giris bekliyor."""
        result = build_pair_vector(CODE_A, CODE_B, fitted_vectorizer)
        assert result.dtype == np.float32

    def test_cos_token_at_index_zero(self, fitted_vectorizer):
        """Cascade filtresi cos_token'u her zaman index 0'dan okuyor."""
        result = build_pair_vector(CODE_A, CODE_A, fitted_vectorizer)
        cos_token = float(result[0, 0])
        assert cos_token > 0.99, (
            f"Ayni kod icin cos_token > 0.99 bekleniyor, {cos_token:.4f} alindi"
        )

    def test_identical_codes_high_similarity(self, fitted_vectorizer):
        """Ayni kodu kendi kendisiyle karsilastirinca cos_token ~1.0 olmali."""
        result = build_pair_vector(CODE_A, CODE_A, fitted_vectorizer)
        assert float(result[0, 0]) > 0.99

    def test_different_codes_lower_similarity(self, fitted_vectorizer):
        """Farkli kodlarin cos_token'i ayni koddan kucuk olmali."""
        result_same = build_pair_vector(CODE_A, CODE_A, fitted_vectorizer)
        result_diff = build_pair_vector(CODE_A, CODE_C, fitted_vectorizer)
        assert float(result_same[0, 0]) > float(result_diff[0, 0])

    def test_values_in_valid_range(self, fitted_vectorizer):
        """cos_token ve length_ratio [0, 1] araliginda olmali."""
        result = build_pair_vector(CODE_A, CODE_B, fitted_vectorizer)
        assert 0.0 <= float(result[0, 0]) <= 1.0
        assert 0.0 <= float(result[0, 1]) <= 1.0

    def test_symmetry(self, fitted_vectorizer):
        """(A,B) ve (B,A) ciftlerinin cos_token ve length_ratio degerleri ayni olmali."""
        ab = build_pair_vector(CODE_A, CODE_B, fitted_vectorizer)
        ba = build_pair_vector(CODE_B, CODE_A, fitted_vectorizer)
        assert abs(float(ab[0, 0]) - float(ba[0, 0])) < 1e-4
        assert abs(float(ab[0, 1]) - float(ba[0, 1])) < 1e-4

    def test_short_code_no_crash(self, fitted_vectorizer):
        """Cok kisa kod (tek satir) parse edilebilmeli."""
        result = build_pair_vector(CODE_SHORT, CODE_A, fitted_vectorizer)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1

    def test_empty_code_no_crash(self, fitted_vectorizer):
        """Bos kod ile karsilastirinca sifir bolme hatasi olmamali."""
        try:
            result = build_pair_vector(CODE_EMPTY, CODE_A, fitted_vectorizer)
            assert result.shape[0] == 1
        except Exception as e:
            pytest.fail(f"Bos kod exception firlatti: {e}")
