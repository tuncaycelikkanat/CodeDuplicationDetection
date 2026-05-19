"""
Edge Case Testleri - CodeDuplicationDetection
=============================================
Sinir durumlarini ve regresyon senaryolarini test eder:
  - positive_ratio=0.0 (yalnizca negatifler)
  - extract_all_features bos/kisa kod toleransi
  - generate_test_clones determinizm testi
  - Stage-1 threshold degisince sistem davranisi
  - cascade filter helper fonksiyonu
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
from utils.logger import Log

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier

from preprocessing.tokenizer import tokenize, normalize_tokens
from preprocessing.code_features import extract_all_features
from pairing.pair_generator import generate_pairs
from config import CASCADE_STAGE1_THRESHOLD, STAGE1_FEATURE_COUNT


# --- Corpus: Her etiket en az 2 ornege sahip olmali (hard mining zorunlulugu) ---
# Etiket = problem sinifi (POJ-104 benzeri)
# Her sinif 2 kod iceriyor: biri orijinal, biri semantik e-deger (Type-4 klon gibi)

ALL_CODES = [
    # sum (label=0)
    "int sum(int a[], int n){int s=0;for(int i=0;i<n;i++)s+=a[i];return s;}",
    "int total(int arr[], int sz){int r=0;for(int i=0;i<sz;i++)r+=arr[i];return r;}",
    # fact (label=1)
    "int factorial(int n){if(n<=1)return 1;return n*factorial(n-1);}",
    "int fact(int n){int r=1;for(int i=2;i<=n;i++)r*=i;return r;}",
    # swap (label=2)
    "void swap(int* a,int* b){int t=*a;*a=*b;*b=t;}",
    "void exchange(int* x,int* y){int tmp=*x;*x=*y;*y=tmp;}",
    # bsearch (label=3)
    "int bs(int a[],int l,int r,int x){while(l<=r){int m=l+(r-l)/2;if(a[m]==x)return m;if(a[m]<x)l=m+1;else r=m-1;}return -1;}",
    "int bsr(int a[],int l,int r,int x){if(r<l)return -1;int m=l+(r-l)/2;if(a[m]==x)return m;if(a[m]<x)return bsr(a,m+1,r,x);return bsr(a,l,m-1,x);}",
]
# Her cift ard arda gelen 2 eleman ayni siniftan: [0,0,1,1,2,2,3,3]
LABELS = ["sum", "sum", "fact", "fact", "swap", "swap", "bsearch", "bsearch"]


@pytest.fixture(scope="module")
def small_pipeline():
    processed = [" ".join(normalize_tokens(tokenize(c))) for c in ALL_CODES]
    vec = TfidfVectorizer(min_df=1, max_df=1.0, ngram_range=(1, 2),
                          token_pattern=r"[^ ]+", sublinear_tf=True, max_features=100)
    X_tok = vec.fit_transform(processed)
    svd = TruncatedSVD(n_components=5, random_state=42)
    X_svd = svd.fit_transform(X_tok)
    cf, cp, sf = extract_all_features(ALL_CODES)
    return {"processed": processed, "vectorizer": vec,
            "X_token": X_tok, "X_svd": X_svd,
            "code_features": cf, "cf_patterns": cp, "sem_features": sf}


# --- positive_ratio edge cases -------------------------------------------

class TestPositiveRatioEdgeCases:
    def test_zero_ratio_all_negatives(self, small_pipeline):
        """positive_ratio=0.0 ile tum ciftler negatif olmali."""
        X, y = generate_pairs(
            small_pipeline["X_token"], LABELS, 30, small_pipeline["processed"],
            code_features=small_pipeline["code_features"],
            cf_patterns=small_pipeline["cf_patterns"],
            semantic_features=small_pipeline["sem_features"],
            X_svd=small_pipeline["X_svd"],
            random_state=42, positive_ratio=0.0
        )
        assert y.sum() == 0, f"positive_ratio=0.0 ile pozitif cift olmamali, {y.sum()} alindi"

    def test_half_ratio_has_positives(self, small_pipeline):
        """positive_ratio=0.5 ile en az bir pozitif cift olmali."""
        X, y = generate_pairs(
            small_pipeline["X_token"], LABELS, 30, small_pipeline["processed"],
            code_features=small_pipeline["code_features"],
            cf_patterns=small_pipeline["cf_patterns"],
            semantic_features=small_pipeline["sem_features"],
            X_svd=small_pipeline["X_svd"],
            random_state=42, positive_ratio=0.5
        )
        assert y.sum() > 0, "positive_ratio=0.5 ile pozitif cift olmali"
        assert y.sum() < len(y), "positive_ratio=0.5 ile hepsi pozitif olmamali"


# --- extract_all_features bos/kisa kod toleransi -------------------------

class TestFeatureExtractionEdgeCases:
    def test_empty_string(self):
        """Bos kod string'i feature extraction'i patlatmamali."""
        try:
            features, cf, sem = extract_all_features([""])
            assert features.shape[0] == 1
        except Exception as e:
            pytest.fail(f"Bos string exception firlatti: {e}")

    def test_very_short_code(self):
        """Tek satir kod parse edilebilmeli."""
        try:
            features, cf, sem = extract_all_features(["int x;"])
            assert features.shape[0] == 1
        except Exception as e:
            pytest.fail(f"Cok kisa kod exception firlatti: {e}")

    def test_mixed_valid_invalid(self):
        """Gecerli ve bos kodlarin karisimi toplam sayiyi koruyor olmali."""
        codes = ["int sum(int a){return a;}", "", "void f(){}"]
        features, cf, sem = extract_all_features(codes)
        assert features.shape[0] == len(codes), (
            f"Beklenen {len(codes)} satir, {features.shape[0]} alindi"
        )


# --- Determinizm testi ---------------------------------------------------

class TestDeterminism:
    def test_generate_pairs_deterministic(self, small_pipeline):
        """Ayni seed ile generate_pairs ayni y dizisini uretmeli."""
        kw = dict(X_svd=small_pipeline["X_svd"], random_state=123, positive_ratio=0.5)
        X1, y1 = generate_pairs(small_pipeline["X_token"], LABELS, 20,
                                 small_pipeline["processed"], **kw)
        X2, y2 = generate_pairs(small_pipeline["X_token"], LABELS, 20,
                                 small_pipeline["processed"], **kw)
        np.testing.assert_array_equal(y1, y2, err_msg="Ayni seed ile y degerleri farkli!")
        np.testing.assert_array_almost_equal(
            X1[:, 0], X2[:, 0], decimal=5,
            err_msg="Ayni seed ile cos_token degerleri farkli!"
        )

    def test_different_seeds_differ(self, small_pipeline):
        """Farkli seed'ler farkli pair siralari uretmeli."""
        _, y1 = generate_pairs(small_pipeline["X_token"], LABELS, 24,
                                small_pipeline["processed"],
                                X_svd=small_pipeline["X_svd"],
                                random_state=1, positive_ratio=0.5)
        _, y2 = generate_pairs(small_pipeline["X_token"], LABELS, 24,
                                small_pipeline["processed"],
                                X_svd=small_pipeline["X_svd"],
                                random_state=2, positive_ratio=0.5)
        assert not np.array_equal(y1, y2), "Farkli seed'ler ayni y'yi uretmemeli"


# --- Cascade Filter Helper -----------------------------------------------

class TestCascadeFilterHelper:
    def _make_pairs(self, small_pipeline, n=40, seed=42):
        X, y = generate_pairs(
            small_pipeline["X_token"], LABELS, n, small_pipeline["processed"],
            code_features=small_pipeline["code_features"],
            cf_patterns=small_pipeline["cf_patterns"],
            semantic_features=small_pipeline["sem_features"],
            X_svd=small_pipeline["X_svd"],
            random_state=seed, positive_ratio=0.5
        )
        return X.astype(np.float32), y

    def _make_stage1(self, X, y):
        stage1 = HistGradientBoostingClassifier(max_iter=20, random_state=42)
        stage1.fit(X[:, :STAGE1_FEATURE_COUNT], y)
        return stage1

    def test_filter_removes_easy_positives(self, small_pipeline):
        """_apply_cascade_filter kolay klonlari kaldirmali."""
        from main import _apply_cascade_filter
        X, y = self._make_pairs(small_pipeline)
        stage1 = self._make_stage1(X, y)
        X_filt, y_filt, n_removed = _apply_cascade_filter(X, y, stage1)
        assert X_filt.shape[0] == len(y_filt), "X ve y boyutlari uyumsuz"
        assert X_filt.shape[0] <= X.shape[0], "Filtreleme sonrasi boyut artmamali"
        assert n_removed >= 0
        assert n_removed == (X.shape[0] - X_filt.shape[0])

    def test_filter_with_threshold_one(self, small_pipeline):
        """threshold=1.0 ile hicbir ornek kaldirilmamali."""
        from main import _apply_cascade_filter
        X, y = self._make_pairs(small_pipeline, seed=88)
        stage1 = self._make_stage1(X, y)
        X_filt, _, n_removed = _apply_cascade_filter(X, y, stage1, threshold=1.0)
        assert X_filt.shape[0] == X.shape[0], "threshold=1.0 ile hic eleman kaldirilmamali"
        assert n_removed == 0


# --- Stage-1 Threshold Davranisi -----------------------------------------

class TestCascadeThresholdBehavior:
    def test_lower_threshold_removes_more(self, small_pipeline):
        """Dusuk threshold daha fazla kolay klon kaldirmali."""
        from main import _apply_cascade_filter

        X, y = generate_pairs(
            small_pipeline["X_token"], LABELS, 40, small_pipeline["processed"],
            code_features=small_pipeline["code_features"],
            cf_patterns=small_pipeline["cf_patterns"],
            semantic_features=small_pipeline["sem_features"],
            X_svd=small_pipeline["X_svd"],
            random_state=42, positive_ratio=0.5
        )
        X = X.astype(np.float32)
        stage1 = HistGradientBoostingClassifier(max_iter=30, random_state=42)
        stage1.fit(X[:, :STAGE1_FEATURE_COUNT], y)

        _, _, n_high = _apply_cascade_filter(X, y, stage1, threshold=0.99)
        _, _, n_low  = _apply_cascade_filter(X, y, stage1, threshold=0.50)

        assert n_low >= n_high, (
            f"Dusuk threshold ({0.50}) daha az kaldirmali degil: "
            f"n_low={n_low}, n_high={n_high}"
        )
