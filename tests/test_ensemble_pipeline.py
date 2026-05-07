"""
Integration Test: Ensemble Model Pipeline
=========================================
Ensemble modelinin tam bir küçük veri setiyle sorunsuz çalıştığını doğrular:
  - generate_pairs → dense array
  - StackingClassifier fit + predict_proba
  - Cascade filtresi (cos_token @ index 0)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from preprocessing.tokenizer import tokenize, normalize_tokens
from preprocessing.code_features import extract_all_features
from pairing.pair_generator import generate_pairs
from models.ensemble import build_ensemble
from utils.feature_pipeline import build_pair_vector


# ─── Minimal Code Corpus ─────────────────────────────────────────────────────

CODES = {
    "class_A": [
        "int sum(int a[], int n) { int s=0; for(int i=0;i<n;i++) s+=a[i]; return s; }",
        "int total(int arr[], int size) { int r=0; for(int i=0;i<size;i++) r+=arr[i]; return r; }",
        "int accumulate(int v[], int n) { int t=0; int i=0; while(i<n){ t+=v[i]; i++;} return t;}",
    ],
    "class_B": [
        "int factorial(int n){ if(n<=1) return 1; return n*factorial(n-1);}",
        "int fact(int n){ int r=1; for(int i=2;i<=n;i++) r*=i; return r;}",
        "long fac(long n){ long res=1; while(n>1){ res*=n; n--;} return res;}",
    ],
    "class_C": [
        "void swap(int* a, int* b){ int t=*a; *a=*b; *b=t;}",
        "void exchange(int* x, int* y){ int tmp=*x; *x=*y; *y=tmp;}",
        "void swapVals(int& p, int& q){ int hold=p; p=q; q=hold;}",
    ],
}

ALL_CODES_RAW = []
LABELS = []
for lbl, snippets in CODES.items():
    for s in snippets:
        ALL_CODES_RAW.append(s)
        LABELS.append(lbl)


@pytest.fixture(scope="module")
def pipeline():
    """Küçük corpus için tam pipeline: TF-IDF, SVD, feature extraction."""
    processed = [" ".join(normalize_tokens(tokenize(c))) for c in ALL_CODES_RAW]

    vec = TfidfVectorizer(
        min_df=1, max_df=1.0, ngram_range=(1, 2),
        token_pattern=r"[^ ]+", sublinear_tf=True, max_features=100
    )
    X_token = vec.fit_transform(processed)

    svd = TruncatedSVD(n_components=10, random_state=42)
    X_svd = svd.fit_transform(X_token)

    code_features, cf_patterns, sem_features = extract_all_features(ALL_CODES_RAW)

    return {
        "processed": processed,
        "vectorizer": vec,
        "X_token": X_token,
        "X_svd": X_svd,
        "code_features": code_features,
        "cf_patterns": cf_patterns,
        "sem_features": sem_features,
    }


class TestEnsemblePipeline:
    def test_generate_pairs_returns_dense(self, pipeline):
        """generate_pairs artık dense numpy array döndürmeli."""
        X, y = generate_pairs(
            pipeline["X_token"], LABELS, 60, pipeline["processed"],
            code_features=pipeline["code_features"],
            cf_patterns=pipeline["cf_patterns"],
            semantic_features=pipeline["sem_features"],
            X_svd=pipeline["X_svd"],
            random_state=42
        )
        assert isinstance(X, np.ndarray), f"Dense array bekleniyor, {type(X)} alındı"
        assert X.dtype == np.float32
        assert X.shape[0] == 60
        assert X.ndim == 2

    def test_cos_token_at_index_zero(self, pipeline):
        """Cascade filtresi cos_token'ı index 0'dan okuyor — aralık kontrolü."""
        X, y = generate_pairs(
            pipeline["X_token"], LABELS, 30, pipeline["processed"],
            code_features=pipeline["code_features"],
            cf_patterns=pipeline["cf_patterns"],
            semantic_features=pipeline["sem_features"],
            X_svd=pipeline["X_svd"],
            random_state=0
        )
        cos_tokens = X[:, 0]
        assert (cos_tokens >= 0.0).all() and (cos_tokens <= 1.0).all(), (
            "cos_token değerleri [0, 1] dışında!"
        )

    def test_ensemble_fit_predict(self, pipeline):
        """StackingClassifier küçük bir veri setinde fit/predict_proba çalışmalı."""
        X, y = generate_pairs(
            pipeline["X_token"], LABELS, 80, pipeline["processed"],
            code_features=pipeline["code_features"],
            cf_patterns=pipeline["cf_patterns"],
            semantic_features=pipeline["sem_features"],
            X_svd=pipeline["X_svd"],
            random_state=99
        )
        # Ensure both classes exist
        if len(np.unique(y)) < 2:
            pytest.skip("Tek sınıf var, ensemble testi atlanıyor.")

        model = build_ensemble(random_state=42, device="cpu")
        model.fit(X, y)

        proba = model.predict_proba(X)
        preds = model.predict(X)

        assert proba.shape == (len(y), 2), f"predict_proba şekli hatalı: {proba.shape}"
        assert set(preds).issubset({0, 1}), f"Geçersiz tahmin değerleri: {set(preds)}"
        assert ((proba >= 0) & (proba <= 1)).all(), "Olasılık değerleri [0,1] dışında!"

    def test_cascade_filter_logic(self, pipeline):
        """
        cos_sim > CASCADE_THRESHOLD olan çiftler klon olarak işaretlenmeli
        (XGBoost'a gitmeden).
        """
        from config import CASCADE_THRESHOLD
        from utils.feature_pipeline import build_pair_vector

        # Aynı kod → cos_sim ~1.0 → cascade tarafından yakalanmalı
        vec = pipeline["vectorizer"]
        code_a = ALL_CODES_RAW[0]
        result = build_pair_vector(code_a, code_a, vec)
        cos_token = float(result[0, 0])
        assert cos_token > CASCADE_THRESHOLD, (
            f"Aynı kod cascade'i tetiklemeli: cos={cos_token:.4f}, threshold={CASCADE_THRESHOLD}"
        )

    def test_positive_ratio_parameter(self, pipeline):
        """positive_ratio parametresi gerçek sınıf dağılımını etkilemeli."""
        X_bal, y_bal = generate_pairs(
            pipeline["X_token"], LABELS, 100, pipeline["processed"],
            X_svd=pipeline["X_svd"], random_state=1, positive_ratio=0.5
        )
        X_imb, y_imb = generate_pairs(
            pipeline["X_token"], LABELS, 100, pipeline["processed"],
            X_svd=pipeline["X_svd"], random_state=1, positive_ratio=0.1
        )
        pos_rate_bal = y_bal.sum() / len(y_bal)
        pos_rate_imb = y_imb.sum() / len(y_imb)
        # Dengeli sette pozitif oranı dengesizden daha yüksek olmalı
        assert pos_rate_bal > pos_rate_imb, (
            f"Dengeli: {pos_rate_bal:.2f}, Dengesiz: {pos_rate_imb:.2f}"
        )
