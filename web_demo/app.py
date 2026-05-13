import os
import sys
import re
import time
import math
import pickle

import numpy as np
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
import shap
from rapidfuzz.distance import Levenshtein

# Ensure project root is on sys.path for imports
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import CASCADE_THRESHOLD
from utils.feature_pipeline import build_pair_vector
from utils.similarity_utils import _jaccard_sim, _string_bigram_jaccard

from preprocessing.tokenizer import normalize_tokens, tokenize
from preprocessing.code_features import (
    _extract_single, cf_pattern_similarity, FEATURE_NAMES as AST_FEATURE_NAMES
)

# ================= APP =================
app = FastAPI()

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================= EXPERIMENT SELECTION =================
def get_experiment_path(base_dir="experiments"):
    """Find the specified experiment or fallback to the latest one."""
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(_PROJECT_ROOT, base_dir)

    exp_id_env = os.environ.get("EXP_ID")
    if exp_id_env:
        try:
            target_id = int(exp_id_env)
        except ValueError:
            raise RuntimeError("EXP_ID environment variable must be an integer.")

        for name in os.listdir(base_dir):
            m = re.match(r"exp_(\d+)_", name)
            if m and int(m.group(1)) == target_id:
                return os.path.join(base_dir, name)
        raise RuntimeError(f"Experiment ID {target_id} not found in {base_dir}")

    # Auto-detect latest
    exp_nums = []
    for name in os.listdir(base_dir):
        m = re.match(r"exp_(\d+)_", name)
        if m:
            exp_nums.append((int(m.group(1)), name))

    if not exp_nums:
        raise RuntimeError("No experiments found in experiments/ directory.")

    _, latest_name = max(exp_nums, key=lambda x: x[0])
    return os.path.join(base_dir, latest_name)


EXP_PATH = get_experiment_path()
print(f"📦 Loading experiment: {EXP_PATH}")

with open(f"{EXP_PATH}/model.pkl", "rb") as f:
    model = pickle.load(f)

with open(f"{EXP_PATH}/tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load char TF-IDF vectorizer if available
char_vectorizer = None
char_tfidf_path = f"{EXP_PATH}/char_tfidf.pkl"
if os.path.exists(char_tfidf_path):
    with open(char_tfidf_path, "rb") as f:
        char_vectorizer = pickle.load(f)
    print("   ✅ Char TF-IDF vectorizer loaded")
else:
    print("   ⚠️  No char TF-IDF vectorizer found (using legacy mode)")

# Load SVD model
svd_model = None
svd_path = f"{EXP_PATH}/svd.pkl"
if os.path.exists(svd_path):
    with open(svd_path, "rb") as f:
        svd_model = pickle.load(f)
    print("   ✅ SVD model loaded")
else:
    print("   ⚠️  No SVD model found")

# Load Stage1 model
stage1_model = None
stage1_path = f"{EXP_PATH}/stage1_model.pkl"
if os.path.exists(stage1_path):
    with open(stage1_path, "rb") as f:
        stage1_model = pickle.load(f)
    print("   ✅ Stage-1 Lexical model loaded")
else:
    print("   ⚠️  No Stage-1 Lexical model found")



# ── Cascade inference hazırlığı ──────────────────────────────────────────────
_IS_CASCADE       = "CASCADE" in EXP_PATH
# TF-IDF özellikleri vektörden çıkarıldığı için cos_token artık doğrudan 0. indekstedir.
_COS_TOKEN_IDX    = 0

print(f"   {'🌊 CASCADE mode aktif' if _IS_CASCADE else '📊 Standart mode aktif'} "
      f"(cascade threshold={CASCADE_THRESHOLD})")


# ================= FEATURE NAMES =================
def _build_feature_names():
    """
    SHAP açıklamaları için okunabilir feature isimleri.
    Sıra build_pair_vector() / pair_generator.py ile birebir eşleştirilmiştir.
    """
    names = []

    # TF-IDF (Token/Char) sütunları sistemden çıkarıldığı için özellik isimleri
    # artık doğrudan extra_features ile başlıyor.

    # Extra features (sıra pair_generator.py ile aynı olmalı)
    names.append("cosine_similarity_token")
    names.append("length_ratio")
    names.append("manhattan_token")    # Bug #11 düzeltildi: önceden eksikti
    names.append("euclidean_token")    # Bug #11 düzeltildi: önceden eksikti

    if char_vectorizer is not None:
        names.append("cosine_similarity_char")

    # AST feature ratios (14) + diffs (14) = 28 — Bug #11 düzeltildi
    for feat_name in AST_FEATURE_NAMES:
        names.append(f"{feat_name}_ratio")
        names.append(f"{feat_name}_diff")

    # CF pattern similarity
    names.append("cf_pattern_similarity")

    # Semantic similarity features (7)
    names.append("semantic_library_call_jaccard")
    names.append("semantic_library_categories_jaccard")
    names.append("semantic_data_struct_jaccard")
    names.append("semantic_io_pattern_jaccard")
    names.append("semantic_math_op_jaccard")
    names.append("semantic_skeleton_jaccard")
    names.append("semantic_abstract_cf_similarity")
    names.append("semantic_type_profile_cosine")

    # SVD farkları
    if svd_model is not None:
        for i in range(svd_model.n_components):
            names.append(f"svd_diff_{i}")

    return names


FEATURE_NAMES = _build_feature_names()
print(f"   📊 Total feature names: {len(FEATURE_NAMES)}")


# ================= SHAP EXPLAINER =================
print("   🔬 Initializing SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)
print("   ✅ SHAP explainer ready")


# ================= SCHEMA =================
class CodePair(BaseModel):
    code1: str
    code2: str


# ================= PREPROCESS =================
def preprocess(code: str) -> str:
    tokens = tokenize(code)
    norm_tokens = normalize_tokens(tokens)
    return " ".join(norm_tokens)


# ================= ROUTES =================
_HTML_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/", response_class=HTMLResponse)
def home():
    html_path = os.path.join(_HTML_DIR, "index.html")
    with open(html_path, "r") as f:
        return f.read()


@app.post("/predict")
def predict(pair: CodePair):
    try:
        raw1 = pair.code1.strip()
        raw2 = pair.code2.strip()

        if not raw1 or not raw2:
            raise HTTPException(status_code=400, detail="Both code snippets are required.")

        X_pair = build_pair_vector(raw1, raw2, vectorizer, char_vectorizer, svd_model)

        # ---- CASCADE modunda cos_token ön-filtresi ----
        if stage1_model is not None:
            X_lexical = X_pair[:, :4]
            y_prob_stage1 = float(stage1_model.predict_proba(X_lexical)[0][1])
            if y_prob_stage1 >= 0.95:
                return {
                    "probability": 1.0,
                    "prediction": "Duplicated",
                    "cascade_filtered": True,
                    "cos_token": None,
                    "shap": None
                }
        elif _IS_CASCADE:
            _cell = X_pair[0, _COS_TOKEN_IDX]
            cos_token = float(_cell.toarray().ravel()[0]) if hasattr(_cell, 'toarray') else float(_cell)
            if cos_token > CASCADE_THRESHOLD:
                prob = 1.0
                return {
                    "probability": round(prob, 4),
                    "prediction": "Duplicated",
                    "cascade_filtered": True,
                    "cos_token": round(cos_token, 4),
                    "shap": None
                }

        # ---- Predict ----
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_pair)[0][1]
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(X_pair)[0]
            prob = 1 / (1 + math.exp(-decision))
        else:
            pred = model.predict(X_pair)[0]
            prob = float(pred)

        # ---- SHAP Explanation ----
        shap_data = None
        try:
            X_dense = X_pair.toarray() if hasattr(X_pair, 'toarray') else np.array(X_pair)
            shap_values = explainer.shap_values(X_dense)

            # Binary classification: shap_values is list [class0, class1]
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0]

            base_value = float(explainer.expected_value[1]) if isinstance(
                explainer.expected_value, (list, np.ndarray)
            ) else float(explainer.expected_value)

            # Top-100 features by absolute SHAP value
            top_k = 100
            abs_sv = np.abs(sv)
            top_indices = np.argsort(abs_sv)[-top_k:][::-1]

            shap_features = []
            for idx in top_indices:
                fname = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"
                display_name = fname.replace("tfidf_", "").replace("char_tfidf_", "char:")
                shap_features.append({
                    "feature": display_name,
                    "value": round(float(X_dense[0, idx]), 4),
                    "shap_value": round(float(sv[idx]), 4)
                })

            shap_data = {
                "base_value": round(base_value, 4),
                "features": shap_features
            }
        except Exception as e:
            print(f"⚠️ SHAP explanation failed: {e}")
            shap_data = None

        return {
            "probability": round(float(prob), 4),
            "prediction": "Duplicated" if prob > 0.95 else "Not Duplicated",
            "cascade_filtered": False,
            "shap": shap_data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
