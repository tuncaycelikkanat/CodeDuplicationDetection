import os
import sys
import re
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
import shap

# Ensure project root is on sys.path for imports
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from preprocessing.tokenizer import tokenize, normalize_tokens
from preprocessing.code_features import (
    _count_loops, _count_branches, _count_func_calls,
    _compute_nesting_depth, _count_operators,
    _extract_cf_pattern, cf_pattern_similarity
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


# ================= AUTO-DETECT LATEST EXPERIMENT =================
def get_latest_experiment(base_dir="experiments"):
    """Find the experiment with the highest number."""
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(_PROJECT_ROOT, base_dir)

    exp_nums = []
    for name in os.listdir(base_dir):
        m = re.match(r"exp_(\d+)_", name)
        if m:
            exp_nums.append((int(m.group(1)), name))

    if not exp_nums:
        raise RuntimeError("No experiments found in experiments/ directory.")

    _, latest_name = max(exp_nums, key=lambda x: x[0])
    return os.path.join(base_dir, latest_name)


EXP_PATH = get_latest_experiment()
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


# ================= FEATURE NAMES =================
def _build_feature_names():
    """Build human-readable feature names for SHAP explanations."""
    names = []

    # Token TF-IDF feature names
    token_features = vectorizer.get_feature_names_out()
    for feat in token_features:
        names.append(f"tfidf_{feat}")

    # Extra features (always in this order)
    names.append("cosine_similarity_token")
    names.append("jaccard_overlap")
    names.append("length_ratio")

    if char_vectorizer is not None:
        names.append("cosine_similarity_char")

    # AST feature ratios
    names.append("loop_ratio")
    names.append("branch_ratio")
    names.append("func_call_ratio")
    names.append("nesting_depth_ratio")
    names.append("operator_ratio")

    # CF pattern similarity
    names.append("cf_pattern_similarity")

    # New features (edit distance, line/char ratios)
    names.append("edit_distance_ratio")
    names.append("line_count_ratio")
    names.append("char_length_ratio")

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


def extract_features_for_code(raw_code):
    """Extract AST + control flow features for a single code."""
    features = [
        _count_loops(raw_code),
        _count_branches(raw_code),
        _count_func_calls(raw_code),
        _compute_nesting_depth(raw_code),
        _count_operators(raw_code)
    ]
    cf_pattern = _extract_cf_pattern(raw_code)
    return features, cf_pattern


def _build_pair_vector(raw1, raw2):
    """Build the feature vector for a code pair. Returns X_pair sparse matrix."""
    code1 = preprocess(raw1)
    code2 = preprocess(raw2)

    X1 = vectorizer.transform([code1])
    X2 = vectorizer.transform([code2])

    # Token TF-IDF diff
    diff = abs(X1 - X2)
    cos_token = cosine_similarity(X1, X2)[0][0]

    # Jaccard overlap
    tokens1 = set(code1.split())
    tokens2 = set(code2.split())
    union = len(tokens1 | tokens2)
    overlap = len(tokens1 & tokens2) / union if union > 0 else 0.0

    # Length ratio
    len1 = len(code1.split())
    len2 = len(code2.split())
    length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0

    extra = [cos_token, overlap, length_ratio]

    # Char TF-IDF cosine + diff
    char_diff = None
    if char_vectorizer is not None:
        C1 = char_vectorizer.transform([code1])
        C2 = char_vectorizer.transform([code2])
        cos_char = cosine_similarity(C1, C2)[0][0]
        extra.append(cos_char)
        char_diff = abs(C1 - C2)

    # AST feature ratios
    feat1, cf1 = extract_features_for_code(raw1)
    feat2, cf2 = extract_features_for_code(raw2)

    for v1, v2 in zip(feat1, feat2):
        max_val = max(v1, v2)
        ratio = min(v1, v2) / max_val if max_val > 0 else 1.0
        extra.append(ratio)

    # CF pattern similarity
    cf_sim = cf_pattern_similarity(cf1, cf2)
    extra.append(cf_sim)

    # Edit distance ratio (truncated to 500 chars for speed)
    from pairing.pair_generator import _edit_distance_ratio
    edit_dist = _edit_distance_ratio(raw1[:500], raw2[:500])
    extra.append(edit_dist)

    # Line count ratio
    lc1 = raw1.count('\n') + 1
    lc2 = raw2.count('\n') + 1
    max_lc = max(lc1, lc2)
    extra.append(min(lc1, lc2) / max_lc if max_lc > 0 else 1.0)

    # Char length ratio
    cl1 = len(raw1)
    cl2 = len(raw2)
    max_cl = max(cl1, cl2)
    extra.append(min(cl1, cl2) / max_cl if max_cl > 0 else 1.0)

    # Combine
    extra_features = csr_matrix([extra])
    parts = [diff]
    if char_diff is not None:
        parts.append(char_diff)
    parts.append(extra_features)
    X_pair = hstack(parts)

    return X_pair


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

        X_pair = _build_pair_vector(raw1, raw2)

        # ---- Predict ----
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_pair)[0][1]
        elif hasattr(model, "decision_function"):
            import math
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

            # For binary classification, shap_values can be a list [class0, class1]
            if isinstance(shap_values, list):
                sv = shap_values[1][0]  # class 1 (duplicated)
            else:
                sv = shap_values[0]

            base_value = float(explainer.expected_value[1]) if isinstance(
                explainer.expected_value, (list, np.ndarray)
            ) else float(explainer.expected_value)

            # Top-15 features by absolute SHAP value
            top_k = 15
            abs_sv = np.abs(sv)
            top_indices = np.argsort(abs_sv)[-top_k:][::-1]

            shap_features = []
            for idx in top_indices:
                fname = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"
                # Clean up tfidf_ prefix for display
                display_name = fname.replace("tfidf_", "")
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
            "prediction": "Duplicated" if prob > 0.98 else "Not Duplicated",
            "shap": shap_data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
