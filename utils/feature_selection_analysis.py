"""
Feature Selection Analysis — CodeDuplicationDetection
=====================================================
Gelişmiş Feature Importance + Feature Selection analizi:
  - Feature Group bazlı importance (Lexical, AST, CF, Semantic, SVD, SSL_diff, SSL_product)
  - XGBoost Feature Importance + RandomForest OOB Importance
  - Korelasyon temelli gereksiz feature eliminasyonu
  - Recursive Feature Elimination (RFE) 
  - Type-1/2/3/4 bazlı detaylı analiz

Kullanım:
  python utils/feature_selection_analysis.py --exp-id 78
  python utils/feature_selection_analysis.py --exp-id 78 --test-dir evaluation/test_clones_balanced --top-k 30
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.logger import Log, Colors
from utils.test_automation import get_experiment_path, load_pairs
from utils.feature_pipeline import build_pair_vector
from config import SSL_PCA_COMPONENTS, SVD_N_COMPONENTS


# ================= FEATURE NAMING =================

def build_feature_name_list(has_svd=False, has_ssl=False, n_svd=SVD_N_COMPONENTS, n_ssl=SSL_PCA_COMPONENTS):
    """
    Feature isimlerini pair_generator.py / feature_pipeline.py ile tutarlı şekilde oluşturur.
    """
    from preprocessing.code_features import FEATURE_NAMES as AST_FEATURE_NAMES

    names = []

    # [0..3] Lexical
    names.extend(["cosine_similarity_token", "length_ratio", "manhattan_token", "euclidean_token"])

    # [4..43] AST ratios (20) + diffs (20)
    for feat_name in AST_FEATURE_NAMES:
        names.append(f"{feat_name}_ratio")
        names.append(f"{feat_name}_diff")

    # [44] CF pattern similarity
    names.append("cf_pattern_similarity")

    # [45..51] Semantic features (7)
    names.extend([
        "semantic_library_call_jaccard",
        "semantic_library_categories_jaccard",
        "semantic_data_struct_jaccard",
        "semantic_io_pattern_jaccard",
        "semantic_math_op_jaccard",
        "semantic_skeleton_jaccard",
        "semantic_abstract_cf_similarity",
    ])

    # [52] Type profile cosine
    names.append("semantic_type_profile_cosine")

    # SVD
    if has_svd:
        for i in range(n_svd):
            names.append(f"svd_diff_{i}")

    # SSL abs diff + element-wise product
    if has_ssl:
        for i in range(n_ssl):
            names.append(f"ssl_pca_diff_{i}")
        for i in range(n_ssl):
            names.append(f"ssl_pca_prod_{i}")

    return names


def assign_feature_groups(names):
    """Her feature'ı bir gruba atar."""
    groups = {}
    for i, name in enumerate(names):
        if name.startswith("cosine_") or name.startswith("length_") or name.startswith("manhattan") or name.startswith("euclidean"):
            groups[i] = "Lexical"
        elif "_ratio" in name or "_diff" in name and name.startswith(("branch_", "loop_", "nesting_", "operator_", "return_", "accumulator_", "param_", "math_op_", "library_call_", "data_struct_", "io_pattern_", "halstead_", "mccabe_", "array_access_", "ptr_deref_")):
            groups[i] = "AST"
        elif name.startswith("cf_"):
            groups[i] = "CF"
        elif name.startswith("semantic_") or name.startswith("type_profile"):
            groups[i] = "Semantic"
        elif name.startswith("svd_"):
            groups[i] = "SVD"
        elif name.startswith("ssl_pca_diff_"):
            groups[i] = "SSL_diff"
        elif name.startswith("ssl_pca_prod_"):
            groups[i] = "SSL_product"
        else:
            groups[i] = "AST"  # fallback for AST features
    return groups


# ================= HELPERS =================

def load_pairs_for_type(test_dir, t_name):
    """Loads positives and negatives for a specific type from test_dir."""
    if not os.path.isabs(test_dir):
        test_dir = os.path.join(_PROJECT_ROOT, test_dir)

    positives = load_pairs(os.path.join(test_dir, t_name), label=1)
    negatives = load_pairs(os.path.join(test_dir, "negatives"), label=0)

    if len(positives) > 0 and len(negatives) >= len(positives):
        negatives = sorted(negatives, key=lambda x: x.get('p_name', ''))[:len(positives)]

    positives.extend(negatives)
    return positives


def extract_features(pairs, vectorizer, svd_model=None, ssl_pipeline=None, ssl_pca=None):
    """Tüm çiftlerden feature vektörlerini çıkarır."""
    X_list, y_list = [], []
    for p in tqdm(pairs, desc="Extracting features"):
        vec = build_pair_vector(
            p['c1'], p['c2'], vectorizer,
            svd_model=svd_model,
            ssl_pipeline=ssl_pipeline, ssl_pca=ssl_pca
        )
        X_list.append(vec[0])
        y_list.append(p['label'])
    return np.array(X_list), np.array(y_list)


# ================= ANALYSIS =================

def analyze_feature_importance(X, y, feature_names, feature_groups, top_k=30):
    """
    XGBoost ve RandomForest ile feature importance analizi yapar.
    Hem tekil hem de grup bazlı sonuçları döndürür.
    """
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric='logloss', verbosity=0
    )
    xgb.fit(X, y)
    xgb_importance = xgb.feature_importances_
    xgb_f1 = np.mean(cross_val_score(xgb, X, y, cv=cv, scoring='f1'))

    # RandomForest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    rf_f1 = np.mean(cross_val_score(rf, X, y, cv=cv, scoring='f1'))

    # Combined importance (normalized average)
    xgb_norm = xgb_importance / (xgb_importance.sum() + 1e-9)
    rf_norm = rf_importance / (rf_importance.sum() + 1e-9)
    combined = (xgb_norm + rf_norm) / 2

    # Top-K features
    top_indices = np.argsort(combined)[-top_k:][::-1]
    top_features = []
    for idx in top_indices:
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        group = feature_groups.get(idx, "Unknown")
        top_features.append({
            "index": idx,
            "name": name,
            "group": group,
            "xgb": float(xgb_importance[idx]),
            "rf": float(rf_importance[idx]),
            "combined": float(combined[idx]),
        })

    # Group importance
    group_importance = defaultdict(float)
    group_count = defaultdict(int)
    for idx, imp in enumerate(combined):
        group = feature_groups.get(idx, "Unknown")
        group_importance[group] += imp
        group_count[group] += 1

    group_results = {}
    for group in sorted(group_importance.keys()):
        group_results[group] = {
            "total_importance": float(group_importance[group]),
            "count": group_count[group],
            "avg_importance": float(group_importance[group] / group_count[group]),
        }

    return {
        "xgb_f1": xgb_f1,
        "rf_f1": rf_f1,
        "top_features": top_features,
        "group_results": group_results,
        "combined_importance": combined,
    }


def run_feature_selection(X, y, combined_importance, feature_names, thresholds=[0.3, 0.5, 0.7]):
    """
    Farklı eşiklerle feature selection yapıp sonuçları karşılaştırır.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Baseline (all features)
    rf_all = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    f1_all = np.mean(cross_val_score(rf_all, X, y, cv=cv, scoring='f1'))
    acc_all = np.mean(cross_val_score(rf_all, X, y, cv=cv, scoring='accuracy'))

    results = [{
        "threshold": "ALL",
        "n_features": X.shape[1],
        "f1": f1_all,
        "accuracy": acc_all,
    }]

    sorted_indices = np.argsort(combined_importance)[::-1]

    for threshold in thresholds:
        n_keep = max(1, int(len(sorted_indices) * threshold))
        selected_indices = sorted_indices[:n_keep]
        X_sel = X[:, selected_indices]

        rf_sel = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        f1_sel = np.mean(cross_val_score(rf_sel, X_sel, y, cv=cv, scoring='f1'))
        acc_sel = np.mean(cross_val_score(rf_sel, X_sel, y, cv=cv, scoring='accuracy'))

        results.append({
            "threshold": f"Top {int(threshold*100)}%",
            "n_features": n_keep,
            "f1": f1_sel,
            "accuracy": acc_sel,
        })

    return results


def analyze_correlation(X, feature_names, threshold=0.95):
    """Yüksek korelasyonlu feature çiftlerini tespit eder."""
    corr = np.corrcoef(X.T)
    redundant_pairs = []
    n = corr.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) > threshold:
                name_i = feature_names[i] if i < len(feature_names) else f"f_{i}"
                name_j = feature_names[j] if j < len(feature_names) else f"f_{j}"
                redundant_pairs.append((name_i, name_j, float(corr[i, j])))
    return sorted(redundant_pairs, key=lambda x: abs(x[2]), reverse=True)


# ================= MAIN =================

def main():
    parser = argparse.ArgumentParser(description="Advanced Feature Selection Analysis per Clone Type.")
    parser.add_argument("--exp-id", type=int, required=True, help="Experiment ID to load models from")
    parser.add_argument("--test-dir", type=str, default="evaluation/test_clones_balanced", help="Test clones directory")
    parser.add_argument("--top-k", type=int, default=30, help="Number of top features to display")
    args = parser.parse_args()

    exp_path = get_experiment_path(args.exp_id)
    if not exp_path:
        Log.error(f"Experiment ID {args.exp_id} not found.")
        return

    Log.step(f"Loading from: {os.path.basename(exp_path)}")

    with open(os.path.join(exp_path, "tfidf.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    svd_model = None
    svd_path = os.path.join(exp_path, "svd.pkl")
    if os.path.exists(svd_path):
        with open(svd_path, "rb") as f:
            svd_model = pickle.load(f)
        Log.substep(f"SVD model loaded ({svd_model.n_components} components)")

    config_path = os.path.join(exp_path, "config.json")
    use_ssl = False
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            use_ssl = config.get("use_ssl", False)

    ssl_pipeline, ssl_pca = None, None
    if use_ssl:
        from vectorization.ssl_encoder import build_ssl_pipeline
        ssl_pipeline = build_ssl_pipeline(device="cpu")
        pca_path = os.path.join(exp_path, "ssl_pca.pkl")
        if os.path.exists(pca_path):
            with open(pca_path, "rb") as f:
                ssl_pca = pickle.load(f)
            Log.substep(f"SSL PCA loaded ({ssl_pca.n_components} components)")

    # Feature names
    has_svd = svd_model is not None
    has_ssl = ssl_pipeline is not None and ssl_pca is not None
    n_svd = svd_model.n_components if has_svd else 0
    n_ssl = ssl_pca.n_components if has_ssl else 0
    feature_names = build_feature_name_list(has_svd, has_ssl, n_svd, n_ssl)
    feature_groups = assign_feature_groups(feature_names)

    types = ["type1", "type2", "type3", "type4"]

    print(f"\n{Colors.MAGENTA}{Colors.BOLD}{'='*90}")
    print(f"   ADVANCED FEATURE SELECTION ANALYSIS")
    print(f"{'='*90}{Colors.RESET}\n")

    all_results = {}

    for t in types:
        print(f"\n{Colors.CYAN}{Colors.BOLD}── {t.upper()} ──{Colors.RESET}")
        pairs = load_pairs_for_type(args.test_dir, t)
        if not pairs:
            Log.warning(f"No pairs found for {t}")
            continue

        X, y = extract_features(pairs, vectorizer, svd_model, ssl_pipeline, ssl_pca)
        actual_n_features = X.shape[1]

        # Adjust feature names if mismatch
        while len(feature_names) < actual_n_features:
            feature_names.append(f"feature_{len(feature_names)}")
        while len(feature_names) > actual_n_features:
            feature_names = feature_names[:actual_n_features]

        Log.substep(f"Dataset: {X.shape[0]} pairs, {actual_n_features} features")

        # 1. Feature importance
        Log.substep("Computing feature importance (XGBoost + RF)...")
        importance_result = analyze_feature_importance(X, y, feature_names, feature_groups, top_k=args.top_k)

        # 2. Feature selection
        Log.substep("Running feature selection experiments...")
        selection_results = run_feature_selection(X, y, importance_result["combined_importance"], feature_names)

        # 3. Correlation analysis
        Log.substep("Analyzing feature correlations...")
        correlations = analyze_correlation(X, feature_names, threshold=0.95)

        all_results[t] = {
            "importance": importance_result,
            "selection": selection_results,
            "correlations": correlations,
        }

        # Print results
        print(f"\n  {Colors.WHITE}{Colors.BOLD}Feature Group Importance:{Colors.RESET}")
        print(f"  {'Group':<16} {'Total %':<10} {'Count':<8} {'Avg %':<10}")
        print(f"  {'-'*44}")
        for group, data in sorted(
            importance_result["group_results"].items(),
            key=lambda x: x[1]["total_importance"], reverse=True
        ):
            total_pct = data["total_importance"] * 100
            avg_pct = data["avg_importance"] * 100
            print(f"  {group:<16} {total_pct:>6.2f}%   {data['count']:<8} {avg_pct:>6.3f}%")

        print(f"\n  {Colors.WHITE}{Colors.BOLD}Top-{min(15, args.top_k)} Features:{Colors.RESET}")
        print(f"  {'Rank':<5} {'Feature':<40} {'Group':<14} {'XGB':<8} {'RF':<8} {'Avg':<8}")
        print(f"  {'-'*83}")
        for rank, feat in enumerate(importance_result["top_features"][:15], 1):
            print(f"  {rank:<5} {feat['name']:<40} {feat['group']:<14} "
                  f"{feat['xgb']:.4f}  {feat['rf']:.4f}  {feat['combined']:.4f}")

        print(f"\n  {Colors.WHITE}{Colors.BOLD}Feature Selection Results:{Colors.RESET}")
        print(f"  {'Config':<12} {'Features':<12} {'F1':<10} {'Accuracy':<10} {'Delta F1':<10}")
        print(f"  {'-'*54}")
        baseline_f1 = selection_results[0]["f1"]
        for res in selection_results:
            delta = res["f1"] - baseline_f1
            delta_str = f"{delta:+.4f}" if res["threshold"] != "ALL" else "—"
            color = Colors.GREEN if delta >= 0 else Colors.RED
            if res["threshold"] == "ALL":
                color = Colors.WHITE
            print(f"  {res['threshold']:<12} {res['n_features']:<12} "
                  f"{res['f1']:.4f}    {res['accuracy']:.4f}    "
                  f"{color}{delta_str}{Colors.RESET}")

        if correlations:
            print(f"\n  {Colors.YELLOW}High Correlation Pairs (>{0.95}):{Colors.RESET} {len(correlations)} found")
            for f1, f2, corr in correlations[:5]:
                print(f"    {f1} ↔ {f2}: {corr:.4f}")

    # Summary table
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*90}")
    print(f"   SUMMARY TABLE")
    print(f"{'='*90}{Colors.RESET}\n")
    print(f"  {'Type':<8} {'XGB F1':<10} {'RF F1':<10} {'All Feat':<10} {'Top50% F1':<10} {'Top30% F1':<10}")
    print(f"  {'-'*58}")
    for t in types:
        if t not in all_results:
            continue
        res = all_results[t]
        imp = res["importance"]
        sel = res["selection"]
        sel_50 = next((s for s in sel if "50" in str(s["threshold"])), None)
        sel_30 = next((s for s in sel if "30" in str(s["threshold"])), None)
        print(f"  {t:<8} {imp['xgb_f1']:.4f}    {imp['rf_f1']:.4f}    "
              f"{sel[0]['f1']:.4f}    "
              f"{sel_50['f1']:.4f if sel_50 else 'N/A':<10} "
              f"{sel_30['f1']:.4f if sel_30 else 'N/A':<10}")

    print(f"\n{Colors.GREEN}{Colors.BOLD}Analysis complete!{Colors.RESET}\n")


if __name__ == "__main__":
    main()
