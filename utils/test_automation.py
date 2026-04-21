import os
import sys
import time
import json
import re
import pickle
import numpy as np
import math
import argparse
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.feature_pipeline import build_pair_vector

def get_experiment_path(exp_id=None, base_dir="experiments"):
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(_PROJECT_ROOT, base_dir)
    
    if not os.path.exists(base_dir):
        return None

    if exp_id is not None:
        for name in os.listdir(base_dir):
            m = re.match(r"exp_(\d+)_", name)
            if m and int(m.group(1)) == exp_id:
                return os.path.join(base_dir, name)
        print(f"❌ Experiment ID {exp_id} not found.")
        return None

    # Fallback to latest
    exp_nums = []
    for name in os.listdir(base_dir):
        m = re.match(r"exp_(\d+)_", name)
        if m:
            exp_nums.append((int(m.group(1)), name))
    
    if not exp_nums: return None
    _, latest_name = max(exp_nums, key=lambda x: x[0])
    return os.path.join(base_dir, latest_name)

def calculate_metrics(tp, fp, tn, fn):
    # Safe division helpers
    def safe_div(n, d): return n / d if d > 0 else 0.0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, tp + fp + tn + fn)
    
    # Matthews Correlation Coefficient (MCC)
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_div(mcc_num, mcc_den)

    # Balanced Accuracy
    tpr = safe_div(tp, tp + fn)
    tnr = safe_div(tn, tn + fp)
    balanced_acc = (tpr + tnr) / 2

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "mcc": round(mcc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }

def calculate_auc_roc(y_true, y_prob):
    # Simple AUC calculation
    if not y_true or not y_prob: return 0.0
    from sklearn.metrics import roc_auc_score
    try:
        return round(roc_auc_score(y_true, y_prob), 4)
    except:
        return 0.0

def load_pairs(dir_path, label):
    pairs = []
    if not os.path.exists(dir_path):
        return pairs
    pair_folders = sorted([d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))])
    for p_folder in pair_folders:
        p_path = os.path.join(dir_path, p_folder)
        try:
            with open(os.path.join(p_path, "original.txt"), "r") as f:
                c1 = f.read()
            with open(os.path.join(p_path, "clone.txt"), "r") as f:
                c2 = f.read()
            pairs.append({
                "c1": c1, "c2": c2, "label": label, 
                "p_name": f"{os.path.basename(dir_path)}/{p_folder}"
            })
        except FileNotFoundError:
            continue
    return pairs

def run_automation(test_dir="test_clones", threshold=0.95, exp_id=None):
    if not os.path.isabs(test_dir):
        test_dir = os.path.join(_PROJECT_ROOT, test_dir)

    exp_path = get_experiment_path(exp_id=exp_id)
    if not exp_path:
        print("❌ No experiments found or specified experiment missing.")
        return

    print(f"📦 Loading experiment: {exp_path}")
    with open(os.path.join(exp_path, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(exp_path, "tfidf.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    char_vectorizer = None
    char_tfidf_path = os.path.join(exp_path, "char_tfidf.pkl")
    if os.path.exists(char_tfidf_path):
        with open(char_tfidf_path, "rb") as f:
            char_vectorizer = pickle.load(f)

    # Bug #4 düzeltildi: cos_token feature'sının indeksi char_vectorizer'a göre hesapla.
    # build_pair_vector() çıktısı yapısı: [token_diff | char_diff? | extra...]
    # extra'nın ilk feature'sı her zaman cos_token'dur.
    _token_feat_count = len(vectorizer.vocabulary_)
    _char_feat_count  = len(char_vectorizer.vocabulary_) if char_vectorizer is not None else 0
    COS_TOKEN_IDX     = _token_feat_count + _char_feat_count  # cos_token'un gerçek sütun indeksi

    def _get_scalar(mat, col_idx):
        """csr_matrix[0, col] sparse veya scalar döndürebilir — ikisini de float'a çevirir."""
        val = mat[0, col_idx]
        if hasattr(val, 'toarray'):
            return float(val.toarray().ravel()[0])
        return float(val)

    # Load negative pairs once
    negatives = load_pairs(os.path.join(test_dir, "negatives"), label=0)
    print(f"🔍 Found {len(negatives)} negative pairs.")

    types = ["type1", "type2", "type3", "type4"]
    report = {
        "run_info": {
            "experiment": exp_path,
            "threshold": threshold,
            "timestamp": int(time.time())
        },
        "per_type": {},
        "global": {}
    }

    global_tp, global_fp, global_tn, global_fn = 0, 0, 0, 0
    global_y_true, global_y_prob = [], []

    for t in types:
        positives = load_pairs(os.path.join(test_dir, t), label=1)
        if not positives:
            print(f"⚠️ No positive pairs found for {t}. Skipping.")
            continue
        
        print(f"\n🚀 Evaluating {t} ({len(positives)} positives, {len(negatives)} negatives)...")
        test_pairs = positives + negatives
        
        tp, fp, tn, fn = 0, 0, 0, 0
        y_true, y_prob = [], []
        details = []

        for p in tqdm(test_pairs, desc=t):
            X_pair = build_pair_vector(p['c1'], p['c2'], vectorizer, char_vectorizer)

            # cos_token: build_pair_vector'da extra'nın ilk elemanı
            cos_token = _get_scalar(X_pair, COS_TOKEN_IDX)

            # Cascade Logic
            if "CASCADE" in exp_path and cos_token > 0.85:
                prob = 1.0
            else:
                if hasattr(model, "predict_proba"):
                    try:
                        prob = float(model.predict_proba(X_pair)[0][1])
                    except:
                        prob = float(model.predict(X_pair)[0])
                else:
                    prob = float(model.predict(X_pair)[0])
                
            pred = 1 if prob >= threshold else 0
            
            details.append({
                "pair": p['p_name'],
                "label": p['label'],
                "probability": round(prob, 4),
                "prediction": pred
            })
            
            y_true.append(p['label'])
            y_prob.append(prob)

            if p['label'] == 1 and pred == 1: tp += 1
            elif p['label'] == 1 and pred == 0: fn += 1
            elif p['label'] == 0 and pred == 1: fp += 1
            elif p['label'] == 0 and pred == 0: tn += 1
            
        metrics = calculate_metrics(tp, fp, tn, fn)
        metrics["auc_roc"] = calculate_auc_roc(y_true, y_prob)
        
        report["per_type"][t] = {
            "total_pairs": len(test_pairs),
            "positive_pairs": len(positives),
            "negative_pairs": len(negatives),
            **metrics,
            "details": details
        }
        
        # Accumulate global metrics (note: negatives will be counted multiple times if we just sum them,
        # but standard global usually means across ALL pairs tested. 
        # Alternatively, we could test negatives once for the global.
        # Let's count them once for the global scope to be accurate on negatives.
        global_tp += tp
        global_fn += fn

    # Test negatives once for global scope to avoid counting them N times
    print("\n🚀 Evaluating Negatives for Global Scope...")
    tn, fp = 0, 0
    neg_y_true, neg_y_prob = [], []
    for p in tqdm(negatives, desc="Global Negatives"):
        X_pair = build_pair_vector(p['c1'], p['c2'], vectorizer, char_vectorizer)
        cos_token = _get_scalar(X_pair, COS_TOKEN_IDX)

        if "CASCADE" in exp_path and cos_token > 0.85:
            prob = 1.0
        else:
            prob = float(model.predict_proba(X_pair)[0][1]) if hasattr(model, "predict_proba") else float(model.predict(X_pair)[0])
            
        pred = 1 if prob >= threshold else 0
        neg_y_true.append(0)
        neg_y_prob.append(prob)
        if pred == 1: fp += 1
        else: tn += 1
        
    global_fp, global_tn = fp, tn
    
    # Collect all global positive y_true/y_prob
    all_pos_true, all_pos_prob = [], []
    for t in types:
        if t in report["per_type"]:
            # Positives are the first N elements
            n_pos = report["per_type"][t]["positive_pairs"]
            all_pos_true.extend([1] * n_pos)
            all_pos_prob.extend([d["probability"] for d in report["per_type"][t]["details"][:n_pos]])

    global_y_true = all_pos_true + neg_y_true
    global_y_prob = all_pos_prob + neg_y_prob

    global_metrics = calculate_metrics(global_tp, global_fp, global_tn, global_fn)
    global_metrics["auc_roc"] = calculate_auc_roc(global_y_true, global_y_prob)
    global_metrics["total_pairs"] = len(global_y_true)
    
    report["global"] = global_metrics

    # Save Results
    out_dir = os.path.join(_PROJECT_ROOT, "test_results", f"run_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    with open(os.path.join(out_dir, "confusion_matrix.json"), "w") as f:
        cm_data = { "global": {"tp": global_tp, "fp": global_fp, "tn": global_tn, "fn": global_fn} }
        for t in types:
            if t in report["per_type"]:
                rm = report["per_type"][t]
                cm_data[t] = {"tp": rm["tp"], "fp": rm["fp"], "tn": rm["tn"], "fn": rm["fn"]}
        json.dump(cm_data, f, indent=4)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"Experiment: {exp_path}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write("="*40 + "\nGLOBAL METRICS\n" + "="*40 + "\n")
        for k, v in report["global"].items():
            f.write(f"{k.ljust(20)}: {v}\n")
        
        for t in types:
            if t in report["per_type"]:
                f.write("\n" + "="*40 + f"\n{t.upper()} METRICS\n" + "="*40 + "\n")
                rm = report["per_type"][t]
                for k in ["total_pairs", "precision", "recall", "f1_score", "accuracy", "mcc", "auc_roc", "tp", "fp", "tn", "fn"]:
                    f.write(f"{k.ljust(20)}: {rm.get(k)}\n")

    print(f"\n✅ Results saved to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run automation tests on code clone models.")
    parser.add_argument("--exp-id", type=int, default=None, help="Specific experiment ID to test (e.g. 54). If not provided, uses latest.")
    parser.add_argument("--threshold", type=float, default=0.95, help="Classification threshold (default 0.95)")
    args = parser.parse_args()
    
    run_automation(threshold=args.threshold, exp_id=args.exp_id)
