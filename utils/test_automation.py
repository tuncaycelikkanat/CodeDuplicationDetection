import os
import sys
import time
import json
import re
import pickle
import numpy as np
import math
import argparse
from utils.logger import Log, Colors

from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import CASCADE_STAGE1_THRESHOLD, STAGE1_FEATURE_COUNT



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

def calculate_auc(y_true, y_prob):
    if not y_true or not y_prob: return 0.0, 0.0
    from sklearn.metrics import roc_auc_score, average_precision_score
    try:
        return round(roc_auc_score(y_true, y_prob), 4), round(average_precision_score(y_true, y_prob), 4)
    except Exception as e:
        Log.warning(f"AUC hesabı başarısız: {e}")
        return 0.0, 0.0

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

def run_automation(test_dir="evaluation/test_clones", threshold=0.95, exp_id=None, auto_thresh=False):
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
        
    config_path = os.path.join(exp_path, "config.json")
    use_ssl = False
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            use_ssl = config.get("use_ssl", False)
            
    ssl_pipeline = None
    ssl_pca = None
    if use_ssl:
        Log.substep("Loading SSL pipeline for inference...")
        from vectorization.ssl_encoder import build_ssl_pipeline
        ssl_pipeline = build_ssl_pipeline(device="cpu") # Inference on CPU for testing by default
        _ssl_pca_path = os.path.join(exp_path, "ssl_pca.pkl")
        if os.path.exists(_ssl_pca_path):
            with open(_ssl_pca_path, "rb") as f:
                ssl_pca = pickle.load(f)
            Log.substep(f"ssl_pca loaded ({ssl_pca.n_components} components)")
        else:
            Log.warning("ssl_pca.pkl bulunamadi — eski 2-skaler mod")

    char_vectorizer = None  # Deprecated

    svd_model = None
    svd_path = os.path.join(exp_path, "svd.pkl")
    if os.path.exists(svd_path):
        with open(svd_path, "rb") as f:
            svd_model = pickle.load(f)

    stage1_model = None
    stage1_path = os.path.join(exp_path, "stage1_model.pkl")
    if os.path.exists(stage1_path):
        with open(stage1_path, "rb") as f:
            stage1_model = pickle.load(f)

    # TF-IDF özelliklerini vektörden çıkardığımız için cos_token artık doğrudan 0. indekstedir.
    COS_TOKEN_IDX = 0

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
    # Testte Negatif çiftlerin tüm tiplerde aynı sayıları (FP/TN) üretmesini engellemek için,
    # negatifleri tiplere bölüyoruz (Örn: 950 negatif varsa, 4 tipe yaklaşık 237'şer tane dağıtılır)
    np.random.seed(42) # Her testte aynı dağılım olsun
    negatives_shuffled = negatives.copy()
    np.random.shuffle(negatives_shuffled)
    
    # Calculate chunk size
    chunk_size = len(negatives_shuffled) // len(types)
    type_negatives_map = {}
    
    for i, t in enumerate(types):
        start_idx = i * chunk_size
        # Give remaining pairs to the last type
        end_idx = (i + 1) * chunk_size if i < len(types) - 1 else len(negatives_shuffled)
        type_negatives_map[t] = negatives_shuffled[start_idx:end_idx]

    global_tp, global_fp, global_tn, global_fn = 0, 0, 0, 0
    global_y_true, global_y_prob = [], []
    for t in types:
        positives = load_pairs(os.path.join(test_dir, t), label=1)
        if not positives:
            Log.warning(f"No positive pairs found for {t}. Skipping.")
            continue
            
        # Generate Negatives for this specific Type
        t_negatives = type_negatives_map.get(t, [])
        neg_tp, neg_fp, neg_tn, neg_fn = 0, 0, 0, 0
        neg_y_true, neg_y_prob = [], []
        neg_details = []
        
        if t_negatives:
            Log.step(f"Evaluating {len(t_negatives)} Negatives specifically for {t}...")
            for p in tqdm(t_negatives, desc=f"{t} (Neg)"):
                X_pair = build_pair_vector(
                    p['c1'], p['c2'],
                    vectorizer,
                    svd_model=svd_model,
                    ssl_pipeline=ssl_pipeline,
                    ssl_pca=ssl_pca
                )

                cos_token = _get_scalar(X_pair, COS_TOKEN_IDX)

                if stage1_model is not None:
                    X_stage1 = X_pair[:, :STAGE1_FEATURE_COUNT]
                    y_prob_stage1 = float(stage1_model.predict_proba(X_stage1)[0][1])
                    if y_prob_stage1 >= CASCADE_STAGE1_THRESHOLD:
                        prob = 1.0
                    else:
                        prob = float(model.predict_proba(X_pair)[0][1]) if hasattr(model, "predict_proba") else float(model.predict(X_pair)[0])
                elif "CASCADE" in exp_path and cos_token > CASCADE_STAGE1_THRESHOLD:
                    prob = 1.0
                else:
                    prob = float(model.predict_proba(X_pair)[0][1]) if hasattr(model, "predict_proba") else float(model.predict(X_pair)[0])
                    
                pred = 1 if prob >= threshold else 0
                
                neg_details.append({
                    "pair": p['p_name'],
                    "label": p['label'],
                    "probability": round(prob, 4),
                    "prediction": pred
                })
                
                neg_y_true.append(0)
                neg_y_prob.append(prob)

                if pred == 1: neg_fp += 1
                else: neg_tn += 1

        print(f"\n🚀 Evaluating {t} ({len(positives)} positives)...")
        
        tp, fp, tn, fn = 0, 0, 0, 0
        y_true, y_prob = [], []
        details = []

        for p in tqdm(positives, desc=t):
            X_pair = build_pair_vector(
                p['c1'], p['c2'],
                vectorizer,
                svd_model=svd_model,
                ssl_pipeline=ssl_pipeline,
                ssl_pca=ssl_pca
            )

            cos_token = _get_scalar(X_pair, COS_TOKEN_IDX)

            if stage1_model is not None:
                X_stage1 = X_pair[:, :STAGE1_FEATURE_COUNT]
                y_prob_stage1 = float(stage1_model.predict_proba(X_stage1)[0][1])
                if y_prob_stage1 >= CASCADE_STAGE1_THRESHOLD:
                    prob = 1.0
                else:
                    if hasattr(model, "predict_proba"):
                        try:
                            prob = float(model.predict_proba(X_pair)[0][1])
                        except Exception as e:
                            prob = float(model.predict(X_pair)[0])
                    else:
                        prob = float(model.predict(X_pair)[0])
            elif "CASCADE" in exp_path and cos_token > CASCADE_STAGE1_THRESHOLD:
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

            if pred == 1: tp += 1
            else: fn += 1
            
        # Combine with pre-computed negatives for this type's report
        combined_tp = tp + neg_tp
        combined_fp = fp + neg_fp
        combined_tn = tn + neg_tn
        combined_fn = fn + neg_fn
        
        combined_y_true = y_true + neg_y_true
        combined_y_prob = y_prob + neg_y_prob
        
        metrics = calculate_metrics(combined_tp, combined_fp, combined_tn, combined_fn)
        roc_auc, pr_auc = calculate_auc(combined_y_true, combined_y_prob)
        metrics["auc_roc"] = roc_auc
        metrics["pr_auc"] = pr_auc
        
        report["per_type"][t] = {
            "total_pairs": len(positives) + len(negatives),
            "positive_pairs": len(positives),
            "negative_pairs": len(negatives),
            **metrics,
            "details": details + neg_details
        }
        
        global_tp += tp
        global_fp += neg_fp
        global_tn += neg_tn
        global_fn += fn

        # Collect global prob tracking directly here
        global_y_true.extend(y_true + neg_y_true)
        global_y_prob.extend(y_prob + neg_y_prob)


    # Save true labels and probabilities for visualization
    report["global_y_true"] = global_y_true
    report["global_y_prob"] = global_y_prob

    if auto_thresh:
        print(f"\n  {Colors.MAGENTA}[Auto-Threshold] Tip Bazlı Optimizasyon Başlıyor...{Colors.RESET}")
        from sklearn.metrics import f1_score
        
        best_thresholds = {}
        for t in types:
            if t in report["per_type"]:
                rm = report["per_type"][t]
                y_t = [d["label"] for d in rm["details"]]
                y_p = [d["probability"] for d in rm["details"]]
                
                # Sadece bu type icin y_true ve y_prob kaydet (grafikler icin)
                rm["y_true"] = y_t
                rm["y_prob"] = y_p
                
                best_f1 = -1
                best_th = threshold
                for th in [i/100.0 for i in range(1, 100)]:
                    preds = [1 if p >= th else 0 for p in y_p]
                    f1 = f1_score(y_t, preds, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_th = th
                
                best_thresholds[t] = best_th
                print(f"  {Colors.BLUE}→{Colors.RESET} {t.upper()} en iyi eşik: {Colors.YELLOW}{best_th:.2f}{Colors.RESET} (Maksimum F1: {best_f1:.4f})")
                
                # Recalculate metrics for this type
                tp, fp, tn, fn = 0, 0, 0, 0
                for d in rm["details"]:
                    pred = 1 if d["probability"] >= best_th else 0
                    d["prediction"] = pred
                    if d["label"] == 1 and pred == 1: tp += 1
                    elif d["label"] == 1 and pred == 0: fn += 1
                    elif d["label"] == 0 and pred == 1: fp += 1
                    elif d["label"] == 0 and pred == 0: tn += 1
                rm.update(calculate_metrics(tp, fp, tn, fn))
                rm["best_threshold"] = best_th
                
        # Global recalculation
        global_tp, global_fp, global_tn, global_fn = 0, 0, 0, 0
        for t in types:
            if t in report["per_type"]:
                rm = report["per_type"][t]
                global_tp += rm["tp"]
                global_fp += rm["fp"]
                global_tn += rm["tn"]
                global_fn += rm["fn"]
    else:
        # Save arrays even if not auto-thresh
        for t in types:
            if t in report["per_type"]:
                rm = report["per_type"][t]
                rm["y_true"] = [d["label"] for d in rm["details"]]
                rm["y_prob"] = [d["probability"] for d in rm["details"]]
                rm["best_threshold"] = threshold

    global_metrics = calculate_metrics(global_tp, global_fp, global_tn, global_fn)
    roc_auc, pr_auc = calculate_auc(global_y_true, global_y_prob)
    global_metrics["auc_roc"] = roc_auc
    global_metrics["pr_auc"] = pr_auc
    global_metrics["total_pairs"] = len(global_y_true)
    
    report["global"] = global_metrics

    # Save Results
    out_dir = os.path.join(_PROJECT_ROOT, "evaluation", "test_results", f"run_{int(time.time())}")
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
                for k in ["total_pairs", "precision", "recall", "f1_score", "accuracy", "mcc", "auc_roc", "pr_auc", "tp", "fp", "tn", "fn"]:
                    f.write(f"{k.ljust(20)}: {rm.get(k)}\n")

    Log.success(f"Results saved to: {out_dir}")
    
    # --- CONSOLE PRETTY PRINT ---
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*85}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD} 🧪 EXPERIMENT RESULTS: {os.path.basename(exp_path)} (Threshold: {threshold:.2f}) {Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*85}{Colors.RESET}")
    
    # HEADER
    print(f"{Colors.WHITE}{Colors.BOLD}{'TYPE':<10} | {'THRESH':<6} | {'PRECISION':<9} | {'RECALL':<8} | {'F1-SCORE':<8} | {'PR-AUC':<8} | {'AUC-ROC':<8} | {'TP':<5} | {'FP':<5} | {'TN':<5} | {'FN':<5}{Colors.RESET}")
    print(f"{Colors.DIM}{'-' * 115}{Colors.RESET}")
    
    types_to_print = ["global"] + [t for t in types if t in report["per_type"]]
    
    for t in types_to_print:
        if t == "global":
            rm = report["global"]
            row_name = f"{Colors.MAGENTA}{Colors.BOLD}GLOBAL{Colors.RESET}"
        else:
            rm = report["per_type"][t]
            row_name = f"{Colors.YELLOW}{t.upper()}{Colors.RESET}"
            
        f1 = f"{rm.get('f1_score', 0):.4f}"
        mcc = f"{rm.get('mcc', 0):.4f}"
        prauc = f"{rm.get('pr_auc', 0):.4f}"
        
        aucroc_val = rm.get('auc_roc', 0)
        # Highlight high AUC-ROC with Green
        if aucroc_val > 0.90:
            aucroc = f"{Colors.GREEN}{aucroc_val:.4f}{Colors.RESET}"
        elif aucroc_val > 0.80:
            aucroc = f"{Colors.BLUE}{aucroc_val:.4f}{Colors.RESET}"
        else:
            aucroc = f"{aucroc_val:.4f}"
            
        tp = f"{Colors.GREEN}{rm.get('tp',0)}{Colors.RESET}"
        fp = f"{Colors.RED}{rm.get('fp',0)}{Colors.RESET}"
        tn = f"{Colors.GREEN}{rm.get('tn',0)}{Colors.RESET}"
        fn = f"{Colors.RED}{rm.get('fn',0)}{Colors.RESET}"
        
        th_val = rm.get('best_threshold', threshold)
        th_str = f"{Colors.YELLOW}{th_val:.2f}{Colors.RESET}" if auto_thresh and t != "global" else f"{th_val:.2f}"
        
        prec = f"{rm.get('precision', 0):.4f}"
        rec = f"{rm.get('recall', 0):.4f}"
        
        print(f"{row_name:<19} | {th_str:<15} | {prec:<9} | {rec:<8} | {f1:<8} | {prauc:<8} | {aucroc:<17} | {tp:<14} | {fp:<14} | {tn:<14} | {fn:<14}")
        
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*115}{Colors.RESET}\n") + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run automation tests on code clone models.")
    parser.add_argument("--exp-id", type=int, default=None, help="Specific experiment ID to test (e.g. 54). If not provided, uses latest.")
    parser.add_argument("--threshold", type=float, default=0.95, help="Classification threshold (default 0.95)")
    parser.add_argument("--scenario", type=str, default="original", choices=["original", "imbalanced", "balanced", "all"], help="Test scenario folder to use")
    parser.add_argument("--auto-threshold", action="store_true", help="Automatically find the best threshold using PR-AUC/F1")
    args = parser.parse_args()
    
    scenarios = ["original", "imbalanced", "balanced"] if args.scenario == "all" else [args.scenario]
    for sc in scenarios:
        print(f"\n{'='*50}\n 🧪 RUNNING SCENARIO: {sc.upper()}\n{'='*50}")
        run_automation(test_dir=f"evaluation/test_clones_{sc}", threshold=args.threshold, exp_id=args.exp_id, auto_thresh=args.auto_threshold)
