import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.logger import Log
from utils.feature_pipeline import build_pair_vector
from pairing.pair_generator import load_pairs
from config import COS_TOKEN_IDX
from sklearn.inspection import permutation_importance
import joblib

def load_latest_report(results_dir):
    runs = glob.glob(os.path.join(results_dir, "run_*"))
    if not runs:
        return None, None
    latest_run = max(runs, key=os.path.getctime)
    report_path = os.path.join(latest_run, "report.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            return latest_run, json.load(f)
    return latest_run, None

def analyze_errors(report_path=None):
    results_dir = os.path.join(_PROJECT_ROOT, "evaluation", "test_results")
    if report_path is None:
        run_dir, report = load_latest_report(results_dir)
        if report is None:
            Log.error("No report.json found in test_results.")
            return
    else:
        run_dir = os.path.dirname(report_path)
        with open(report_path, "r") as f:
            report = json.load(f)

    Log.step(f"Analyzing run: {run_dir}")
    
    # Extract feature names
    from config import STAGE1_FEATURE_COUNT
    feature_names = [
        "cos_token", "length_ratio", "manhattan_token", "euclidean_token",
    ] + [f"ast_ratio_{i}" for i in range(20)] + [f"ast_diff_{i}" for i in range(20)] + [
        "cf_sim", "sem_lib_calls", "sem_lib_cats", "sem_data_structs", 
        "sem_io_pattern", "sem_math_ops", "sem_skeleton", "sem_abstract_cf", "type_profile_cos"
    ]
    
    analysis_file = os.path.join(run_dir, "error_analysis_and_features.md")
    
    with open(analysis_file, "w") as f:
        f.write("# Error Analysis & Feature Importance\n\n")
        f.write("Bu rapor, modelin en yüksek özgüvenle (probability) yanıldığı **False Positive** (yanlış alarm) ve **False Negative** (kaçırılan klon) vakalarını inceler.\n\n")
        
        test_dir = os.path.join(_PROJECT_ROOT, "dataset", "test")
        
        # Sadece tiplerin içindeki detayları al
        if "per_type" not in report:
            Log.error("No 'per_type' data in report.")
            return
            
        for t, metrics in report["per_type"].items():
            f.write(f"## {t.upper()}\n\n")
            details = metrics.get("details", [])
            if not details:
                continue
                
            # FP and FN extraction
            # FP: label=0, prediction=1
            fps = [d for d in details if d["label"] == 0 and d["prediction"] == 1]
            # FN: label=1, prediction=0
            fns = [d for d in details if d["label"] == 1 and d["prediction"] == 0]
            
            # Sort FPs by probability descending (Modelin en emin olduğu ama yanıldığı negatifler)
            fps = sorted(fps, key=lambda x: x["probability"], reverse=True)[:5]
            
            # Sort FNs by probability ascending (Modelin kesin negatif dediği ama aslında pozitif olanlar)
            fns = sorted(fns, key=lambda x: x["probability"])[:5]
            
            # Load raw codes to compute features and show them
            pos_pairs_dict = {p['p_name']: p for p in load_pairs(os.path.join(test_dir, t), label=1)}
            neg_pairs_dict = {p['p_name']: p for p in load_pairs(os.path.join(test_dir, t), label=0)}
            
            def render_examples(title, examples, pairs_dict):
                f.write(f"### {title}\n")
                if not examples:
                    f.write("None found.\n\n")
                    return
                for ex in examples:
                    pair_name = ex["pair"]
                    f.write(f"**Pair**: `{pair_name}` | **Model Probability**: `{ex['probability']:.4f}`\n")
                    # Try to get raw code
                    p = pairs_dict.get(pair_name)
                    if p:
                        f.write("```cpp\n// --- CODE 1 ---\n" + p['c1'][:200].replace('\n', ' ') + "...\n")
                        f.write("// --- CODE 2 ---\n" + p['c2'][:200].replace('\n', ' ') + "...\n```\n")
                    f.write("\n")
            
            render_examples("🔴 Top 5 False Positives (En Yüksek Olasılıklı Yanlış Alarmlar)", fps, neg_pairs_dict)
            render_examples("🟡 Top 5 False Negatives (En Düşük Olasılıklı Kaçırılan Klonlar)", fns, pos_pairs_dict)
            
    Log.step(f"Analysis saved to: {analysis_file}")
    
    # ---------------------------------------------------------
    # Feature Importance (Permutation Importance Per Type)
    # ---------------------------------------------------------
    Log.step("Calculating Feature Importances Per Type...")
    exp_dir = report.get("experiment_dir", "")
    if not exp_dir:
        exp_dir = os.path.join(_PROJECT_ROOT, "experiments", "exp_070_CASCADE_Ensemble_800k")
        
    model_path = os.path.join(exp_dir, "model.pkl")
    vectorizer_path = os.path.join(exp_dir, "vectorizer.pkl")
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        with open(analysis_file, "a") as f:
            f.write("## 🏆 Feature Importances (Tip Bazlı Permutation Importance)\n\n")
            f.write("Aşağıdaki tablolar, modelin kararlarını verirken hangi tip klonlarda hangi özelliklere (Lexical, AST, Semantic vb.) güvendiğini gösterir.\n\n")
        
        for t in ["type1", "type2", "type3", "type4"]:
            Log.step(f"Calculating Feature Importance for {t.upper()}...")
            pos = load_pairs(os.path.join(test_dir, t), label=1)[:100]
            neg = load_pairs(os.path.join(test_dir, t), label=0)[:100]
            t_samples = pos + neg
            
            if not t_samples:
                continue
                
            X_list, y_list = [], []
            for p in tqdm(t_samples, desc=f"Building {t} subset"):
                v = build_pair_vector(p['c1'], p['c2'], vectorizer)
                X_list.append(v[0]) 
                y_list.append(p['label'])
                
            X_sample = np.array(X_list)
            y_sample = np.array(y_list)
            
            Log.substep(f"Running permutation importance for {t}...")
            r = permutation_importance(model, X_sample, y_sample, n_repeats=5, random_state=42, n_jobs=-1)
            
            importances = []
            for i in r.importances_mean.argsort()[::-1]:
                if r.importances_mean[i] > 0.001:  # Sadece anlamlı etkisi olanlar
                    name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                    importances.append((name, r.importances_mean[i], r.importances_std[i]))
                    
            df_imp = pd.DataFrame(importances, columns=["Feature", "Importance_Mean", "Importance_Std"])
            csv_path = os.path.join(run_dir, f"feature_importance_{t}.csv")
            df_imp.to_csv(csv_path, index=False)
            
            with open(analysis_file, "a") as f:
                f.write(f"### {t.upper()} Feature Importance\n\n")
                f.write("| Feature | Importance Mean | Std |\n")
                f.write("|---------|-----------------|-----|\n")
                for _, row in df_imp.head(15).iterrows():
                    f.write(f"| {row['Feature']} | {row['Importance_Mean']:.4f} | {row['Importance_Std']:.4f} |\n")
                f.write("\n")
            
            Log.substep(f"Saved {t} importance to {csv_path}")

        print(f"\n✅ All analysis completed. See:\n  - {analysis_file}\n  - And 4 CSV files in {run_dir}")

if __name__ == "__main__":
    analyze_errors()
