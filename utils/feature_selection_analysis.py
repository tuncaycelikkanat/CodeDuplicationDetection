import os
import sys
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_selection import SelectFromModel

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.logger import Log, Colors
from utils.test_automation import get_experiment_path
from utils.feature_pipeline import build_pair_vector

def load_pairs_for_type(test_dir, t_name):
    """Loads positives and negatives for a specific type from test_dir."""
    pairs = []
    
    # Load positives
    pos_dir = os.path.join(test_dir, t_name)
    if os.path.exists(pos_dir):
        for f in os.listdir(pos_dir):
            if f.endswith(".json"):
                with open(os.path.join(pos_dir, f), "r") as fp:
                    data = json.load(fp)
                    data['label'] = 1
                    pairs.append(data)
                    
    # Load negatives (assuming they are in 'negatives' or mapped)
    # Balanced test set typically has 'negatives' folder or specific negs.
    # In test_automation.py, we mapped negatives by taking subsets.
    # We will just load all negatives and take a subset equal to positives.
    neg_dir = os.path.join(test_dir, "negatives")
    negs = []
    if os.path.exists(neg_dir):
        for f in os.listdir(neg_dir):
            if f.endswith(".json"):
                with open(os.path.join(neg_dir, f), "r") as fp:
                    data = json.load(fp)
                    data['label'] = 0
                    negs.append(data)
                    
    # Balance negatives
    if len(pos_dir) > 0 and len(negs) >= len(pairs):
        # Deterministic slice
        negs = sorted(negs, key=lambda x: x.get('p_name', ''))[:len(pairs)]
    
    pairs.extend(negs)
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Perform Feature Selection Analysis per Clone Type.")
    parser.add_argument("--exp-id", type=int, required=True, help="Experiment ID to load vectorizers/SSL from (e.g. 78)")
    parser.add_argument("--test-dir", type=str, default="evaluation/test_clones_balanced", help="Directory containing test clones")
    args = parser.parse_args()

    exp_path = get_experiment_path(args.exp_id)
    if not exp_path:
        Log.error(f"Experiment ID {args.exp_id} not found.")
        return

    Log.step(f"Loading feature extraction components from: {os.path.basename(exp_path)}")
    
    with open(os.path.join(exp_path, "tfidf.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
        
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

    # Output storage for the table
    results_table = {}
    types = ["type1", "type2", "type3", "type4"]
    
    print(f"\n{Colors.MAGENTA}{Colors.BOLD}[Feature Selection Analysis Started]{Colors.RESET}\n")

    for t in types:
        Log.substep(f"Processing {t.upper()}...")
        pairs = load_pairs_for_type(args.test_dir, t)
        if not pairs:
            Log.warning(f"No pairs found for {t}")
            continue
            
        # Extract features
        X_list, y_list = [], []
        for p in tqdm(pairs, desc=f"Extracting 117 features ({t})"):
            vec = build_pair_vector(
                p['c1'], p['c2'], vectorizer, 
                svd_model=None, 
                ssl_pipeline=ssl_pipeline, ssl_pca=ssl_pca
            )
            X_list.append(vec[0]) # build_pair_vector returns shape (1, 117)
            y_list.append(p['label'])
            
        X = np.array(X_list)
        y = np.array(y_list)
        
        orig_feature_count = X.shape[1]
        
        # 1. Baseline Model (All features)
        # We use a fast Random Forest with Cross Validation to get stable metrics
        rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Train on full dataset to get importances
        rf_base.fit(X, y)
        acc_base = np.mean(cross_val_score(rf_base, X, y, cv=cv, scoring='accuracy'))
        f1_base = np.mean(cross_val_score(rf_base, X, y, cv=cv, scoring='f1'))
        
        # 2. Feature Selection
        # Select features that have importance > median importance
        selector = SelectFromModel(rf_base, prefit=True, threshold="median")
        X_selected = selector.transform(X)
        selected_feature_count = X_selected.shape[1]
        
        # 3. New Model (Selected features)
        rf_new = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        acc_new = np.mean(cross_val_score(rf_new, X_selected, y, cv=cv, scoring='accuracy'))
        f1_new = np.mean(cross_val_score(rf_new, X_selected, y, cv=cv, scoring='f1'))
        
        results_table[t] = {
            "base_f": orig_feature_count,
            "base_acc": acc_base,
            "base_f1": f1_base,
            "sel_f": selected_feature_count,
            "sel_acc": acc_new,
            "sel_f1": f1_new
        }

    # Print requested table
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.RESET}")
    print(f"{Colors.WHITE}{Colors.BOLD}TYPE  | NORMAL FEATURE (117)          | FEATURE SELECTION{Colors.RESET}")
    print(f"      | Acc     F1      F.Number  | Acc     F1      F.Number")
    print(f"{Colors.CYAN}{'-'*80}{Colors.RESET}")
    
    for t in types:
        if t not in results_table: continue
        res = results_table[t]
        
        t_num = t.replace('type', '')
        b_acc = f"{res['base_acc']:.4f}"
        b_f1  = f"{res['base_f1']:.4f}"
        b_f   = f"{res['base_f']:<8}"
        
        s_acc = f"{res['sel_acc']:.4f}"
        s_f1  = f"{res['sel_f1']:.4f}"
        s_f   = f"{res['sel_f']:<8}"
        
        # Highlight if selection improved or stayed same
        if res['sel_f1'] >= res['base_f1']:
            s_f1 = f"{Colors.GREEN}{s_f1}{Colors.RESET}"
            
        print(f"{t_num:<5} | {b_acc:<7} {b_f1:<7} {b_f}  | {s_acc:<7} {s_f1:<7} {s_f}")
        
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.RESET}\n")

if __name__ == "__main__":
    main()
