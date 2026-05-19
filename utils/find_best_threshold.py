import os
import sys
import json
import math
import numpy as np
from sklearn.metrics import f1_score
import argparse

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.logger import Log, Colors

def find_best_threshold(run_dir=None):
    # Find latest test results
    results_dir = os.path.join(_PROJECT_ROOT, "evaluation", "test_results")
    if run_dir:
        report_path = os.path.join(run_dir, "report.json")
        latest_run = os.path.basename(run_dir)
    else:
        if not os.path.exists(results_dir):
            Log.error("No test results found.")
            return

        runs = sorted([d for d in os.listdir(results_dir) if d.startswith("run_")])
        if not runs:
            Log.error("No test runs found.")
            return

        latest_run = runs[-1]
        report_path = os.path.join(results_dir, latest_run, "report.json")

    if not os.path.exists(report_path):
        Log.error(f"No report.json in {latest_run}")
        return

    with open(report_path, "r") as f:
        report = json.load(f)

    Log.step(f"Analyzing thresholds for run: {latest_run}")
    
    types = ["type1", "type2", "type3", "type4"]
    best_thresholds = {}

    for t in types:
        if t in report.get("per_type", {}):
            rm = report["per_type"][t]
            y_t = np.array(rm.get("y_true", [d["label"] for d in rm.get("details", [])]))
            y_p = np.array(rm.get("y_prob", [d["probability"] for d in rm.get("details", [])]))
            
            if len(y_t) == 0:
                continue
                
            Log.substep(f"Optimizing for {Colors.YELLOW}{t.upper()}{Colors.RESET} ({len(y_t)} pairs)...")
            
            best_f1 = -1.0
            best_thresh_f1 = 0.0
            
            # Print a mini tradeoff table for this type
            print(f"\n  {Colors.WHITE}{Colors.BOLD}| {'Thresh':<6} | {'Precision':<9} | {'Recall':<6} | {'F1-Score':<8} | {'TP':<4} | {'FP':<4} | {'TN':<4} | {'FN':<4} |{Colors.RESET}")
            print(f"  {Colors.DIM}|--------|-----------|--------|----------|------|------|------|------|{Colors.RESET}")
            
            for thresh in np.arange(0.05, 1.00, 0.05):
                y_pred = (y_p >= thresh).astype(int)
                
                tp = int(np.sum((y_t == 1) & (y_pred == 1)))
                fp = int(np.sum((y_t == 0) & (y_pred == 1)))
                tn = int(np.sum((y_t == 0) & (y_pred == 0)))
                fn = int(np.sum((y_t == 1) & (y_pred == 0)))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh_f1 = thresh
                    
                # Highlight the current best row or regular row
                if f1 == best_f1 and f1 > 0:
                    row_color = Colors.GREEN
                else:
                    row_color = ""
                    
                print(f"  {row_color}| {thresh:.2f}   | {precision:.4f}  | {recall:.4f} | {f1:.4f}   | {tp:<4} | {fp:<4} | {tn:<4} | {fn:<4} |{Colors.RESET}")
                
            best_thresholds[t] = best_thresh_f1
            print(f"\n  {Colors.MAGENTA}>>> Best {t.upper()} Threshold: {best_thresh_f1:.2f} (F1: {best_f1:.4f}){Colors.RESET}\n")

    Log.success("Optimization Complete.")
    Log.info("Suggested threshold configuration for production:")
    print("thresholds = {")
    for t, th in best_thresholds.items():
        print(f"    '{t}': {th:.2f},")
    print("}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find best threshold per type from test results.")
    parser.add_argument("--run-dir", type=str, default=None, help="Path to a specific run_XXX directory (optional)")
    args = parser.parse_args()
    
    find_best_threshold(args.run_dir)
