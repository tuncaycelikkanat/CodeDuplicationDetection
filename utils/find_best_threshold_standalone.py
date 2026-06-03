"""
Standalone Best Threshold Finder
---------------------------------
JSON + numpy only — sklearn bağımlılığı yok.
Threshold adımı 0.01 hassasiyetiyle tip bazında en iyi F1 bulur.
"""
import os
import sys
import json
import argparse

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np


def find_best_threshold(run_dir=None, step=0.01):
    results_dir = os.path.join(_PROJECT_ROOT, "evaluation", "test_results")
    if run_dir:
        report_path = os.path.join(run_dir, "report.json")
        latest_run = os.path.basename(run_dir)
    else:
        if not os.path.exists(results_dir):
            print("[ERROR] No test results found.")
            return
        runs = sorted([d for d in os.listdir(results_dir) if d.startswith("run_")])
        if not runs:
            print("[ERROR] No test runs found.")
            return
        latest_run = runs[-1]
        report_path = os.path.join(results_dir, latest_run, "report.json")

    if not os.path.exists(report_path):
        print(f"[ERROR] No report.json in {latest_run}")
        return

    with open(report_path, "r") as f:
        report = json.load(f)

    print(f"\n{'='*80}")
    print(f"  Threshold Optimization  |  Run: {latest_run}  |  Step: {step:.2f}")
    print(f"{'='*80}")

    types = ["type1", "type2", "type3", "type4"]
    best_thresholds = {}

    for t in types:
        if t not in report.get("per_type", {}):
            continue

        rm = report["per_type"][t]
        y_t_raw = rm.get("y_true", [d["label"] for d in rm.get("details", [])])
        y_p_raw = rm.get("y_prob", [d["probability"] for d in rm.get("details", [])])

        if not y_t_raw:
            continue

        y_t = np.array([int(x) for x in y_t_raw])
        y_p = np.array([float(x) for x in y_p_raw])
        n = len(y_t)

        print(f"\n  ┌─ {t.upper()}  ({n} pairs) ───────────────────────────────────────────────────")
        print(f"  │  {'Thresh':>6} │ {'Precision':>9} │ {'Recall':>6} │ {'F1-Score':>8} │ {'TP':>5} │ {'FP':>5} │ {'TN':>5} │ {'FN':>5} │")
        print(f"  │  {'-'*6}-+-{'-'*9}-+-{'-'*6}-+-{'-'*8}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-│")

        best_f1 = -1.0
        best_thresh_f1 = 0.0
        rows = []

        # İlk geçiş: tüm threshold'ları hesapla ve en iyiyi bul
        n_steps = int(round((1.0 - step) / step))
        for i in range(1, n_steps + 1):
            thresh = round(i * step, 4)
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

            rows.append((thresh, precision, recall, f1, tp, fp, tn, fn))

        # İkinci geçiş: tabloyu yazdır, en iyiyi işaretle
        for thresh, precision, recall, f1, tp, fp, tn, fn in rows:
            is_best = abs(thresh - best_thresh_f1) < 1e-6
            marker = " ◄ BEST" if is_best else ""
            prefix = "  │  "
            if is_best:
                # ANSI yeşil (terminal'de görünür)
                line = (f"{prefix}\033[92m{thresh:.2f}   │ {precision:.4f}    │ {recall:.4f} │ {f1:.4f}   │ {tp:>5} │ {fp:>5} │ {tn:>5} │ {fn:>5} │{marker}\033[0m")
            else:
                line = (f"{prefix}{thresh:.2f}   │ {precision:.4f}    │ {recall:.4f} │ {f1:.4f}   │ {tp:>5} │ {fp:>5} │ {tn:>5} │ {fn:>5} │")
            print(line)

        best_thresholds[t] = best_thresh_f1
        print(f"  └─ \033[95m>>> Best {t.upper()} Threshold: {best_thresh_f1:.2f}  (F1: {best_f1:.4f})\033[0m")

    print(f"\n{'='*80}")
    print(f"  ÖNERILEN THRESHOLD YAPISI (config.py için):")
    print(f"{'='*80}")
    print("  thresholds = {")
    for t, th in best_thresholds.items():
        print(f"      '{t}': {th:.2f},")
    print("  }")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find best threshold per clone type (step=0.01).")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Path to a specific run_XXX directory (optional, uses latest if omitted)")
    parser.add_argument("--step", type=float, default=0.01,
                        help="Threshold search step size (default: 0.01)")
    args = parser.parse_args()
    find_best_threshold(args.run_dir, step=args.step)
