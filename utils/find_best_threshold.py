import os
import sys
import json
import math
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def find_best_threshold():
    # Find latest test results
    results_dir = os.path.join(_PROJECT_ROOT, "test_results")
    if not os.path.exists(results_dir):
        print("No test results found.")
        return

    runs = sorted([d for d in os.listdir(results_dir) if d.startswith("run_")])
    if not runs:
        print("No test runs found.")
        return

    latest_run = runs[-1]
    report_path = os.path.join(results_dir, latest_run, "report.json")

    if not os.path.exists(report_path):
        print(f"No report.json in {latest_run}")
        return

    with open(report_path, "r") as f:
        report = json.load(f)

    # Extract all unique pairs and their probabilities
    pair_data = {}
    for t_name, t_data in report.get("per_type", {}).items():
        for detail in t_data.get("details", []):
            pair_name = detail["pair"]
            pair_data[pair_name] = {
                "label": detail["label"],
                "prob": detail["probability"]
            }

    if not pair_data:
        print("No detail data found in the report.")
        return

    y_true = np.array([d["label"] for d in pair_data.values()])
    y_prob = np.array([d["prob"] for d in pair_data.values()])

    print(f"Total unique pairs found: {len(y_true)}")
    print(f"Positive pairs: {sum(y_true == 1)}, Negative pairs: {sum(y_true == 0)}")

    # Bug #2 düzeltildi: tek bir threshold aralığı tanımlanıyor (önceden 0.00-1.00
    # tanımlanıp hemen ardından 0.90-1.00 ile üzerine yazılıyordu).
    thresholds = np.arange(0.00, 1.01, 0.01)

    best_f1 = -1.0
    best_thresh_f1 = 0.0
    best_mcc = -2.0
    best_thresh_mcc = 0.0

    print("\n--- Threshold Optimization ---")
    print("\n| Threshold | Precision | Recall | F1-Score | Accuracy | MCC | TP | FP | TN | FN |")
    print("|---|---|---|---|---|---|---|---|---|---|")

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        acc       = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = math.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0.0

        print(f"| {thresh:.2f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {acc:.4f} | {mcc:.4f} | {tp} | {fp} | {tn} | {fn} |")

        # Bug #5 & #20 düzeltildi: en iyi değerler güncelleniyor
        if f1 > best_f1:
            best_f1 = f1
            best_thresh_f1 = thresh
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh_mcc = thresh

    print("\n" + "=" * 50)
    print(f"  Best F1={best_f1:.4f}  @ threshold={best_thresh_f1:.2f}")
    print(f"  Best MCC={best_mcc:.4f} @ threshold={best_thresh_mcc:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    find_best_threshold()
