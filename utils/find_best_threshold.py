import os
import sys
import json
import numpy as np

def find_best_threshold():
    # Find latest test results
    results_dir = "test_results"
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
            # Save label and probability
            pair_data[pair_name] = {
                "label": detail["label"],
                "prob": detail["probability"]
            }
            
    if not pair_data:
        print("No detail data found in the report.")
        return
        
    y_true = []
    y_prob = []
    for data in pair_data.values():
        y_true.append(data["label"])
        y_prob.append(data["prob"])
        
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    print(f"Total unique pairs found: {len(y_true)}")
    print(f"Positive pairs: {sum(y_true == 1)}, Negative pairs: {sum(y_true == 0)}")
    
    best_f1 = 0
    best_thresh_f1 = 0
    best_mcc = -1
    best_thresh_mcc = 0
    best_acc = 0
    
    print("\n--- Threshold Optimization ---")
    
    # Check thresholds from 0.00 to 1.00
    thresholds = np.arange(0.00, 1.01, 0.01)
    
    print("\n| Threshold | Precision | Recall | F1-Score | Accuracy | MCC | TP | FP | TN | FN |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    
    thresholds = np.arange(0.90, 1.00, 0.01)
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        acc = (tp + tn) / (tp + fp + tn + fn)
        
        num = (tp * tn) - (fp * fn)
        den = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
        mcc = num / den if den > 0 else 0
        
        print(f"| {thresh:.2f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {acc:.4f} | {mcc:.4f} | {tp} | {fp} | {tn} | {fn} |")

if __name__ == "__main__":
    find_best_threshold()
