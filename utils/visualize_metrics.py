import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
from utils.logger import Log, Colors

def load_report(run_dir):
    report_path = os.path.join(run_dir, "report.json")
    if not os.path.exists(report_path):
        Log.error(f"report.json not found in {run_dir}")
        return None
    with open(report_path, "r") as f:
        return json.load(f)

def plot_pr_curve(report, out_dir):
    plt.figure(figsize=(10, 8))
    
    types = ["type1", "type2", "type3", "type4"]
    colors = ['blue', 'green', 'orange', 'red']
    
    for t, color in zip(types, colors):
        if t in report["per_type"]:
            y_t = report["per_type"][t].get("y_true", [])
            y_p = report["per_type"][t].get("y_prob", [])
            if not y_t: continue
            
            precision, recall, _ = precision_recall_curve(y_t, y_p)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, color=color, lw=2, label=f'{t.upper()} (PR-AUC = {pr_auc:.4f})')
            
    if "global_y_true" in report:
        precision, recall, _ = precision_recall_curve(report["global_y_true"], report["global_y_prob"])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color='black', lw=3, linestyle='--', label=f'GLOBAL (PR-AUC = {pr_auc:.4f})')

    plt.xlabel('Recall (Kapsayıcılık)', fontsize=14)
    plt.ylabel('Precision (Hassasiyet)', fontsize=14)
    plt.title('Precision-Recall (PR) Curve', fontsize=16)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(out_dir, "pr_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    Log.substep(f"Saved: {save_path}")

def plot_roc_curve(report, out_dir):
    plt.figure(figsize=(10, 8))
    
    types = ["type1", "type2", "type3", "type4"]
    colors = ['blue', 'green', 'orange', 'red']
    
    for t, color in zip(types, colors):
        if t in report["per_type"]:
            y_t = report["per_type"][t].get("y_true", [])
            y_p = report["per_type"][t].get("y_prob", [])
            if not y_t: continue
            
            fpr, tpr, _ = roc_curve(y_t, y_p)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=color, lw=2, label=f'{t.upper()} (AUC = {roc_auc:.4f})')
            
    if "global_y_true" in report:
        fpr, tpr, _ = roc_curve(report["global_y_true"], report["global_y_prob"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='black', lw=3, linestyle='--', label=f'GLOBAL (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    Log.substep(f"Saved: {save_path}")

def plot_threshold_tradeoff(report, out_dir):
    plt.figure(figsize=(10, 8))
    
    # We will plot Tradeoff just for Global to keep it readable
    if "global_y_true" not in report:
        return
        
    y_t = np.array(report["global_y_true"])
    y_p = np.array(report["global_y_prob"])
    
    thresholds = np.linspace(0.01, 0.99, 100)
    f1_scores = []
    precisions = []
    recalls = []
    
    for th in thresholds:
        preds = (y_p >= th).astype(int)
        tp = np.sum((preds == 1) & (y_t == 1))
        fp = np.sum((preds == 1) & (y_t == 0))
        fn = np.sum((preds == 0) & (y_t == 1))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        f1_scores.append(f1)
        precisions.append(prec)
        recalls.append(rec)
        
    plt.plot(thresholds, f1_scores, color='green', lw=2, label='F1 Score')
    plt.plot(thresholds, precisions, color='blue', lw=2, linestyle=':', label='Precision')
    plt.plot(thresholds, recalls, color='orange', lw=2, linestyle='-.', label='Recall')
    
    # Find max F1
    max_f1_idx = np.argmax(f1_scores)
    max_th = thresholds[max_f1_idx]
    plt.axvline(x=max_th, color='red', linestyle='--', label=f'Best Threshold = {max_th:.2f}')

    plt.xlabel('Decision Threshold', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Threshold vs F1/Precision/Recall (GLOBAL)', fontsize=16)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(out_dir, "threshold_tradeoff.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    Log.substep(f"Saved: {save_path}")

def plot_confusion_matrices(report, out_dir):
    types = [t for t in ["type1", "type2", "type3", "type4"] if t in report["per_type"]]
    
    if not types: return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, t in enumerate(types):
        rm = report["per_type"][t]
        tp, fp, tn, fn = rm.get("tp",0), rm.get("fp",0), rm.get("tn",0), rm.get("fn",0)
        
        # Build matrix: [[TN, FP], [FN, TP]]
        cm = np.array([[tn, fp], [fn, tp]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                    xticklabels=['Non-Clone (Pred)', 'Clone (Pred)'],
                    yticklabels=['Non-Clone (True)', 'Clone (True)'],
                    annot_kws={"size": 14})
        
        best_th = rm.get('best_threshold', 0)
        axes[i].set_title(f'{t.upper()} Confusion Matrix (Thresh: {best_th:.2f})', fontsize=14)
        
    plt.tight_layout()
    save_path = os.path.join(out_dir, "confusion_matrices.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    Log.substep(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Metrics from report.json")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the test_results run directory")
    args = parser.parse_args()
    
    Log.step(f"Visualizing metrics for run: {args.run_dir}")
    report = load_report(args.run_dir)
    
    if report:
        # Create 'plots' folder inside the run_dir
        plots_dir = os.path.join(args.run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_pr_curve(report, plots_dir)
        plot_roc_curve(report, plots_dir)
        plot_threshold_tradeoff(report, plots_dir)
        plot_confusion_matrices(report, plots_dir)
        
        Log.success(f"All plots saved successfully in {plots_dir}/")
