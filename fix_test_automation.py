import sys
import re

with open("utils/test_automation.py", "r") as f:
    content = f.read()

# Replace scenario choices
content = content.replace('choices=["original", "imbalanced", "balanced"]', 'choices=["original", "imbalanced", "balanced", "all"]')

# Add --auto-threshold argument
arg_insert = 'parser.add_argument("--auto-threshold", action="store_true", help="Automatically find the best threshold using PR-AUC/F1")'
content = content.replace('args = parser.parse_args()', arg_insert + '\n    args = parser.parse_args()')

# Replace the run_automation call to handle multiple scenarios
main_block_old = """    run_automation(test_dir=f"test_clones_{args.scenario}", threshold=args.threshold, exp_id=args.exp_id)"""
main_block_new = """    scenarios = ["original", "imbalanced", "balanced"] if args.scenario == "all" else [args.scenario]
    for sc in scenarios:
        print(f"\\n{'='*50}\\n 🧪 RUNNING SCENARIO: {sc.upper()}\\n{'='*50}")
        run_automation(test_dir=f"test_clones_{sc}", threshold=args.threshold, exp_id=args.exp_id, auto_thresh=args.auto_threshold)"""
content = content.replace(main_block_old, main_block_new)

# Add auto_thresh argument to run_automation
content = content.replace('def run_automation(test_dir="test_clones", threshold=0.95, exp_id=None):', 'def run_automation(test_dir="test_clones", threshold=0.95, exp_id=None, auto_thresh=False):')

# In run_automation, after computing all probabilities, find best threshold if auto_thresh is True
# We need to inject code after we collect global_y_true and global_y_prob.
# Line 316 is: global_y_true = all_pos_true + neg_y_true
# Line 317 is: global_y_prob = all_pos_prob + neg_y_prob

injection = """
    if auto_thresh:
        print("\\n  [Auto-Threshold] Bulunuyor...")
        best_f1 = -1
        best_t = threshold
        from sklearn.metrics import f1_score
        for t in [i/100.0 for i in range(1, 100)]:
            preds = [1 if p >= t else 0 for p in global_y_prob]
            f1 = f1_score(global_y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        print(f"  [Auto-Threshold] En iyi eşik: {best_t:.2f} (F1: {best_f1:.4f})")
        threshold = best_t
        
        # Yeniden hesapla
        global_tp, global_fp, global_tn, global_fn = 0, 0, 0, 0
        for t in types:
            if t in report["per_type"]:
                rm = report["per_type"][t]
                details = rm["details"]
                tp, fp, tn, fn = 0, 0, 0, 0
                for d in details:
                    pred = 1 if d["probability"] >= threshold else 0
                    d["prediction"] = pred
                    if d["label"] == 1 and pred == 1: tp += 1
                    elif d["label"] == 1 and pred == 0: fn += 1
                    elif d["label"] == 0 and pred == 1: fp += 1
                    elif d["label"] == 0 and pred == 0: tn += 1
                rm.update(calculate_metrics(tp, fp, tn, fn))
                global_tp += tp
                global_fn += fn
                
        # global neg
        fp, tn = 0, 0
        for p in neg_y_prob:
            if p >= threshold: fp += 1
            else: tn += 1
        global_fp, global_tn = fp, tn
"""
content = content.replace('    global_metrics = calculate_metrics(global_tp, global_fp, global_tn, global_fn)', injection + '\n    global_metrics = calculate_metrics(global_tp, global_fp, global_tn, global_fn)')

with open("utils/test_automation.py", "w") as f:
    f.write(content)

print("Updated test_automation.py")
