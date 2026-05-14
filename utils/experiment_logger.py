import os
import json
import re
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    average_precision_score,
    matthews_corrcoef
)


def _make_json_serializable(obj):
    """Recursively convert an object to a JSON-serializable form."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def generate_experiment_name(model_name, pair_count, base_dir="experiments"):
    """
    exp_001_RandomForest_400k
    exp_002_LinearSVM_400k
    """
    os.makedirs(base_dir, exist_ok=True)

    exp_nums = []
    for name in os.listdir(base_dir):
        m = re.match(r"exp_(\d+)_", name)
        if m:
            exp_nums.append(int(m.group(1)))

    next_id = max(exp_nums) + 1 if exp_nums else 1
    return f"exp_{next_id:03d}_{model_name}_{pair_count//1000}k"


def _compute_metrics(y_true, y_pred, y_prob=None):
    """Compute accuracy, F1, MCC, and optionally AUC-ROC / PR-AUC."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }
    if y_prob is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    return metrics


def save_experiment(
    exp_name,
    model_name,
    model,
    vectorizer,
    pair_count,
    X_train,
    y_train,
    y_train_pred,
    X_test,
    y_test,
    y_test_pred,
    base_dir="experiments",
    extra_vectorizers=None,
    timing_info=None,
    stage1_model=None,
    X_val=None,
    y_val=None,
    y_val_pred=None,
    use_ssl=False,
    ssl_pca=None,
):
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # ================= CONFIG =================
    config = {
        "model_name": model_name,
        "pair_count": pair_count,
        "model_params": _make_json_serializable(model.get_params()),
        "tfidf_params": _make_json_serializable(vectorizer.get_params()),
        "num_features": int(X_train.shape[1]),
        "use_ssl": use_ssl
    }

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # ================= PROBABILITIES =================
    y_train_prob = None
    y_test_prob  = None
    y_val_prob   = None
    if hasattr(model, "predict_proba"):
        y_test_prob = model.predict_proba(X_test)[:, 1]
        if y_val is not None:
            y_val_prob = model.predict_proba(X_val)[:, 1]

    # ================= METRICS =================
    train_metrics = _compute_metrics(y_train, y_train_pred)
    test_metrics  = _compute_metrics(y_test, y_test_pred, y_test_prob)

    with open(os.path.join(exp_dir, "metrics_train.json"), "w") as f:
        json.dump(train_metrics, f, indent=4)

    with open(os.path.join(exp_dir, "metrics_test.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)

    # Val metrikleri opsiyonel olarak kaydedilir (#19)
    if y_val is not None and y_val_pred is not None:
        val_metrics = _compute_metrics(y_val, y_val_pred, y_val_prob)
        with open(os.path.join(exp_dir, "metrics_val.json"), "w") as f:
            json.dump(val_metrics, f, indent=4)

    # ================= REPORTS =================
    for split, y_true, y_pred in [
        ("train", y_train, y_train_pred),
        ("test", y_test, y_test_pred)
    ]:
        # Text report
        report_txt = classification_report(y_true, y_pred)
        with open(os.path.join(exp_dir, f"classification_report_{split}.txt"), "w") as f:
            f.write(report_txt)

        # JSON report
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        with open(os.path.join(exp_dir, f"classification_report_{split}.json"), "w") as f:
            json.dump(report_dict, f, indent=4)

    # ================= CONFUSION MATRICES =================
    for split, y_true, y_pred in [
        ("train", y_train, y_train_pred),
        ("test", y_test, y_test_pred)
    ]:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap="Blues")
        plt.title(f"{model_name} - {split.upper()}")
        plt.savefig(os.path.join(exp_dir, f"confusion_matrix_{split}.png"))
        plt.close()

    # ================= NOTES (with timing) =================
    with open(os.path.join(exp_dir, "notes.txt"), "w") as f:
        f.write("Notes:\n\n")
        if timing_info:
            from datetime import datetime
            f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*55}\n")
            f.write(f"  TIMING SUMMARY\n")
            f.write(f"{'='*55}\n")
            for phase, secs in timing_info.items():
                mins = secs / 60
                if phase == 'TOTAL':
                    f.write(f"  {'─'*51}\n")
                f.write(f"  {phase:<40} {mins:>6.2f} min ({secs:.1f}s)\n")
            f.write(f"{'='*55}\n")

    # ================= SERIALIZATION =================
    with open(os.path.join(exp_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    if stage1_model is not None:
        with open(os.path.join(exp_dir, "stage1_model.pkl"), "wb") as f:
            pickle.dump(stage1_model, f)

    with open(os.path.join(exp_dir, "tfidf.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    if extra_vectorizers:
        for name, vec in extra_vectorizers.items():
            with open(os.path.join(exp_dir, f"{name}.pkl"), "wb") as f:
                pickle.dump(vec, f)

    if ssl_pca is not None:
        with open(os.path.join(exp_dir, "ssl_pca.pkl"), "wb") as f:
            pickle.dump(ssl_pca, f)
        print(f"   ssl_pca: saved ({ssl_pca.n_components} components, "
              f"explained var: {ssl_pca.explained_variance_ratio_.sum():.2%})")

    # ================= SUMMARY =================
    print(f"\n✅ Experiment saved: {exp_dir}")
    print(f"   Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"   F1 Score : {test_metrics['f1_score']:.4f}")
    if "auc_roc" in test_metrics:
        print(f"   AUC-ROC  : {test_metrics['auc_roc']:.4f}")


def save_cv_results(
    model_name,
    fold_metrics,
    pair_count,
    cv_folds,
    base_dir="experiments"
):
    """
    Save cross-validation results to experiments directory.

    Args:
        model_name: Name of the model
        fold_metrics: List of dicts, each with 'accuracy', 'f1_score', 'auc_roc'
        pair_count: Total number of pairs used
        cv_folds: Number of CV folds
        base_dir: Base directory for experiments
    """
    import numpy as np

    exp_name = generate_experiment_name(
        model_name=f"{model_name}_CV{cv_folds}",
        pair_count=pair_count
    )
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # ================= PER-FOLD METRICS =================
    with open(os.path.join(exp_dir, "cv_fold_metrics.json"), "w") as f:
        json.dump(fold_metrics, f, indent=4)

    # ================= SUMMARY (mean ± std) =================
    metric_names = list(fold_metrics[0].keys())
    summary = {}
    for m in metric_names:
        values = [fold[m] for fold in fold_metrics if fold.get(m) is not None]
        if values:
            summary[m] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "per_fold": values
            }

    with open(os.path.join(exp_dir, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # ================= CONFIG =================
    config = {
        "model_name": model_name,
        "pair_count": pair_count,
        "cv_folds": cv_folds,
        "mode": "cross_validation"
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # ================= CONSOLE OUTPUT =================
    print(f"\n{'='*60}")
    print(f"  Cross-Validation Results ({cv_folds}-Fold)")
    print(f"{'='*60}")
    print(f"  {'Fold':<8}", end="")
    for m in metric_names:
        print(f"{m:<14}", end="")
    print()
    print(f"  {'-'*8}", end="")
    for _ in metric_names:
        print(f"{'-'*14}", end="")
    print()

    for i, fold in enumerate(fold_metrics):
        print(f"  {'Fold '+str(i+1):<8}", end="")
        for m in metric_names:
            val = fold.get(m)
            print(f"{val:<14.4f}" if val is not None else f"{'N/A':<14}", end="")
        print()

    print(f"  {'-'*8}", end="")
    for _ in metric_names:
        print(f"{'-'*14}", end="")
    print()

    print(f"  {'Mean':<8}", end="")
    for m in metric_names:
        if m in summary:
            print(f"{summary[m]['mean']:<14.4f}", end="")
    print()

    print(f"  {'Std':<8}", end="")
    for m in metric_names:
        if m in summary:
            print(f"{summary[m]['std']:<14.4f}", end="")
    print()

    print(f"{'='*60}")
    print(f"\n✅ CV results saved: {exp_dir}")
