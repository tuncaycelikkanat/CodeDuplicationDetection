import os
import json
import re
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def _make_json_serializable(d):
    safe = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            safe[k] = v
        except TypeError:
            safe[k] = str(v)
    return safe


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
    base_dir="experiments"
):
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # ================= CONFIG =================
    config = {
        "model_name": model_name,
        "pair_count": pair_count,
        "model_params": _make_json_serializable(model.get_params()),
        "tfidf_params": _make_json_serializable(vectorizer.get_params()),
        "num_features": int(X_train.shape[1])
    }

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # ================= METRICS =================
    train_metrics = {
        "accuracy": accuracy_score(y_train, y_train_pred)
    }

    test_metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred)
    }

    with open(os.path.join(exp_dir, "metrics_train.json"), "w") as f:
        json.dump(train_metrics, f, indent=4)

    with open(os.path.join(exp_dir, "metrics_test.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)

    # ================= REPORTS =================
    with open(os.path.join(exp_dir, "classification_report_train.txt"), "w") as f:
        f.write(classification_report(y_train, y_train_pred))

    with open(os.path.join(exp_dir, "classification_report_test.txt"), "w") as f:
        f.write(classification_report(y_test, y_test_pred))

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

    # ================= NOTES =================
    with open(os.path.join(exp_dir, "notes.txt"), "w") as f:
        f.write("Notes:\n")

    # ================= SERIALIZATION =================
    with open(os.path.join(exp_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(exp_dir, "tfidf.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"✅ Experiment saved: {exp_dir}")
