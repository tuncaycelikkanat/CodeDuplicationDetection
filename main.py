import os
import gc
import argparse
import random

# Optimize threads for Intel Core Ultra 5 125H (P-Cores)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

# Enable Intel scikit-learn optimizations (Must be before other sklearn imports)
from sklearnex import patch_sklearn
patch_sklearn()

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models.linear_svm import build_linear_svm
from models.xgboost import build_xgboost
from models.ensemble import build_voting_ensemble
from preprocessing.tokenizer import tokenize, normalize_tokens
from preprocessing.code_features import extract_all_features
from vectorization.tfidf import build_tfidf_vectorizer, build_char_tfidf_vectorizer
from pairing.pair_generator import generate_pairs
from models.random_forest import build_random_forest
from models.dl_model import build_dl_model

from utils.experiment_logger import (
    generate_experiment_name,
    save_experiment
)


# <----------> ARGUMENT PARSING <---------->
def parse_args():
    parser = argparse.ArgumentParser(description="Code Duplication Detection - Training Pipeline")
    parser.add_argument("--model", type=str, default="xgboost",
                        choices=["random_forest", "linear_svm", "xgboost", "ensemble", "dl_model"],
                        help="Model to train (default: xgboost)")
    parser.add_argument("--dataset", type=str, default="data/poj104",
                        help="Path to dataset directory (default: data/poj104)")
    parser.add_argument("--pairs", type=int, default=800_000,
                        help="Number of pairs to generate (default: 800_000)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test split ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter tuning before training")
    parser.add_argument("--tune-trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    return parser.parse_args()


def main():
    args = parse_args()

    # <----------> CONFIG <---------->
    DATASET_PATH = args.dataset
    NUM_PAIRS = args.pairs
    TEST_SIZE = args.test_size
    RANDOM_STATE = args.seed

    random.seed(RANDOM_STATE)

    # <----------> LOAD DATA <---------->
    print("---> Loading dataset...")

    all_codes = []
    labels = []

    for label in sorted(os.listdir(DATASET_PATH)):
        class_path = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            if file.endswith(".txt"):
                with open(os.path.join(class_path, file), "r", errors="ignore") as f:
                    all_codes.append(f.read())
                    labels.append(label)

    print(f"---> Total codes: {len(all_codes)}")

    # <----------> PREPROCESS <---------->
    print("---> Preprocessing codes...")

    processed_codes = []
    for code in tqdm(all_codes, desc="Tokenizing"):
        tokens = tokenize(code)
        norm_tokens = normalize_tokens(tokens)
        processed_codes.append(" ".join(norm_tokens))

    # <----------> CODE FEATURES (AST + Control Flow) <---------->
    print("---> Extracting structural features...")
    code_features_all, cf_patterns_all = extract_all_features(all_codes)

    # Keep raw codes split for new features (edit distance, line/char ratios)
    # They will be freed after pair generation

    # <----------> SPLIT CODES FIRST (prevents data leakage) <---------->
    print("---> Splitting codes into train/test...")

    indices = list(range(len(processed_codes)))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]
    train_codes = [processed_codes[i] for i in train_idx]
    test_codes = [processed_codes[i] for i in test_idx]
    train_raw_codes = [all_codes[i] for i in train_idx]
    test_raw_codes = [all_codes[i] for i in test_idx]

    # Free full lists — we have train/test copies now
    del processed_codes, all_codes
    gc.collect()

    # Split structural features
    train_code_features = code_features_all[train_idx]
    test_code_features = code_features_all[test_idx]
    train_cf_patterns = [cf_patterns_all[i] for i in train_idx]
    test_cf_patterns = [cf_patterns_all[i] for i in test_idx]

    # Free full feature arrays
    del code_features_all, cf_patterns_all
    gc.collect()

    print(f"Train codes: {len(train_idx)}, Test codes: {len(test_idx)}")

    # <----------> TF-IDF (Token + Char) <---------->
    print("---> Vectorizing with Token TF-IDF...")
    vectorizer = build_tfidf_vectorizer()
    X_train_token = vectorizer.fit_transform(train_codes)
    X_test_token = vectorizer.transform(test_codes)
    print(f"Token TF-IDF shape: {X_train_token.shape}")

    print("---> Vectorizing with Char TF-IDF...")
    char_vectorizer = build_char_tfidf_vectorizer()
    X_train_char = char_vectorizer.fit_transform(train_codes)
    X_test_char = char_vectorizer.transform(test_codes)
    print(f"Char TF-IDF shape: {X_train_char.shape}")

    print(f"Total feature count: {X_train_token.shape[1]} (token) + {X_train_char.shape[1]} (char)")

    # <----------> PAIRS (from separate splits) <---------->
    num_train_pairs = int(NUM_PAIRS * (1 - TEST_SIZE))
    num_test_pairs = int(NUM_PAIRS * TEST_SIZE)

    print(f"---> Generating {num_train_pairs} train pairs...")
    X_train, y_train = generate_pairs(
        X_train_token, train_labels, num_train_pairs, train_codes,
        X_char=X_train_char,
        code_features=train_code_features,
        cf_patterns=train_cf_patterns,
        raw_codes=train_raw_codes,
        random_state=RANDOM_STATE
    )

    # Free train intermediate data
    del X_train_token, X_train_char, train_code_features, train_cf_patterns, train_codes, train_raw_codes
    gc.collect()
    print("---> Freed train intermediate data.")

    print(f"---> Generating {num_test_pairs} test pairs...")
    X_test, y_test = generate_pairs(
        X_test_token, test_labels, num_test_pairs, test_codes,
        X_char=X_test_char,
        code_features=test_code_features,
        cf_patterns=test_cf_patterns,
        raw_codes=test_raw_codes,
        random_state=RANDOM_STATE + 1
    )

    # Free test intermediate data
    del X_test_token, X_test_char, test_code_features, test_cf_patterns, test_codes, test_raw_codes
    gc.collect()
    print("---> Freed test intermediate data.")

    print(f"Train pair matrix: {X_train.shape}")
    print(f"Test pair matrix: {X_test.shape}")

    # <----------> HYPERPARAMETER TUNING (optional) <---------->
    MODEL_BUILDERS = {
        "random_forest": ("RandomForest", build_random_forest),
        "linear_svm": ("LinearSVM", build_linear_svm),
        "xgboost": ("XGBoost", build_xgboost),
        "ensemble": ("Ensemble", build_voting_ensemble),
        "dl_model": ("DeepLearning", build_dl_model),
    }

    model_name, build_fn = MODEL_BUILDERS[args.model]

    if args.tune:
        print(f"\n---> Tuning {model_name} with Optuna ({args.tune_trials} trials)...")
        from utils.hyperparameter_tuner import tune_hyperparameters
        best_params, best_score = tune_hyperparameters(
            args.model, X_train, y_train,
            random_state=RANDOM_STATE,
            n_trials=args.tune_trials
        )

        # Build model with best params
        if args.model == "xgboost":
            from xgboost import XGBClassifier
            model = XGBClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
        elif args.model == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
        elif args.model == "linear_svm":
            from sklearn.svm import LinearSVC
            from sklearn.calibration import CalibratedClassifierCV
            model = CalibratedClassifierCV(
                LinearSVC(**best_params, random_state=RANDOM_STATE), cv=3
            )
    else:
        model = build_fn(RANDOM_STATE)

    # <----------> TRAIN <---------->
    print(f"---> Training {model_name}...")
    model.fit(X_train, y_train)

    # <----------> EVALUATION <---------->
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # <----------> SAVE EXPERIMENT <---------->
    exp_name = generate_experiment_name(
        model_name=model_name,
        pair_count=NUM_PAIRS
    )

    save_experiment(
        exp_name=exp_name,
        model_name=model_name,
        model=model,
        vectorizer=vectorizer,
        pair_count=NUM_PAIRS,
        X_train=X_train,
        y_train=y_train,
        y_train_pred=y_train_pred,
        X_test=X_test,
        y_test=y_test,
        y_test_pred=y_test_pred,
        extra_vectorizers={"char_tfidf": char_vectorizer}
    )


if __name__ == "__main__":
    main()

# başka ml modeller, bir tane dl model, ensemble, tune, future engineering