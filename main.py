# import os and gc at top
import os
import gc
import time
import argparse
import random

def apply_intel_optimizations():
    # Optimize threads for Intel processors (P-Cores)
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    
    # Enable Intel scikit-learn optimizations (Must be before other sklearn imports)
    try:
        from sklearnex import patch_sklearn
        patch_sklearn()
        print("---> Intel scikit-learn optimizations enabled.")
    except ImportError:
        print("---> sklearnex not found, skipping Intel scikit-learn optimizations.")

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
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
    save_experiment,
    save_cv_results
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
    parser.add_argument("--device", type=str, default="auto",
                        choices=["cpu", "cuda", "xpu", "auto"],
                        help="Device to use for training (default: auto)")
    parser.add_argument("--cv", action="store_true",
                        help="Run Stratified K-Fold cross-validation instead of single train/test split")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--cv-pairs", type=int, default=None,
                        help="Pairs per fold for CV (default: uses --pairs value)")
    return parser.parse_args()


def run_cross_validation(args, all_codes, labels, processed_codes,
                         code_features_all, cf_patterns_all,
                         semantic_features_all,
                         model_name, build_fn):
    """
    Run Stratified K-Fold cross-validation.
    Splits at the CODE level to prevent data leakage.
    """
    cv_pairs = args.cv_pairs if args.cv_pairs is not None else args.pairs
    cv_folds = args.cv_folds

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=args.seed)
    fold_metrics = []

    print(f"\n{'='*60}")
    print(f"  Starting {cv_folds}-Fold Cross-Validation")
    print(f"  Pairs per fold: {cv_pairs}")
    print(f"{'='*60}")

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(processed_codes, labels)):
        print(f"\n--- Fold {fold_idx + 1}/{cv_folds} ---")
        print(f"  Train codes: {len(train_idx)}, Test codes: {len(test_idx)}")

        # Split data for this fold
        train_labels = [labels[i] for i in train_idx]
        test_labels = [labels[i] for i in test_idx]
        train_codes = [processed_codes[i] for i in train_idx]
        test_codes = [processed_codes[i] for i in test_idx]

        train_code_features = code_features_all[train_idx]
        test_code_features = code_features_all[test_idx]
        train_cf_patterns = [cf_patterns_all[i] for i in train_idx]
        test_cf_patterns = [cf_patterns_all[i] for i in test_idx]

        # Split semantic features
        train_semantic = {k: [v[i] for i in train_idx] for k, v in semantic_features_all.items()}
        test_semantic = {k: [v[i] for i in test_idx] for k, v in semantic_features_all.items()}

        # TF-IDF: fit on train, transform both
        print("  → Vectorizing with Token TF-IDF...")
        vectorizer = build_tfidf_vectorizer()
        X_train_token = vectorizer.fit_transform(train_codes)
        X_test_token = vectorizer.transform(test_codes)

        # Generate pairs
        test_ratio = len(test_idx) / (len(train_idx) + len(test_idx))
        num_train_pairs = int(cv_pairs * (1 - test_ratio))
        num_test_pairs = int(cv_pairs * test_ratio)

        print(f"  → Generating {num_train_pairs} train pairs...")
        X_train, y_train = generate_pairs(
            X_train_token, train_labels, num_train_pairs, train_codes,
            code_features=train_code_features,
            cf_patterns=train_cf_patterns,
            semantic_features=train_semantic,
            random_state=args.seed + fold_idx
        )
        X_train = X_train.astype(np.float32)

        del X_train_token, train_code_features, train_cf_patterns, train_codes, train_semantic
        gc.collect()

        print(f"  → Generating {num_test_pairs} test pairs...")
        X_test, y_test = generate_pairs(
            X_test_token, test_labels, num_test_pairs, test_codes,
            code_features=test_code_features,
            cf_patterns=test_cf_patterns,
            semantic_features=test_semantic,
            random_state=args.seed + fold_idx + 1000
        )
        X_test = X_test.astype(np.float32)

        del X_test_token, test_code_features, test_cf_patterns, test_codes, test_semantic
        gc.collect()

        # Build & train model
        if args.model in ["xgboost", "dl_model", "ensemble"]:
            model = build_fn(args.seed, device=args.device)
        else:
            model = build_fn(args.seed)

        print(f"  → Training {model_name}...")
        if args.model == "xgboost":
            # Apply feature_weights for Type-4 semantic feature prioritization
            from preprocessing.code_features import FEATURE_NAMES
            feature_weights = np.ones(X_train.shape[1], dtype=np.float32)
            num_cf = 1
            num_semantic = 6
            num_extra = 2 + len(FEATURE_NAMES) + num_cf + num_semantic
            if num_extra > 0:
                feature_weights[-num_extra:] = 1000.0
            model.set_params(feature_weights=feature_weights)
            # NOTE: No eval_set in CV to prevent data leakage
            # (test fold must not influence training)
            model.fit(X_train, y_train, verbose=False)
        else:
            model.fit(X_train, y_train)

        # Evaluate
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        y_train_pred = model.predict(X_train)
        y_pred = model.predict(X_test)

        fold_result = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_f1_score": f1_score(y_train, y_train_pred),
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }

        if hasattr(model, "predict_proba"):
            y_train_prob = model.predict_proba(X_train)[:, 1]
            y_prob = model.predict_proba(X_test)[:, 1]
            fold_result["train_auc_roc"] = roc_auc_score(y_train, y_train_prob)
            fold_result["auc_roc"] = roc_auc_score(y_test, y_prob)

        fold_metrics.append(fold_result)
        print(f"  → Fold {fold_idx + 1}: "
              f"Train Acc={fold_result['train_accuracy']:.4f}, "
              f"Test Acc={fold_result['accuracy']:.4f}, "
              f"Test F1={fold_result['f1_score']:.4f}", end="")
        if "auc_roc" in fold_result:
            print(f", Test AUC={fold_result['auc_roc']:.4f}")
        else:
            print()

        # Free fold data
        del X_train, y_train, X_test, y_test, model
        gc.collect()

    # Save CV results
    save_cv_results(
        model_name=model_name,
        fold_metrics=fold_metrics,
        pair_count=cv_pairs,
        cv_folds=cv_folds
    )


def main():
    args = parse_args()

    # <----------> CONFIG <---------->
    DATASET_PATH = args.dataset
    NUM_PAIRS = args.pairs
    TEST_SIZE = args.test_size
    RANDOM_STATE = args.seed

    # <----------> DEVICE SELECTION <---------->
    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            args.device = "xpu"
        else:
            args.device = "cpu"
    
    print(f"---> Using device: {args.device}")

    # Only apply Intel optimizations for CPU or XPU
    if args.device in ["cpu", "xpu"]:
        apply_intel_optimizations()

    random.seed(RANDOM_STATE)

    # <----------> LOAD DATA <---------->
    print("---> Loading dataset...")
    t_start_total = time.time()
    t_phase = time.time()

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
    t_load = time.time() - t_phase

    # <----------> PREPROCESS <---------->
    print("---> Preprocessing codes...")
    t_phase = time.time()

    processed_codes = []
    for code in tqdm(all_codes, desc="Tokenizing"):
        tokens = tokenize(code)
        norm_tokens = normalize_tokens(tokens)
        processed_codes.append(" ".join(norm_tokens))

    t_preprocess = time.time() - t_phase

    # <----------> CODE FEATURES (AST + Control Flow + Semantic) <---------->
    print("---> Extracting structural and semantic features...")
    t_phase = time.time()
    code_features_all, cf_patterns_all, semantic_features_all = extract_all_features(all_codes)
    t_features = time.time() - t_phase

    # Keep raw codes split for new features (edit distance, line/char ratios)
    # They will be freed after pair generation

    # <---------> CROSS-VALIDATION MODE <---------->
    if args.cv:
        MODEL_BUILDERS = {
            "random_forest": ("RandomForest", build_random_forest),
            "linear_svm": ("LinearSVM", build_linear_svm),
            "xgboost": ("XGBoost", build_xgboost),
            "ensemble": ("Ensemble", build_voting_ensemble),
            "dl_model": ("DeepLearning", build_dl_model),
        }
        model_name, build_fn = MODEL_BUILDERS[args.model]

        run_cross_validation(
            args, all_codes, labels, processed_codes,
            code_features_all, cf_patterns_all,
            semantic_features_all,
            model_name, build_fn
        )
        return

    # <---------> SPLIT CODES FIRST (prevents data leakage) <---------->
    print("---> Splitting codes into train/test...")
    t_phase = time.time()

    indices = list(range(len(processed_codes)))
    # First split: Train+Val (80%) and Test (20%)
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )
    
    # Second split: Train (roughly 80% of 80% = 64% total) and Val (roughly 20% of 80% = 16% total)
    train_val_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_val_labels
    )

    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    test_labels = [labels[i] for i in test_idx]
    
    train_codes = [processed_codes[i] for i in train_idx]
    val_codes = [processed_codes[i] for i in val_idx]
    test_codes = [processed_codes[i] for i in test_idx]

    # Free full lists — we have train/test copies now
    del processed_codes, all_codes
    gc.collect()

    # Split structural features
    train_code_features = code_features_all[train_idx]
    val_code_features = code_features_all[val_idx]
    test_code_features = code_features_all[test_idx]
    
    train_cf_patterns = [cf_patterns_all[i] for i in train_idx]
    val_cf_patterns = [cf_patterns_all[i] for i in val_idx]
    test_cf_patterns = [cf_patterns_all[i] for i in test_idx]

    # Split semantic features
    train_semantic = {k: [v[i] for i in train_idx] for k, v in semantic_features_all.items()}
    val_semantic = {k: [v[i] for i in val_idx] for k, v in semantic_features_all.items()}
    test_semantic = {k: [v[i] for i in test_idx] for k, v in semantic_features_all.items()}

    # Free full feature arrays
    del code_features_all, cf_patterns_all, semantic_features_all
    gc.collect()

    print(f"Train codes: {len(train_idx)}, Val codes: {len(val_idx)}, Test codes: {len(test_idx)}")

    print("---> Vectorizing with Token TF-IDF...")
    vectorizer = build_tfidf_vectorizer()
    X_train_token = vectorizer.fit_transform(train_codes)
    X_val_token = vectorizer.transform(val_codes)
    X_test_token = vectorizer.transform(test_codes)
    print(f"Token TF-IDF shape: {X_train_token.shape}")

    print(f"Total feature count: {X_train_token.shape[1]} (token)")
    t_split_tfidf = time.time() - t_phase

    # <----------> PAIRS (from separate splits) <---------->
    # Distribute NUM_PAIRS across the three splits
    num_train_pairs = int(NUM_PAIRS * 0.70)
    num_val_pairs = int(NUM_PAIRS * 0.15)
    num_test_pairs = NUM_PAIRS - num_train_pairs - num_val_pairs

    print(f"---> Generating {num_train_pairs} train pairs...")
    t_phase = time.time()
    X_train, y_train = generate_pairs(
        X_train_token, train_labels, num_train_pairs, train_codes,
        code_features=train_code_features,
        cf_patterns=train_cf_patterns,
        semantic_features=train_semantic,
        random_state=RANDOM_STATE
    )
    X_train = X_train.astype(np.float32)

    del X_train_token, train_code_features, train_cf_patterns, train_codes, train_semantic
    gc.collect()

    print(f"---> Generating {num_val_pairs} val pairs...")
    X_val, y_val = generate_pairs(
        X_val_token, val_labels, num_val_pairs, val_codes,
        code_features=val_code_features,
        cf_patterns=val_cf_patterns,
        semantic_features=val_semantic,
        random_state=RANDOM_STATE + 1
    )
    X_val = X_val.astype(np.float32)

    del X_val_token, val_code_features, val_cf_patterns, val_codes, val_semantic
    gc.collect()

    print(f"---> Generating {num_test_pairs} test pairs...")
    X_test, y_test = generate_pairs(
        X_test_token, test_labels, num_test_pairs, test_codes,
        code_features=test_code_features,
        cf_patterns=test_cf_patterns,
        semantic_features=test_semantic,
        random_state=RANDOM_STATE + 2
    )
    X_test = X_test.astype(np.float32)

    del X_test_token, test_code_features, test_cf_patterns, test_codes, test_semantic
    gc.collect()

    print(f"Train pair matrix: {X_train.shape}")
    print(f"Val pair matrix:   {X_val.shape}")
    print(f"Test pair matrix:  {X_test.shape}")
    t_pairs = time.time() - t_phase

    # <----------> HYPERPARAMETER TUNING (optional) <---------->
    MODEL_BUILDERS = {
        "random_forest": ("RandomForest", build_random_forest),
        "linear_svm": ("LinearSVM", build_linear_svm),
        "xgboost": ("XGBoost", build_xgboost),
        "ensemble": ("Ensemble", build_voting_ensemble),
        "dl_model": ("DeepLearning", build_dl_model),
    }

    model_name, build_fn = MODEL_BUILDERS[args.model]

    # Pre-compute feature weights for Type-4 semantic feature prioritization
    feature_weights = np.ones(X_train.shape[1], dtype=np.float32)
    from preprocessing.code_features import FEATURE_NAMES
    num_cf = 1
    num_semantic = 6  # 6 new semantic pair-level features (A1+A2+B3)
    num_extra = 2 + len(FEATURE_NAMES) + num_cf + num_semantic  # cos, length, ast+ir+a1, cf, semantic
    if num_extra > 0:
        feature_weights[-num_extra:] = 1000.0

    if args.tune:
        print(f"\n---> Tuning {model_name} with Optuna ({args.tune_trials} trials)...")
        t_phase = time.time()
        from utils.hyperparameter_tuner import tune_hyperparameters
        best_params, best_score = tune_hyperparameters(
            args.model, X_train, y_train,
            random_state=RANDOM_STATE,
            n_trials=args.tune_trials,
            device=args.device,
            feature_weights=feature_weights if args.model == "xgboost" else None
        )

        # Build model with best params
        if args.model == "xgboost":
            from xgboost import XGBClassifier
            # Configure XGBoost device
            xgb_device = args.device if args.device != "xpu" else "cpu" # XGBoost doesn't natively support XPU yet in standard pip
            model = XGBClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1, device=xgb_device)
        elif args.model == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
        elif args.model == "linear_svm":
            from sklearn.svm import LinearSVC
            from sklearn.calibration import CalibratedClassifierCV
            model = CalibratedClassifierCV(
                LinearSVC(**best_params, random_state=RANDOM_STATE), cv=3
            )
        t_tune = time.time() - t_phase
    else:
        t_tune = 0.0
        # Pass device to builders that support it
        if args.model in ["xgboost", "dl_model", "ensemble"]:
            model = build_fn(RANDOM_STATE, device=args.device)
        else:
            model = build_fn(RANDOM_STATE)

    # <----------> TRAIN <---------->
    print(f"---> Training {model_name}...")
    t_phase = time.time()
    if args.model == "xgboost":
        # Apply pre-computed Type-4 semantic feature weights
        model.set_params(feature_weights=feature_weights)
        
        # USE VALIDATION SET FOR EARLY STOPPING
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    else:
        model.fit(X_train, y_train)
    t_train = time.time() - t_phase

    # <----------> EVALUATION <---------->
    t_phase = time.time()
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    t_eval = time.time() - t_phase
    t_total = time.time() - t_start_total

    # <----------> TIMING SUMMARY <---------->
    timing_info = {
        'Data Loading': t_load,
        'Preprocessing (Tokenization)': t_preprocess,
        'Feature Extraction (AST+IR+A1+A2+B3)': t_features,
        'Split + TF-IDF Vectorization': t_split_tfidf,
        'Pair Generation (train+val+test)': t_pairs,
        'Hyperparameter Tuning': t_tune,
        'Model Training': t_train,
        'Evaluation (Prediction)': t_eval,
        'TOTAL': t_total,
    }

    print(f"\n{'='*55}")
    print(f"  TIMING SUMMARY")
    print(f"{'='*55}")
    for phase, secs in timing_info.items():
        mins = secs / 60
        if phase == 'TOTAL':
            print(f"  {'─'*51}")
        print(f"  {phase:<40} {mins:>6.2f} min ({secs:.1f}s)")
    print(f"{'='*55}")

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
        timing_info=timing_info
    )


if __name__ == "__main__":
    main()