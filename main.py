# import os and gc at top
import os
import gc
import time
import argparse
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CASCADE_THRESHOLD, CASCADE_STAGE1_THRESHOLD, STAGE1_FEATURE_COUNT,
    SVD_N_COMPONENTS, DEFAULT_PAIRS, DEFAULT_SEED, DEFAULT_TEST_SIZE,
    DEFAULT_CV_FOLDS, OMP_NUM_THREADS, MKL_NUM_THREADS, ENSEMBLE_SVD_START_IDX,
    SSL_PCA_COMPONENTS,
)

def apply_intel_optimizations():
    # Optimize threads for Intel processors (P-Cores)
    os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(MKL_NUM_THREADS)
    
    # Enable Intel scikit-learn optimizations (Must be before other sklearn imports)
    try:
        from sklearnex import patch_sklearn
        patch_sklearn()
        Log.step("Intel scikit-learn optimizations enabled.")
    except ImportError:
        Log.step("sklearnex not found, skipping Intel scikit-learn optimizations.")

import numpy as np
from utils.logger import Log

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

from models.xgboost import build_xgboost
from preprocessing.tokenizer import tokenize, normalize_tokens
from preprocessing.code_features import extract_all_features
from vectorization.tfidf import build_tfidf_vectorizer
from pairing.pair_generator import generate_pairs

from utils.experiment_logger import (
    generate_experiment_name,
    save_experiment,
    save_cv_results
)


# <----------> CASCADE FILTER HELPER <---------->
def _apply_cascade_filter(
    X: np.ndarray,
    y: np.ndarray,
    stage1_model,
    threshold: float = CASCADE_STAGE1_THRESHOLD,
    feature_count: int = STAGE1_FEATURE_COUNT,
) -> tuple:
    X_stage1 = X[:, :feature_count]
    y_prob_stage1 = stage1_model.predict_proba(X_stage1)[:, 1]
    
    # ROOT CAUSE FIX: We CANNOT filter out Easy Negatives!
    # If we filter them out in training, Stage-2 never sees a negative and predicts 1 for everything.
    # If we filter them out in inference, we accidentally throw away all Type-4 (Hard Positives) 
    # because they also have low lexical similarity!
    easy_pos_mask = (y == 1) & (y_prob_stage1 >= threshold)
    
    keep_mask = ~easy_pos_mask
    
    pos_before = (y == 1).sum()
    neg_before = (y == 0).sum()
    y_filtered = y[keep_mask]
    pos_after = (y_filtered == 1).sum()
    neg_after = (y_filtered == 0).sum()
    
    Log.substep(f"Cascade Filter Before: {pos_before} Pos, {neg_before} Neg")
    Log.substep(f"Cascade Filter After:  {pos_after} Pos, {neg_after} Neg")
    Log.substep(f"Filtered Easy Pos: {easy_pos_mask.sum()}, Easy Neg: 0")
    
    return X[keep_mask], y_filtered, easy_pos_mask.sum()


# <----------> ARGUMENT PARSING <---------->
def parse_args():
    parser = argparse.ArgumentParser(description="Code Duplication Detection - Cascade Training Pipeline")
    parser.add_argument("--model", type=str, default="xgboost",
                        choices=["xgboost", "ensemble"],
                        help="Model to train (default: xgboost)")
    parser.add_argument("--dataset", type=str, default="data/poj104",
                        help="Path to dataset directory (default: data/poj104)")
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIRS,
                        help=f"Number of pairs to generate (default: {DEFAULT_PAIRS})")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                        help=f"Test split ratio (default: {DEFAULT_TEST_SIZE})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter tuning before training")
    parser.add_argument("--tune-trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["cpu", "cuda", "xpu", "auto"],
                        help="Device to use for training (default: auto)")
    parser.add_argument("--cv", action="store_true",
                        help="Run Stratified K-Fold cross-validation instead of single train/test split")
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS,
                        help=f"Number of CV folds (default: {DEFAULT_CV_FOLDS})")
    parser.add_argument("--cv-pairs", type=int, default=None,
                        help="Pairs per fold for CV (default: uses --pairs value)")
    parser.add_argument("--positive-ratio", type=float, default=0.05,
                        help="Fraction of training pairs that are clones (default: 0.05). "
                             "Used for realistic class imbalance simulation.")
    parser.add_argument("--use-ssl", action="store_true",
                        help="Enable CodeBERT SSL embeddings extraction (Requires Transformers/Torch)")
    parser.add_argument("--ssl-cache", type=str, default=None,
                        help="Path to cache SSL embeddings (e.g. ssl_cache.npy). "
                             "If file exists, embeddings are loaded from cache instead of re-extracting.")
    return parser.parse_args()


def run_cross_validation(args, all_codes, labels, processed_codes,
                         code_features_all, cf_patterns_all,
                         semantic_features_all,
                         model_name, build_fn,
                         ssl_embeddings_all=None):
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

        train_ssl = ssl_embeddings_all[train_idx] if ssl_embeddings_all is not None else None
        test_ssl = ssl_embeddings_all[test_idx] if ssl_embeddings_all is not None else None

        # TF-IDF: fit on train, transform both
        Log.substep("Vectorizing with Token TF-IDF...")
        vectorizer = build_tfidf_vectorizer()
        X_train_token = vectorizer.fit_transform(train_codes)
        X_test_token = vectorizer.transform(test_codes)

        # TruncatedSVD
        Log.substep("Applying TruncatedSVD on Token TF-IDF...")
        svd = TruncatedSVD(n_components=SVD_N_COMPONENTS, random_state=args.seed)
        X_train_svd = svd.fit_transform(X_train_token)
        X_test_svd = svd.transform(X_test_token)
        print(f"    - Explained Variance Ratio (SVD): {svd.explained_variance_ratio_.sum():.4f}")

        # Generate pairs
        test_ratio = len(test_idx) / (len(train_idx) + len(test_idx))
        num_train_pairs = int(cv_pairs * (1 - test_ratio))
        num_test_pairs = int(cv_pairs * test_ratio)

        if train_ssl is not None:
            Log.substep("Fitting SSL PCA ({SSL_PCA_COMPONENTS} dims) for fold...")
            from sklearn.decomposition import PCA
            ssl_pca = PCA(n_components=SSL_PCA_COMPONENTS, random_state=args.seed)
            train_ssl = ssl_pca.fit_transform(train_ssl).astype(np.float32)
            test_ssl = ssl_pca.transform(test_ssl).astype(np.float32)

        Log.substep(f"Generating {num_train_pairs} train pairs...")
        X_train, y_train = generate_pairs(
            X_train_token, train_labels, num_train_pairs, train_codes,
            code_features=train_code_features,
            cf_patterns=train_cf_patterns,
            semantic_features=train_semantic,
            X_svd=X_train_svd,
            ssl_embeddings=train_ssl,
            random_state=args.seed + fold_idx,
            positive_ratio=args.positive_ratio
        )
        X_train = X_train.astype(np.float32)

        del X_train_token, train_code_features, train_cf_patterns, train_codes, train_semantic, X_train_svd, train_ssl
        gc.collect()

        Log.substep(f"Generating {num_test_pairs} test pairs...")
        X_test, y_test = generate_pairs(
            X_test_token, test_labels, num_test_pairs, test_codes,
            code_features=test_code_features,
            cf_patterns=test_cf_patterns,
            semantic_features=test_semantic,
            X_svd=X_test_svd,
            ssl_embeddings=test_ssl,
            random_state=args.seed + fold_idx + 1000
        )
        X_test = X_test.astype(np.float32)

        del X_test_token, test_code_features, test_cf_patterns, test_codes, test_semantic, X_test_svd, test_ssl
        gc.collect()

        # Build & train model


        # ---- Cascade filter (CV modu) ----
        # Eğitim setindeki "kolay" klonları (cos_token > CASCADE_THRESHOLD) çıkar.
        # Bu sayede model sadece zor / Type-4 klonlar üzerinde eğitilir.
        Log.substep("Training Stage-1 Lexical/Structural Model (CV)...")
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.calibration import CalibratedClassifierCV
        base_stage1 = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=args.seed)
        stage1_model = CalibratedClassifierCV(base_stage1, cv=3, method='isotonic')
        
        cos_tok_train = X_train[:, 0]
        easy_train_mask = (y_train == 0) | ((y_train == 1) & (cos_tok_train > CASCADE_THRESHOLD))
        
        X_train_stage1_easy = X_train[easy_train_mask, :STAGE1_FEATURE_COUNT]
        y_train_easy = y_train[easy_train_mask]
        stage1_model.fit(X_train_stage1_easy, y_train_easy)
        
        Log.substep("Filtering easy clones from CV Train set (threshold={CASCADE_STAGE1_THRESHOLD})...")
        X_train, y_train, n_removed = _apply_cascade_filter(X_train, y_train, stage1_model)
        Log.substep("Filtered CV Train: {X_train.shape} (removed {n_removed} easy clones)")

        if args.model == "ensemble":
            from models.ensemble import build_ensemble
            model = build_ensemble(args.seed, device=args.device)
        else:
            pos_count = max(1, (y_train == 1).sum())
            spw = (len(y_train) - pos_count) / pos_count
            model = build_xgboost(args.seed, device=args.device, scale_pos_weight=spw)

        Log.substep("Training {model_name}...")
        model.fit(X_train, y_train, verbose=False)

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
        try:
            import torch
            if torch.cuda.is_available():
                args.device = "cuda"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                args.device = "xpu"
            else:
                args.device = "cpu"
        except ImportError:
            # torch opsiyonel; kurulu değilse CPU kullan
            args.device = "cpu"

    Log.step(f"Using device: {args.device}")
    Log.step(f"Using device: {args.device}")

    # Only apply Intel optimizations for CPU or XPU
    if args.device in ["cpu", "xpu"]:
        apply_intel_optimizations()

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)  # numpy global seed — tam reproducibility için

    # <----------> LOAD DATA <---------->
    Log.step("Loading dataset...")
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

    Log.step(f"Total codes: {len(all_codes)}")
    t_load = time.time() - t_phase

    # <----------> PREPROCESS <---------->
    Log.step("Preprocessing codes...")
    t_phase = time.time()

    processed_codes = []
    for code in tqdm(all_codes, desc="Tokenizing"):
        tokens = tokenize(code)
        norm_tokens = normalize_tokens(tokens)
        processed_codes.append(" ".join(norm_tokens))

    t_preprocess = time.time() - t_phase

    # <----------> CODE FEATURES (AST + Control Flow + Semantic) <---------->
    Log.step("Extracting structural and semantic features...")
    t_phase = time.time()
    code_features_all, cf_patterns_all, semantic_features_all = extract_all_features(all_codes)
    t_features = time.time() - t_phase

    ssl_embeddings_all = None
    ssl_pca = None
    if args.use_ssl:
        from vectorization.ssl_encoder import extract_ssl_embeddings
        from sklearn.decomposition import PCA
        Log.step("Extracting SSL embeddings...")
        t_phase_ssl = time.time()
        raw_ssl = extract_ssl_embeddings(
            all_codes, device=args.device, cache_path=args.ssl_cache
        )
        Log.substep("SSL Embeddings shape: {raw_ssl.shape}  ({time.time() - t_phase_ssl:.1f}s)")

        Log.substep("Keeping SSL embeddings raw (768 dims). PCA will be fitted per split to prevent leakage.")
        ssl_embeddings_all = raw_ssl
        ssl_pca = None
        t_features += (time.time() - t_phase_ssl)

    # Keep raw codes split for new features (edit distance, line/char ratios)
    # They will be freed after pair generation

    # <---------> CROSS-VALIDATION MODE <---------->
    if args.cv:
        if args.model == "ensemble":
            from models.ensemble import build_ensemble
            model_name, build_fn = "Ensemble", build_ensemble
        else:
            model_name, build_fn = "XGBoost", build_xgboost

        run_cross_validation(
            args, all_codes, labels, processed_codes,
            code_features_all, cf_patterns_all,
            semantic_features_all,
            model_name, build_fn,
            ssl_embeddings_all=ssl_embeddings_all
        )
        return

    # <---------> SPLIT CODES FIRST (prevents data leakage) <---------->
    Log.step("Splitting codes into train/test...")
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

    train_ssl = ssl_embeddings_all[train_idx] if ssl_embeddings_all is not None else None
    val_ssl = ssl_embeddings_all[val_idx] if ssl_embeddings_all is not None else None
    test_ssl = ssl_embeddings_all[test_idx] if ssl_embeddings_all is not None else None

    # Free full feature arrays
    del code_features_all, cf_patterns_all, semantic_features_all, ssl_embeddings_all
    gc.collect()

    print(f"Train codes: {len(train_idx)}, Val codes: {len(val_idx)}, Test codes: {len(test_idx)}")

    Log.step("Vectorizing with Token TF-IDF...")
    vectorizer = build_tfidf_vectorizer()
    X_train_token = vectorizer.fit_transform(train_codes)
    X_val_token = vectorizer.transform(val_codes)
    X_test_token = vectorizer.transform(test_codes)
    print(f"Token TF-IDF shape: {X_train_token.shape}")
    
    Log.step("Applying TruncatedSVD on Token TF-IDF...")
    svd = TruncatedSVD(n_components=SVD_N_COMPONENTS, random_state=RANDOM_STATE)
    X_train_svd = svd.fit_transform(X_train_token)
    X_val_svd = svd.transform(X_val_token)
    X_test_svd = svd.transform(X_test_token)
    print(f"    - Explained Variance Ratio (SVD): {svd.explained_variance_ratio_.sum():.4f}")

    print(f"SVD components: {SVD_N_COMPONENTS}, Dense feature vector size: 89")
    t_split_tfidf = time.time() - t_phase

    # <----------> PAIRS (from separate splits) <---------->
    # Distribute NUM_PAIRS across the three splits
    num_train_pairs = int(NUM_PAIRS * 0.70)
    num_val_pairs = int(NUM_PAIRS * 0.15)
    num_test_pairs = NUM_PAIRS - num_train_pairs - num_val_pairs

    if train_ssl is not None:
        Log.step(f"Fitting SSL PCA: 768 → {SSL_PCA_COMPONENTS} dims...")
        from sklearn.decomposition import PCA
        ssl_pca = PCA(n_components=SSL_PCA_COMPONENTS, random_state=RANDOM_STATE)
        train_ssl = ssl_pca.fit_transform(train_ssl).astype(np.float32)
        val_ssl = ssl_pca.transform(val_ssl).astype(np.float32)
        test_ssl = ssl_pca.transform(test_ssl).astype(np.float32)

    Log.step(f"Generating {num_train_pairs} train pairs (positive_ratio={args.positive_ratio})...")
    t_phase = time.time()
    X_train, y_train = generate_pairs(
        X_train_token, train_labels, num_train_pairs, train_codes,
        code_features=train_code_features,
        cf_patterns=train_cf_patterns,
        semantic_features=train_semantic,
        X_svd=X_train_svd,
        ssl_embeddings=train_ssl,
        random_state=RANDOM_STATE,
        positive_ratio=args.positive_ratio
    )
    X_train = X_train.astype(np.float32)

    Log.substep("Training Stage-1 Lexical/Structural Model (HistGradientBoosting) on EASY pairs...")
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    base_stage1 = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=RANDOM_STATE)
    stage1_model = CalibratedClassifierCV(base_stage1, cv=3, method='isotonic')
    
    cos_tokens_train = X_train[:, 0]
    easy_train_mask = (y_train == 0) | ((y_train == 1) & (cos_tokens_train > CASCADE_THRESHOLD))
    
    X_train_stage1_easy = X_train[easy_train_mask, :STAGE1_FEATURE_COUNT]
    y_train_easy = y_train[easy_train_mask]
    stage1_model.fit(X_train_stage1_easy, y_train_easy)
    
    Log.substep("Filtering EASY clones from Train set (Two-Stage approach)...")
    X_train, y_train, n_removed_train = _apply_cascade_filter(X_train, y_train, stage1_model)
    Log.substep("Filtered Train matrix: {X_train.shape} (Removed {n_removed_train} easy clones via Stage-1)")

    del X_train_token, train_code_features, train_cf_patterns, train_codes, train_semantic, X_train_svd, train_ssl
    gc.collect()

    Log.step(f"Generating {num_val_pairs} val pairs...")
    X_val, y_val = generate_pairs(
        X_val_token, val_labels, num_val_pairs, val_codes,
        code_features=val_code_features,
        cf_patterns=val_cf_patterns,
        semantic_features=val_semantic,
        X_svd=X_val_svd,
        ssl_embeddings=val_ssl,
        random_state=RANDOM_STATE + 1,
        positive_ratio=args.positive_ratio
    )
    X_val = X_val.astype(np.float32)

    Log.substep("Filtering EASY clones from Val set (Two-Stage approach)...")
    X_val, y_val, n_removed_val = _apply_cascade_filter(X_val, y_val, stage1_model)
    Log.substep("Filtered Val matrix: {X_val.shape} (Removed {n_removed_val} easy clones via Stage-1)")

    del X_val_token, val_code_features, val_cf_patterns, val_codes, val_semantic, X_val_svd, val_ssl
    gc.collect()

    Log.step(f"Generating {num_test_pairs} test pairs...")
    X_test, y_test = generate_pairs(
        X_test_token, test_labels, num_test_pairs, test_codes,
        code_features=test_code_features,
        cf_patterns=test_cf_patterns,
        semantic_features=test_semantic,
        X_svd=X_test_svd,
        ssl_embeddings=test_ssl,
        random_state=RANDOM_STATE + 2
    )
    X_test = X_test.astype(np.float32)

    del X_test_token, test_code_features, test_cf_patterns, test_codes, test_semantic, X_test_svd, test_ssl
    gc.collect()

    print(f"Train pair matrix: {X_train.shape}")
    print(f"Val pair matrix:   {X_val.shape}")
    print(f"Test pair matrix:  {X_test.shape}")
    t_pairs = time.time() - t_phase

    # <----------> HYPERPARAMETER TUNING (optional) <---------->
    if args.model == "ensemble":
        from models.ensemble import build_ensemble
        model_name, build_fn = "Ensemble", build_ensemble
    else:
        from models.xgboost import build_xgboost
        model_name, build_fn = "XGBoost", build_xgboost

    if args.tune and args.model == "xgboost":
        print(f"\n---> Tuning {model_name} with Optuna ({args.tune_trials} trials)...")
        t_phase = time.time()
        from utils.hyperparameter_tuner import tune_hyperparameters
        best_params, best_score = tune_hyperparameters(
            args.model, X_train, y_train,
            random_state=RANDOM_STATE,
            n_trials=args.tune_trials,
            device=args.device
        )

        # Build model with best params
        from xgboost import XGBClassifier
        xgb_device = args.device if args.device != "xpu" else "cpu" 
        model = XGBClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1, device=xgb_device)
        t_tune = time.time() - t_phase
    else:
        t_tune = 0.0
        if args.model == "xgboost":
            pos_count = max(1, (y_train == 1).sum())
            spw = (len(y_train) - pos_count) / pos_count
            model = build_fn(RANDOM_STATE, device=args.device, scale_pos_weight=spw)
        else:
            model = build_fn(RANDOM_STATE, device=args.device)

    # <----------> TRAIN <---------->
    Log.step("Training {model_name}...")
    t_phase = time.time()
    
    if args.model == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    else:
        model.fit(X_train, y_train)
        
    t_train = time.time() - t_phase

    # <----------> EVALUATION (CASCADE INFERENCE) <---------->
    t_phase = time.time()
    
    # Train evaluation is on the filtered data (Hard examples only)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Test evaluation uses the Two-Stage logic
    Log.step("Running Two-Stage Inference on Test...")
    X_test_stage1 = X_test[:, :STAGE1_FEATURE_COUNT]
    y_test_prob_stage1 = stage1_model.predict_proba(X_test_stage1)[:, 1]
    y_test_pred = np.zeros_like(y_test)
    
    easy_pos_mask_test = y_test_prob_stage1 >= CASCADE_STAGE1_THRESHOLD
    
    y_test_pred[easy_pos_mask_test] = 1  # Pre-filter easy clones
    
    Log.substep("Stage-1 filtered {easy_pos_mask_test.sum()} Easy Pos immediately.")
    
    hard_mask_test = ~easy_pos_mask_test
    if hard_mask_test.sum() > 0:
        y_test_pred[hard_mask_test] = model.predict(X_test[hard_mask_test])
        
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
        model_name="CASCADE_" + model_name,
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
        X_val=X_val,
        y_val=y_val,
        y_val_pred=y_val_pred,
        timing_info=timing_info,
        extra_vectorizers={"svd": svd},
        stage1_model=stage1_model,
        use_ssl=args.use_ssl,
        ssl_pca=ssl_pca,
    )

if __name__ == "__main__":
    main()