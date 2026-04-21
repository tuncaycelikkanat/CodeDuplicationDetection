"""
Hyperparameter tuning with Optuna for XGBoost, Random Forest, and Linear SVM.
Memory-optimized: avoids double parallelism and uses gc between trials.
"""

import gc
import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _xgboost_objective(trial, X, y, random_state, device="cpu", feature_weights=None):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.2),
        'eval_metric': 'logloss',
        'random_state': random_state,
        'n_jobs': -1,
        'device': device if device == "cuda" else "cpu",
        'max_bin': 64,
        'tree_method': 'hist'
    }

    # Apply Type-4 semantic feature weights if provided
    if feature_weights is not None:
        params['feature_weights'] = feature_weights

    model = XGBClassifier(**params)
    # n_jobs=1 for CV to avoid double parallelism (XGBoost already uses n_jobs=-1)
    scores = cross_val_score(model, X, y, cv=3, scoring='f1', n_jobs=1)
    gc.collect()
    return scores.mean()


def _random_forest_objective(trial, X, y, random_state):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 10, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'max_samples': trial.suggest_float('max_samples', 0.3, 0.8),
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': random_state
    }

    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X, y, cv=3, scoring='f1', n_jobs=1)
    gc.collect()
    return scores.mean()


def _linear_svm_objective(trial, X, y, random_state):
    params = {
        'C': trial.suggest_float('C', 0.1, 50.0, log=True),
        'loss': 'squared_hinge',
        'dual': False,
        'max_iter': 5000,
        'tol': 1e-3,
        'class_weight': 'balanced',
        'random_state': random_state
    }

    base = LinearSVC(**params)
    model = CalibratedClassifierCV(base, cv=3)
    scores = cross_val_score(model, X, y, cv=3, scoring='f1', n_jobs=1)
    gc.collect()
    return scores.mean()


# [COMPAT] _OBJECTIVES sözlüğü geriye dönük uyumluluk için korunmaktadır.
# Mevcut argparse yalnızca 'xgboost' seçeneğine izin verdiğinden, rf ve svm
# objective'leri şu an aktif olarak kullanılmamaktadır.
_OBJECTIVES = {
    'xgboost': _xgboost_objective,
    'random_forest': _random_forest_objective,  # [COMPAT] main.py'de choices'a eklenmemiş
    'linear_svm': _linear_svm_objective,        # [COMPAT] main.py'de choices'a eklenmemiş
}


def tune_hyperparameters(model_name, X, y, random_state=42, n_trials=30, device="cpu", feature_weights=None):
    """
    Run Optuna hyperparameter tuning.

    Args:
        model_name: one of 'xgboost', 'random_forest', 'linear_svm'
        X: training feature matrix
        y: training labels
        random_state: random_state
        n_trials: number of Optuna trials
        device: device to use (for XGBoost)
        feature_weights: optional array of feature weights for XGBoost Type-4 tuning

    Returns:
        best_params: dict of the best hyperparameters
        best_score: best F1 score
    """
    objective_fn = _OBJECTIVES[model_name]

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )

    if model_name == "xgboost":
        study.optimize(
            lambda trial: _xgboost_objective(trial, X, y, random_state, device=device, feature_weights=feature_weights),
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True
        )
    else:
        study.optimize(
            lambda trial: objective_fn(trial, X, y, random_state),
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True
        )

    print(f"\n🏆 Best F1: {study.best_value:.4f}")
    print(f"   Best params: {study.best_params}")

    return study.best_params, study.best_value
