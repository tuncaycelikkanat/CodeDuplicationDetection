import numpy as np
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def build_ensemble(random_state=42, device="cpu"):
    """
    Builds a Stacking Ensemble model combining the strengths of:
    1. XGBoost
    2. Random Forest
    3. HistGradientBoosting (LightGBM equivalent)
    
    A Logistic Regression meta-classifier combines their outputs.
    """
    xgb_device = device if device != "xpu" else "cpu"
    
    # 1. XGBoost: The semantic powerhouse
    # Note: No early_stopping_rounds here because StackingClassifier 
    # cross-validation does not natively support passing eval_set.
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=random_state,
        n_jobs=-1,
        device=xgb_device,
        tree_method="hist"
    )

    # 2. Random Forest: The structural expert (less prone to overfitting)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=random_state,
        n_jobs=-1
    )

    # 3. HistGradientBoosting: The speed demon (LightGBM equivalent)
    hgb = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=10,
        min_samples_leaf=5,
        random_state=random_state
    )

    # 4. The Meta-Classifier
    estimators = [
        ('xgb', xgb),
        ('rf', rf),
        ('hgb', hgb)
    ]
    
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
        cv=5,
        n_jobs=1  # Important: keeping n_jobs=1 because base models already use n_jobs=-1
    )
    
    return clf
