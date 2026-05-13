import numpy as np
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def _make_col_pipeline(clf, transformers):
    selector = ColumnTransformer(transformers=transformers, remainder='drop')
    return Pipeline([
        ('selector', selector),
        ('clf', clf)
    ])

def build_ensemble(random_state=42, device="cpu"):
    """
    Builds a Feature-Partitioned Stacking Ensemble:
    1. LightGBM (HistGBM) -> Lexical (0-3) + SVD (41+)
    2. Random Forest -> AST + CF (4-32)
    3. LinearSVC (Calibrated) -> Semantic (33-40)
    
    A Logistic Regression meta-classifier combines their outputs.
    """
    # 1. LightGBM (HistGradientBoosting): Yüzeysel ve Vektörel uzmanı
    hgb = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=10,
        min_samples_leaf=5,
        random_state=random_state
    )
    # Lexical (0-3) ve SVD (41'den sona kadar)
    pipe_hgb = _make_col_pipeline(hgb, [
        ('lex', 'passthrough', [0, 1, 2, 3]),
        ('svd', 'passthrough', slice(41, None))
    ])

    # 2. Random Forest: Yapısal (AST/CF) uzmanı
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=random_state,
        n_jobs=-1
    )
    # AST ve CF özellikleri (4'ten 32'ye kadar)
    pipe_rf = _make_col_pipeline(rf, [
        ('ast_cf', 'passthrough', slice(4, 33))
    ])

    # 3. LinearSVC (Calibrated): Semantik uzmanı
    # SVM, 500k veri için RBF kernel ile çok yavaş olur. LinearSVC çok hızlıdır.
    # CalibratedClassifierCV, SVM'in olasılık (predict_proba) üretmesini sağlar.
    svm = LinearSVC(dual=False, random_state=random_state, max_iter=2000)
    calibrated_svm = CalibratedClassifierCV(svm, cv=3, method='sigmoid')
    
    # Semantik özellikler (33'ten 40'a kadar)
    pipe_svm = _make_col_pipeline(calibrated_svm, [
        ('sem', 'passthrough', slice(33, 41))
    ])

    # 4. Meta-Classifier (Stacking)
    estimators = [
        ('lex_svd_hgb', pipe_hgb),
        ('struct_rf', pipe_rf),
        ('semantic_svm', pipe_svm)
    ]
    
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
        cv=5,
        n_jobs=1  # Important: keeping n_jobs=1 because base models already use n_jobs=-1
    )
    
    return clf
