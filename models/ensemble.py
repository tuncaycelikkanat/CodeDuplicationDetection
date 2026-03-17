from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


def build_voting_ensemble(random_state=42):
    """
    Soft voting ensemble of XGBoost + RandomForest + Calibrated LinearSVM.
    Optimized for speed: hist tree method, reduced RF trees, reduced SVM CV folds.
    """
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=8,
        gamma=0.1,
        reg_alpha=0.01,
        reg_lambda=1.0,
        eval_metric="logloss",
        tree_method='hist',          # histogram-based: 3-5x faster
        random_state=random_state,
        n_jobs=-1
    )

    rf = RandomForestClassifier(
        n_estimators=300,            # reduced from 500 (diminishing returns)
        max_depth=20,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=random_state
    )

    svm_base = LinearSVC(
        C=1.0,
        loss='squared_hinge',
        max_iter=5000,
        class_weight='balanced',
        random_state=random_state
    )
    svm = CalibratedClassifierCV(svm_base, cv=2)  # reduced from 3: ~33% faster

    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb),
            ('rf', rf),
            ('svm', svm)
        ],
        voting='soft',
        n_jobs=1  # inner models already use n_jobs=-1
    )

    return ensemble
