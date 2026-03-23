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
        n_estimators=1808,
        max_depth=12,
        learning_rate=0.08741808323698248,
        subsample=0.8260575232041659,
        colsample_bytree=0.8478684057508008,
        min_child_weight=3,
        gamma=0.0469390212265333,
        reg_alpha=0.028217344413071523,
        reg_lambda=0.1389890374632879,
        scale_pos_weight=0.8861939156810622,
        eval_metric="logloss",
        tree_method='hist',
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
