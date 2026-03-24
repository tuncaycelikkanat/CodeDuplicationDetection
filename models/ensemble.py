from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from models.xgboost import GPUXGBClassifier


def build_voting_ensemble(random_state=42, device="cpu"):
    """
    Soft voting ensemble of XGBoost + RandomForest + Calibrated LinearSVM.
    Optimized for speed: hist tree method, reduced RF trees, reduced SVM CV folds.
    """
    xgb_device = device if device == "cuda" else "cpu"

    # NOTE: These are generic defaults. After adding new features (e.g. True LLVM IR),
    # re-run `--tune` to find optimal hyperparameters for the updated feature set.
    xgb = GPUXGBClassifier(
        n_estimators=1500,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.05,
        reg_alpha=0.03,
        reg_lambda=0.15,
        scale_pos_weight=1.0,
        eval_metric="logloss",
        tree_method='hist',
        random_state=random_state,
        n_jobs=-1,
        device=xgb_device,
        max_bin=256
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

