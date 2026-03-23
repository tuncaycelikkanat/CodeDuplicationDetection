from xgboost import XGBClassifier

def build_xgboost(random_state, device="cpu"):
    # XGBoost device mapping: 'cuda' for NVIDIA, 'cpu' for everything else (Intel XPU not natively in standard pip)
    xgb_device = device if device == "cuda" else "cpu"
    
    return XGBClassifier(
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
        random_state=random_state,
        n_jobs=-1,
        device=xgb_device,
        max_bin=64,
        tree_method="hist"
    )
