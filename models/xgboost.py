from xgboost import XGBClassifier

def build_xgboost(random_state):
    return XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=8,
        gamma=0.1,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1
    )
