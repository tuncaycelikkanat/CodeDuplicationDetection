from sklearn.ensemble import RandomForestClassifier

def build_random_forest(random_state):
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=26,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state
    )
