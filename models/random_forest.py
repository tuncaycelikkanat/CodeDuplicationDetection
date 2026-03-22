from sklearn.ensemble import RandomForestClassifier

def build_random_forest(random_state):
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=4,
        max_features="sqrt",
        max_samples=0.5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state
    )
