from sklearn.svm import LinearSVC

def build_linear_svm(random_state):
    model = LinearSVC(
        C=5.0,
        penalty="l2",
        loss="squared_hinge",
        dual=True,
        max_iter=5000,
        class_weight="balanced",
        random_state=random_state
    )
    return model
