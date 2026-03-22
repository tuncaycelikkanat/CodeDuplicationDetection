from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def build_linear_svm(random_state):
    base = LinearSVC(
        C=5.0,
        penalty="l2",
        loss="squared_hinge",
        dual=False,
        max_iter=5000,
        tol=1e-3,
        class_weight="balanced",
        random_state=random_state
    )
    return CalibratedClassifierCV(base, cv=3)
