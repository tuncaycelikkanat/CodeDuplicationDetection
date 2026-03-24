import numpy as np
import scipy.sparse as sp
from xgboost import XGBClassifier, DMatrix


class GPUXGBClassifier(XGBClassifier):
    """
    XGBClassifier wrapper that keeps predict/predict_proba on GPU.
    Converts input data to DMatrix with the correct device to avoid
    the CPU→GPU fallback warning and maximize GPU utilization.
    """

    def predict(self, X, **kwargs):
        if self.device == "cuda" and (sp.issparse(X) or isinstance(X, np.ndarray)):
            dmat = DMatrix(X)
            # get_booster().predict returns raw margin, apply threshold
            raw = self.get_booster().predict(dmat)
            return (raw > 0.5).astype(np.int32)
        return super().predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        if self.device == "cuda" and (sp.issparse(X) or isinstance(X, np.ndarray)):
            dmat = DMatrix(X)
            pos_proba = self.get_booster().predict(dmat)
            return np.column_stack([1 - pos_proba, pos_proba])
        return super().predict_proba(X, **kwargs)


def build_xgboost(random_state, device="cpu"):
    # XGBoost device mapping: 'cuda' for NVIDIA, 'cpu' for everything else
    xgb_device = device if device == "cuda" else "cpu"

    return GPUXGBClassifier(
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
        max_bin=256,
        tree_method="hist"
    )
