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
        # ── Tree Capacity ──
        n_estimators=3000,          # high ceiling, early_stopping determines actual count
        max_depth=6,                # 8→6: shallower trees = less memorization
        
        # ── Learning ──
        learning_rate=0.03,         # 0.05→0.03: slower learning forces better generalization
        
        # ── Sampling (stochastic boosting) ──
        subsample=0.7,              # 0.8→0.7: more aggressive row sampling
        colsample_bytree=0.6,       # 0.8→0.6: forces each tree to use fewer features
        colsample_bylevel=0.6,      # 0.7→0.6: extra feature randomness per level
        
        # ── Regularization ──
        min_child_weight=8,         # 5→8: require more samples per leaf
        gamma=0.3,                  # 0.1→0.3: much stricter pruning threshold
        reg_alpha=0.1,              # 0.03→0.1: stronger L1 sparsity penalty
        reg_lambda=1.0,             # 0.15→1.0: strong L2 weight shrinkage
        
        # ── Class Balance ──
        scale_pos_weight=1.0,
        
        # ── Early Stopping ──
        early_stopping_rounds=30,   # 50→30: stop sooner to prevent overfitting
        eval_metric="logloss",
        
        # ── System ──
        random_state=random_state,
        n_jobs=-1,
        device=xgb_device,
        max_bin=256,
        tree_method="hist"
    )

