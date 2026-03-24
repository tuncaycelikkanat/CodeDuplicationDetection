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
        n_estimators=3000,          # high ceiling, early_stopping determines the actual target
        max_depth=8,                # 12->8: reduces overfitting and fits max_bin=256 into GPU VRAM
        
        # ── Learning ──
        learning_rate=0.05,         # 0.087->0.05: slower learning = better generalization
        
        # ── Sampling (stochastic boosting) ──
        subsample=0.8,              # row sampling - 80% of data per tree
        colsample_bytree=0.8,       # feature sampling per tree
        colsample_bylevel=0.7,      # feature sampling per depth level (extra variance)
        
        # ── Regularization ──
        min_child_weight=5,         # 3->5: min samples in leaf nodes to protect against noise
        gamma=0.1,                  # 0.047->0.1: more aggressive pruning
        reg_alpha=0.03,             # L1 (for sparse features)
        reg_lambda=0.15,            # L2 (weight shrinkage)
        
        # ── Class Balance ──
        scale_pos_weight=1.0,       # 0.886->1.0: balanced data, no need to heavily penalize clones
        
        # ── Early Stopping ──
        early_stopping_rounds=50,   # stop if no improvement for 50 rounds
        eval_metric="logloss",
        
        # ── System ──
        random_state=random_state,
        n_jobs=-1,
        device=xgb_device,
        max_bin=256,                # fits in VRAM on T4 with max_depth=8
        tree_method="hist"
    )

