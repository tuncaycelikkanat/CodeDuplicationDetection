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
        n_estimators=3000,          # yüksek tavan, early_stopping gerçek sayıyı belirler
        max_depth=8,                # 12→8: overfitting azaltır + GPU VRAM'e max_bin=256 sığar
        
        # ── Learning ──
        learning_rate=0.05,         # 0.087→0.05: daha yavaş öğrenme = daha iyi genelleme
        
        # ── Sampling (stochastic boosting) ──
        subsample=0.8,              # satır örnekleme — her ağaçta %80 veri
        colsample_bytree=0.8,       # ağaç başına feature örnekleme
        colsample_bylevel=0.7,      # derinlik seviyesi başına feature örnekleme (ek çeşitlilik)
        
        # ── Regularization ──
        min_child_weight=5,         # 3→5: yaprak node'larda minimum örnek — gürültüye karşı koruma
        gamma=0.1,                  # 0.047→0.1: daha agresif pruning
        reg_alpha=0.03,             # L1 (sparse feature'lar için)
        reg_lambda=0.15,            # L2 (weight shrinkage)
        
        # ── Class Balance ──
        scale_pos_weight=1.0,       # 0.886→1.0: balanced data, clone'ları cezalandırma
        
        # ── Early Stopping ──
        early_stopping_rounds=50,   # 50 round iyileşme olmazsa dur
        eval_metric="logloss",
        
        # ── System ──
        random_state=random_state,
        n_jobs=-1,
        device=xgb_device,
        max_bin=256,                # T4'te max_depth=8 ile VRAM'e sığar
        tree_method="hist"
    )

