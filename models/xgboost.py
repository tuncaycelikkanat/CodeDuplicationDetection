from xgboost import XGBClassifier

def build_xgboost(random_state):
    return XGBClassifier(
        n_estimators=400,  # ağaç sayısını biraz düşürdük, overfit azaltır
        max_depth=6,  # daha sığ ağaç, overfit'i azaltır
        learning_rate=0.05,  # daha küçük adımlar, daha stabil öğrenme
        subsample=0.7,  # rastgele örnekleme ile genelleme artar
        colsample_bytree=0.7,  # feature seçimiyle overfit azalır
        min_child_weight=8,  # yaprak başına minimum örnek sayısı
        gamma=0.1,  # yaprak bölünmesi için minimum kazanç, overfit azaltır
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1
    )
