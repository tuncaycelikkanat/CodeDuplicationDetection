import re

with open('main.py', 'r') as f:
    content = f.read()

# 1. _apply_cascade_filter
old_cascade = """    X_stage1 = X[:, :feature_count]
    y_prob_stage1 = stage1_model.predict_proba(X_stage1)[:, 1]
    easy_pos_mask = (y == 1) & (y_prob_stage1 >= threshold)
    keep_mask = ~easy_pos_mask
    return X[keep_mask], y[keep_mask], easy_pos_mask.sum()"""

new_cascade = """    X_stage1 = X[:, :feature_count]
    y_prob_stage1 = stage1_model.predict_proba(X_stage1)[:, 1]
    easy_pos_mask = (y == 1) & (y_prob_stage1 >= threshold)
    easy_neg_mask = (y == 0) & (y_prob_stage1 <= (1.0 - threshold))
    
    keep_mask = ~(easy_pos_mask | easy_neg_mask)
    
    pos_before = (y == 1).sum()
    neg_before = (y == 0).sum()
    y_filtered = y[keep_mask]
    pos_after = (y_filtered == 1).sum()
    neg_after = (y_filtered == 0).sum()
    
    print(f"    - Cascade Filter Before: {pos_before} Pos, {neg_before} Neg")
    print(f"    - Cascade Filter After:  {pos_after} Pos, {neg_after} Neg")
    print(f"    - Filtered Easy Pos: {easy_pos_mask.sum()}, Easy Neg: {easy_neg_mask.sum()}")
    
    return X[keep_mask], y_filtered, (easy_pos_mask.sum() + easy_neg_mask.sum())"""

content = content.replace(old_cascade, new_cascade)

# 2. Global PCA Leakage fix
old_pca = """        # PCA ile 768-D -> SSL_PCA_COMPONENTS-D indirgeme
        print(f"  → Fitting PCA: 768 → {SSL_PCA_COMPONENTS} dims...")
        t_pca = time.time()
        ssl_pca = PCA(n_components=SSL_PCA_COMPONENTS, random_state=DEFAULT_SEED)
        ssl_embeddings_all = ssl_pca.fit_transform(raw_ssl).astype(np.float32)
        explained = ssl_pca.explained_variance_ratio_.sum()
        print(f"  → PCA done in {time.time() - t_pca:.1f}s | Explained variance: {explained:.2%}")
        del raw_ssl
        t_features += (time.time() - t_phase_ssl)"""

new_pca = """        print(f"  → Keeping SSL embeddings raw (768 dims). PCA will be fitted per split to prevent leakage.")
        ssl_embeddings_all = raw_ssl
        ssl_pca = None
        t_features += (time.time() - t_phase_ssl)"""

content = content.replace(old_pca, new_pca)

# 3. CV Loop model instantiation removal
old_cv_instantiate = """        if args.model == "ensemble":
            from models.ensemble import build_ensemble
            model = build_ensemble(args.seed, device=args.device)
        else:
            model = build_xgboost(args.seed, device=args.device)

        print(f"  → Training {model_name}...")"""

content = content.replace(old_cv_instantiate, "")

# 4. CV loop PCA and XGBoost fix
old_cv_train_generate = """        print(f"  → Generating {num_train_pairs} train pairs...")"""
new_cv_train_generate = """        if train_ssl is not None:
            print(f"  → Fitting SSL PCA ({SSL_PCA_COMPONENTS} dims) for fold...")
            from sklearn.decomposition import PCA
            ssl_pca = PCA(n_components=SSL_PCA_COMPONENTS, random_state=args.seed)
            train_ssl = ssl_pca.fit_transform(train_ssl).astype(np.float32)
            test_ssl = ssl_pca.transform(test_ssl).astype(np.float32)

        print(f"  → Generating {num_train_pairs} train pairs...")"""
content = content.replace(old_cv_train_generate, new_cv_train_generate)

# 5. CV Loop Stage-1 Calibrated & Model Instantiate
old_cv_stage1 = """        from sklearn.ensemble import HistGradientBoostingClassifier
        stage1_model = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=args.seed)"""

new_cv_stage1 = """        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.calibration import CalibratedClassifierCV
        base_stage1 = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=args.seed)
        stage1_model = CalibratedClassifierCV(base_stage1, cv=3, method='isotonic')"""
content = content.replace(old_cv_stage1, new_cv_stage1)

old_cv_model_train = """        print(f"  → Filtered CV Train: {X_train.shape} (removed {n_removed} easy clones)")

        model.fit(X_train, y_train, verbose=False)"""

new_cv_model_train = """        print(f"  → Filtered CV Train: {X_train.shape} (removed {n_removed} easy clones)")

        if args.model == "ensemble":
            from models.ensemble import build_ensemble
            model = build_ensemble(args.seed, device=args.device)
        else:
            pos_count = max(1, (y_train == 1).sum())
            spw = (len(y_train) - pos_count) / pos_count
            model = build_xgboost(args.seed, device=args.device, scale_pos_weight=spw)

        print(f"  → Training {model_name}...")
        model.fit(X_train, y_train, verbose=False)"""
content = content.replace(old_cv_model_train, new_cv_model_train)

# 6. Standalone Train mode PCA
old_standalone_gen = """    print(f"---> Generating {num_train_pairs} train pairs (positive_ratio={args.positive_ratio})...")"""
new_standalone_gen = """    if train_ssl is not None:
        print(f"---> Fitting SSL PCA: 768 → {SSL_PCA_COMPONENTS} dims...")
        from sklearn.decomposition import PCA
        ssl_pca = PCA(n_components=SSL_PCA_COMPONENTS, random_state=RANDOM_STATE)
        train_ssl = ssl_pca.fit_transform(train_ssl).astype(np.float32)
        val_ssl = ssl_pca.transform(val_ssl).astype(np.float32)
        test_ssl = ssl_pca.transform(test_ssl).astype(np.float32)

    print(f"---> Generating {num_train_pairs} train pairs (positive_ratio={args.positive_ratio})...")"""
content = content.replace(old_standalone_gen, new_standalone_gen)

# 7. Standalone Stage-1 Calibrated
old_std_stage1 = """    from sklearn.ensemble import HistGradientBoostingClassifier
    stage1_model = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=RANDOM_STATE)"""
new_std_stage1 = """    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    base_stage1 = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=RANDOM_STATE)
    stage1_model = CalibratedClassifierCV(base_stage1, cv=3, method='isotonic')"""
content = content.replace(old_std_stage1, new_std_stage1)

# 8. Standalone Model Instantiation
old_std_model = """        t_tune = 0.0
        model = build_fn(RANDOM_STATE, device=args.device)"""
new_std_model = """        t_tune = 0.0
        if args.model == "xgboost":
            pos_count = max(1, (y_train == 1).sum())
            spw = (len(y_train) - pos_count) / pos_count
            model = build_fn(RANDOM_STATE, device=args.device, scale_pos_weight=spw)
        else:
            model = build_fn(RANDOM_STATE, device=args.device)"""
content = content.replace(old_std_model, new_std_model)


with open('main.py', 'w') as f:
    f.write(content)

print("Modifications done!")
