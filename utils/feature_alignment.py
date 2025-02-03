def align_features(X, feature_names, all_feature_names):
    X_aligned = np.zeros((X.shape[0], len(all_feature_names)))
    for i, name in enumerate(feature_names):
        if name in all_feature_names:
            j = all_feature_names.index(name)
            X_aligned[:, j] = X[:, i]
    return X_aligned


def extract_features_consistent(data, vectorizer, preprocessor, all_feature_names, onehot):
    X = extract_features(data)
    path_features = vectorizer.transform(data['path'].fillna(''))

    categorical_columns = ['method']
    numerical_columns = [col for col in X.columns if col not in categorical_columns]

    X_cat = onehot.transform(X[categorical_columns])
    X_num = preprocessor.named_transformers_['num'].transform(X[numerical_columns])

    if issparse(X_num):
        X_num = X_num.toarray()
    if issparse(X_cat):
        X_cat = X_cat.toarray()
    if issparse(path_features):
        path_features = path_features.toarray()

    X_combined = np.hstack((X_num, X_cat, path_features))

    # Ensure feature alignment
    current_feature_names = (
            numerical_columns +
            onehot.get_feature_names_out(categorical_columns).tolist() +
            vectorizer.get_feature_names_out().tolist()
    )

    X_aligned = np.zeros((X_combined.shape[0], len(all_feature_names)))
    for i, name in enumerate(current_feature_names):
        if name in all_feature_names:
            j = all_feature_names.index(name)
            X_aligned[:, j] = X_combined[:, i]

    logging.debug(f"Extracted features shape: {X_aligned.shape}")
    return X_aligned