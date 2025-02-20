import logging

import numpy as np
from scipy.sparse import issparse

from src.features.feature_extractor import extract_features


def add_endpoint_features(X, data):
    """Add more sophisticated features"""
    # Add endpoint-specific features
    X['is_search_endpoint'] = data['path'].str.contains('/search').astype(int)
    X['is_login_endpoint'] = data['path'].str.contains('/login').astype(int)
    X['is_root_endpoint'] = (data['path'] == '/').astype(int)

    # Add complexity metrics
    X['path_depth'] = data['path'].str.count('/')
    X['path_length'] = data['path'].str.len()

    # Query analysis
    X['has_query'] = data['query'].str.len() > 0
    X['query_param_count'] = data['query'].str.count('&') + 1

    # Additional security-focused features
    X['has_special_chars'] = data['path'].str.contains('[<>{}()\'"]').astype(int)
    X['has_sql_keywords'] = data['path'].str.lower().str.contains(
        'select|insert|update|delete|union|drop'
    ).astype(int)

    return X


def extract_features_consistent(data, vectorizer, preprocessor, all_feature_names, onehot=None):
    """Extract features ensuring consistency with training features"""
    # Extract base features
    X = extract_features(data)

    # Add the endpoint features that were present during training
    X = add_endpoint_features(X, data)

    path_features = vectorizer.transform(data['path'].fillna(''))

    categorical_columns = ['method']
    numerical_columns = [col for col in X.columns if col not in categorical_columns]

    # Get onehot encoder from preprocessor if not provided
    if onehot is None:
        onehot = preprocessor.named_transformers_['cat']

    try:
        X_cat = onehot.transform(X[categorical_columns])
        X_num = preprocessor.named_transformers_['num'].transform(X[numerical_columns])
    except ValueError as e:
        print("Feature mismatch detected. Available features:", X.columns.tolist())
        print("Expected numerical features:", numerical_columns)
        raise e

    if issparse(X_num):
        X_num = X_num.toarray()
    if issparse(X_cat):
        X_cat = X_cat.toarray()
    if issparse(path_features):
        path_features = path_features.toarray()

    X_combined = np.hstack((X_num, X_cat, path_features))

    return X_combined
