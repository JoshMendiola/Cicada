import logging
from collections import Counter

import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.feature_extractor import extract_features


def train_improved_model(logs_df, max_features=48):
    print("\nFeature Generation Details:")
    print("---------------------------")

    # Extract the features
    X = extract_features(logs_df)
    print("Initial feature columns:", X.columns.tolist())

    # Vectorize the URLS
    vectorizer = TfidfVectorizer(max_features=max_features)
    all_paths = logs_df['path'].fillna('')
    vectorizer.fit(all_paths)
    path_features = vectorizer.transform(all_paths)
    print("TF-IDF features shape:", path_features.shape)

    # Split up categorical and numerical columns
    categorical_columns = ['method']
    numerical_columns = [col for col in X.columns if col not in categorical_columns]
    print("Numerical columns:", numerical_columns)
    print("Categorical columns:", categorical_columns)

    # Preprocess all the features with one hot encoding
    onehot = OneHotEncoder(handle_unknown='ignore')
    onehot.fit(X[categorical_columns])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', onehot, categorical_columns)
        ])
    X_preprocessed = preprocessor.fit_transform(X)
    path_features = vectorizer.transform(all_paths)

    print("Training shapes:")
    print(f"Preprocessed features shape: {X_preprocessed.shape}")
    print(f"Path features shape: {path_features.shape}")

    # Checks if it is a sparse array and combines the features into one matrix
    if issparse(X_preprocessed):
        X_preprocessed = X_preprocessed.toarray()
    if issparse(path_features):
        path_features = path_features.toarray()
    X_combined = np.hstack((X_preprocessed, path_features))
    print(f"Combined features shape: {X_combined.shape}")

    # Get feature names
    num_feature_names = numerical_columns
    cat_feature_names = onehot.get_feature_names_out(categorical_columns).tolist()
    tfidf_feature_names = vectorizer.get_feature_names_out().tolist()
    all_feature_names = num_feature_names + cat_feature_names + tfidf_feature_names

    logging.debug(f"Number of numerical features: {len(num_feature_names)}")
    logging.debug(f"Number of categorical features: {len(cat_feature_names)}")
    logging.debug(f"Number of TF-IDF features: {len(tfidf_feature_names)}")
    logging.debug(f"Total number of features: {len(all_feature_names)}")

    logging.info(f"Training data shape: {X_combined.shape}")

    # Add new is_attack column
    y = logs_df['is_attack'].astype(int)

    print(f"Total samples: {len(logs_df)}")
    print(f"Class distribution before SMOTE: {Counter(y)}")

    # Apply SMOTE, which artificially inflates positives (attacks) in the data set
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_combined, y)

    print(f"Class distribution after SMOTE: {Counter(y_resampled)}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Create Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Create XGBoost model
    xgb_model = xgb.XGBClassifier(eval_metric='logloss')

    # Create an ensemble model
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model)
        ],
        voting='soft'
    )

    # Train the ensemble model
    ensemble_model.fit(X_train, y_train)

    # Evaluate the ensemble model
    y_pred = ensemble_model.predict(X_test)
    print("\nEnsemble Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return ensemble_model, vectorizer, preprocessor, all_feature_names, onehot
