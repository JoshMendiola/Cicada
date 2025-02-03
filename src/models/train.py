from collections import Counter
from datetime import datetime

import numpy as np
import xgboost as xgb
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import BorderlineSMOTE
from scipy.sparse import issparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.feature_extractor import extract_features
from utils.feature_alignment import add_endpoint_features


def train_improved_model(logs_df, max_features=48):
    print("\nFeature Generation Details:")
    print("---------------------------")

    # 1. FEATURE EXTRACTION AND PREPROCESSING
    # --------------------------------------
    # Extract base features
    X = extract_features(logs_df)
    print("Initial feature columns:", X.columns.tolist())

    # Vectorize URLs
    vectorizer = TfidfVectorizer(max_features=max_features)
    all_paths = logs_df['path'].fillna('')
    vectorizer.fit(all_paths)
    path_features = vectorizer.transform(all_paths)
    print("TF-IDF features shape:", path_features.shape)

    # Add additional endpoint features
    X = add_endpoint_features(X, logs_df)

    # Split columns by type
    categorical_columns = ['method']
    numerical_columns = [col for col in X.columns if col not in categorical_columns]
    print("Numerical columns:", numerical_columns)
    print("Categorical columns:", categorical_columns)

    # Preprocess features
    onehot = OneHotEncoder(handle_unknown='ignore')
    onehot.fit(X[categorical_columns])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', onehot, categorical_columns)
        ])
    X_preprocessed = preprocessor.fit_transform(X)
    path_features = vectorizer.transform(all_paths)

    # Convert to dense arrays and combine
    if issparse(X_preprocessed):
        X_preprocessed = X_preprocessed.toarray()
    if issparse(path_features):
        path_features = path_features.toarray()
    X_combined = np.hstack((X_preprocessed, path_features))

    print("Training shapes:")
    print(f"Preprocessed features shape: {X_preprocessed.shape}")
    print(f"Path features shape: {path_features.shape}")
    print(f"Combined features shape: {X_combined.shape}")

    # Get feature names for later analysis
    num_feature_names = numerical_columns
    cat_feature_names = onehot.get_feature_names_out(categorical_columns).tolist()
    tfidf_feature_names = vectorizer.get_feature_names_out().tolist()
    all_feature_names = num_feature_names + cat_feature_names + tfidf_feature_names

    # 2. DATA BALANCING
    # ----------------
    y = logs_df['is_attack'].astype(int)
    print(f"\nTotal samples: {len(logs_df)}")
    print(f"Class distribution before resampling: {Counter(y)}")

    # Use BorderlineSMOTE for better quality synthetic samples
    b_smote = BorderlineSMOTE(
        sampling_strategy=0.3,
        random_state=42,
        k_neighbors=5,
        m_neighbors=10
    )

    # Clean up borderline cases with SMOTETomek
    smote_tomek = SMOTETomek(
        sampling_strategy=0.3,
        random_state=42
    )

    # Apply resampling pipeline
    X_res, y_res = b_smote.fit_resample(X_combined, y)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_res, y_res)
    print(f"Class distribution after resampling: {Counter(y_resampled)}")

    # 3. TRAIN/VALIDATION/TEST SPLIT
    # -----------------------------
    # Split into three sets for better evaluation
    X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 4. MODEL TRAINING
    # ---------------
    # Create base models with optimized parameters
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        min_samples_split=5,
        random_state=42
    )

    xgb_model = xgb.XGBClassifier(
        eval_metric=['logloss', 'auc'],
        scale_pos_weight=3,
        max_depth=6,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.1,
        early_stopping_rounds=10
    )

    # Create and train ensemble model
    ensemble_model = CalibratedClassifierCV(
        VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('xgb', xgb_model)
            ],
            voting='soft'
        ),
        cv=5
    )

    # Train the ensemble
    print("\nTraining ensemble model...")
    ensemble_model.fit(X_train, y_train)

    # 5. MODEL EVALUATION AND ANALYSIS
    # ------------------------------
    # Find optimal prediction threshold
    print("\nOptimizing prediction threshold...")
    probas = ensemble_model.predict_proba(X_val)
    thresholds = np.arange(0.1, 1.0, 0.1)
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (probas[:, 1] >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Optimal threshold: {best_threshold:.2f}")

    # Make predictions using optimal threshold
    test_probas = ensemble_model.predict_proba(X_test)
    y_pred = (test_probas[:, 1] >= best_threshold).astype(int)

    # Print overall metrics
    print("\nModel Evaluation (with optimal threshold):")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Analyze feature importance (from Random Forest component)
    rf_model = ensemble_model.estimators_[0].estimators_[0]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nTop 10 Most Important Features:")
    for f in range(min(10, len(all_feature_names))):
        print(f"{f + 1}. {all_feature_names[indices[f]]} ({importances[indices[f]]:.4f})")

    # Error analysis
    false_positives = logs_df[(y_test == 0) & (y_pred == 1)]
    print("\nFalse Positive Analysis:")
    print("Most common paths in false positives:")
    print(false_positives['path'].value_counts().head())
    print("\nMethods in false positives:")
    print(false_positives['method'].value_counts())

    # Analyze borderline cases
    borderline_mask = (test_probas[:, 1] > 0.4) & (test_probas[:, 1] < 0.6)
    borderline_cases = logs_df[borderline_mask]
    print("\nBorderline Cases Analysis:")
    print(f"Number of borderline cases: {len(borderline_cases)}")
    print("Methods in borderline cases:")
    print(borderline_cases['method'].value_counts())

    # Cross validation scores
    cv_scores = cross_val_score(ensemble_model, X_combined, y, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Save model metadata
    model_metadata = {
        'training_date': datetime.now().isoformat(),
        'optimal_threshold': best_threshold,
        'feature_names': all_feature_names,
        'performance_metrics': {
            'cv_score_mean': cv_scores.mean(),
            'cv_score_std': cv_scores.std(),
            'best_f1': best_f1
        }
    }

    return (ensemble_model, vectorizer, preprocessor, all_feature_names, onehot,
            best_threshold, model_metadata)