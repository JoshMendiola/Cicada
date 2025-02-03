import logging

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from utils.feature_alignment import extract_features_consistent


def evaluate_model(model, data, vectorizer, preprocessor, all_feature_names, onehot=None):
    """Evaluate model performance"""
    print("\nEvaluating model...")
    if onehot is None:
        onehot = preprocessor.named_transformers_['cat']

    X_combined = extract_features_consistent(data, vectorizer, preprocessor, all_feature_names, onehot)
    y = data['is_attack'].astype(int)

    # Use detailed evaluation
    method_stats = evaluate_model_detailed(model, X_combined, y, data)

    return method_stats


def evaluate_model_detailed(model, X_test, y_test, data_test):
    """Detailed model evaluation breaking down performance by HTTP method and other features"""
    y_pred = model.predict(X_test)

    # Overall metrics
    print("\nOverall Model Performance:")
    print(classification_report(y_test, y_pred))
    print("Overall Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Analysis by HTTP Method
    print("\nAnalysis by HTTP Method:")
    methods = data_test['method'].unique()
    method_stats = {}

    for method in methods:
        method_mask = data_test['method'] == method
        method_y = y_test[method_mask]
        method_pred = y_pred[method_mask]

        # Skip if no samples for this method
        if len(method_y) == 0:
            continue

        method_stats[method] = {
            'total_requests': len(method_y),
            'true_positives': ((method_y == 1) & (method_pred == 1)).sum(),
            'false_positives': ((method_y == 0) & (method_pred == 1)).sum(),
            'true_negatives': ((method_y == 0) & (method_pred == 0)).sum(),
            'false_negatives': ((method_y == 1) & (method_pred == 0)).sum()
        }

        # Calculate rates with zero division protection
        denominator = (method_stats[method]['false_positives'] + method_stats[method]['true_negatives'])
        method_stats[method]['false_positive_rate'] = (
            method_stats[method]['false_positives'] / denominator if denominator > 0 else 0
        )

        denominator = (method_stats[method]['true_positives'] + method_stats[method]['false_positives'])
        method_stats[method]['precision'] = (
            method_stats[method]['true_positives'] / denominator if denominator > 0 else 0
        )

        print(f"\n{method} Method Statistics:")
        print(f"Total Requests: {method_stats[method]['total_requests']}")
        print(f"False Positives: {method_stats[method]['false_positives']}")
        print(f"False Positive Rate: {method_stats[method]['false_positive_rate']:.2%}")
        print(f"Precision: {method_stats[method]['precision']:.2%}")

        unique_labels = np.unique(np.concatenate([method_y, method_pred]))
        if len(unique_labels) > 1:
            method_cm = confusion_matrix(method_y, method_pred)
            print(f"{method} Confusion Matrix:")
            print(method_cm)

    # Analyze false positives in detail
    false_positives_mask = (y_test == 0) & (y_pred == 1)
    false_positives_data = data_test[false_positives_mask]

    print("\nFalse Positives Analysis:")
    print(f"Total False Positives: {len(false_positives_data)}")
    print("\nFalse Positives by Method:")
    print(false_positives_data['method'].value_counts())

    # Analyze common patterns in false positives
    if 'path' in false_positives_data.columns:
        print("\nMost Common Paths in False Positives:")
        print(false_positives_data['path'].value_counts().head())

    if 'body' in false_positives_data.columns:
        has_body = false_positives_data['body'].notna().sum()
        print(f"\nFalse Positives with Body: {has_body} ({has_body / len(false_positives_data):.2%})")

    return method_stats