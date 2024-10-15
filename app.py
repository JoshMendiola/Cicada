import json
import os
import time

import boto3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from utils.aws import upload_to_s3, download_from_s3

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
S3_BUCKET_NAME = 'cicada-data'
s3_client = boto3.client('s3')


def load_csic_data(file_path):
    if not os.path.exists(file_path):
        s3_path = os.path.basename(file_path)
        download_from_s3(s3_path, file_path)

    df = pd.read_csv(file_path, encoding='latin1')
    print("CSIC Dataset columns:", df.columns)
    print("CSIC Dataset sample:", df.head())
    print("CSIC Dataset info:", df.info())

    if 'classification' in df.columns:
        df['is_attack'] = (df['classification'].astype(str).str.lower() != 'normal').astype(int)
    else:
        print("Warning: 'classification' column not found in CSIC dataset")
        df['is_attack'] = 0

    df['http_method'] = df['Method'] if 'Method' in df.columns else 'UNKNOWN'
    df['path'] = df['URL'] if 'URL' in df.columns else ''

    return df[['http_method', 'path', 'is_attack']]


def load_cidds_data(file_path):
    if not os.path.exists(file_path):
        s3_path = os.path.basename(file_path)
        download_from_s3(s3_path, file_path)

    df = pd.read_parquet(file_path)
    print(f"CIDDS Dataset columns ({file_path}):", df.columns)
    print(f"CIDDS Dataset sample ({file_path}):", df.head())
    print(f"CIDDS Dataset info ({file_path}):", df.info())

    if 'label' in df.columns:
        df['is_attack'] = (df['label'].astype(str) != 'normal').astype(int)
    else:
        print(f"Warning: 'label' column not found in CIDDS dataset ({file_path})")
        df['is_attack'] = 0

    df['http_method'] = 'UNKNOWN'
    if 'proto' in df.columns and 'label' in df.columns:
        df['path'] = df['proto'].astype(str) + '://' + df['label'].astype(str)
    else:
        print(f"Warning: 'proto' or 'label' column not found in CIDDS dataset ({file_path})")
        df['path'] = ''

    return df[['http_method', 'path', 'is_attack']]


def load_json_logs(file_path):
    if not os.path.exists(file_path):
        s3_path = os.path.basename(file_path)
        download_from_s3(s3_path, file_path)

    with open(file_path, 'r') as f:
        logs = json.load(f)
    df = pd.DataFrame(logs)

    # Assume 'is_attack' field exists in logs, if not, provide a default value
    df['is_attack'] = df.get('is_attack', 0)

    df['http_method'] = df['method'] if 'method' in df.columns else 'UNKNOWN'
    df['path'] = df['path'] if 'path' in df.columns else ''
    return df[['http_method', 'path', 'is_attack']]


def test_on_logs(model, vectorizer, json_path):
    print("\nTesting model on logs...")
    logs_df = load_json_logs(json_path)

    le = LabelEncoder()
    logs_df['http_method'] = le.fit_transform(logs_df['http_method'])
    path_features = vectorizer.transform(logs_df['path'])

    X_test = np.hstack((logs_df[['http_method']].values, path_features.toarray()))

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    logs_df['predicted_attack'] = predictions
    logs_df['attack_probability'] = probabilities

    print("Test Results:")
    print(f"Total logs: {len(logs_df)}")
    print(f"Actual attacks: {logs_df['is_attack'].sum()}")
    print(f"Predicted attacks: {sum(predictions)}")

    print("\nClassification Report:")
    print(classification_report(logs_df['is_attack'], predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(logs_df['is_attack'], predictions))

    print("\nSample of correctly predicted attacks:")
    print(logs_df[(logs_df['is_attack'] == 1) & (logs_df['predicted_attack'] == 1)].head())

    print("\nSample of missed attacks (false negatives):")
    print(logs_df[(logs_df['is_attack'] == 1) & (logs_df['predicted_attack'] == 0)].head())

    print("\nSample of false positives:")
    print(logs_df[(logs_df['is_attack'] == 0) & (logs_df['predicted_attack'] == 1)].head())

    return logs_df


def combine_datasets(csic_path, cidds_external_path, cidds_openstack_path):
    print("Loading datasets...")
    dataframes = []

    try:
        csic_df = load_csic_data(csic_path)
        dataframes.append(csic_df)
        print(f"CSIC Dataset loaded: {len(csic_df)} samples, {csic_df['is_attack'].sum()} attacks")
    except Exception as e:
        print(f"Error loading CSIC data: {e}")

    for path in [cidds_external_path, cidds_openstack_path]:
        try:
            cidds_df = load_cidds_data(path)
            dataframes.append(cidds_df)
            print(f"CIDDS Dataset loaded from {path}: {len(cidds_df)} samples, {cidds_df['is_attack'].sum()} attacks")
        except Exception as e:
            print(f"Error loading CIDDS data from {path}: {e}")

    if not dataframes:
        raise ValueError("No data available after attempting to load all datasets.")

    print("Combining datasets...")
    combined_df = pd.concat(dataframes, ignore_index=True)

    print("Combined Dataset statistics:")
    print(f"Total samples: {len(combined_df)}")
    print(f"Attack samples: {combined_df['is_attack'].sum()}")
    print(f"Normal samples: {len(combined_df) - combined_df['is_attack'].sum()}")

    return combined_df


def preprocess_data(df):
    le = LabelEncoder()
    df['http_method'] = le.fit_transform(df['http_method'])

    vectorizer = TfidfVectorizer(max_features=1000)
    path_features = vectorizer.fit_transform(df['path'])

    X = np.hstack((df[['http_method']].values, path_features.toarray()))
    y = df['is_attack'].values

    return X, y, vectorizer


def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest model...")
    n_estimators = 100
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1, verbose=0)

    start_time = time.time()

    # Train the entire forest at once
    rf.fit(X_train, y_train)

    # Calculate and print the training time
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    print(f"Average time per tree: {total_time / n_estimators:.2f} seconds")

    print("\nValidating the model...")
    y_pred = rf.predict(X_val)
    print("Validation Results:")
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    return rf


def main():
    # Use EC2 instance storage for temporary files
    PROJECT_ROOT = '/tmp'
    csic_path = os.path.join(PROJECT_ROOT, 'csic_database.csv')
    cidds_external_path = os.path.join(PROJECT_ROOT, 'cidds-001-externalserver.parquet')
    cidds_openstack_path = os.path.join(PROJECT_ROOT, 'cidds-001-openstack.parquet')
    logs_path = os.path.join(PROJECT_ROOT, 'logs.json')

    combined_df = combine_datasets(csic_path, cidds_external_path, cidds_openstack_path)

    if len(combined_df) > 0:
        X, y, vectorizer = preprocess_data(combined_df)

        print("\nStarting model training...")
        print(f"Total samples: {len(X)}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Class distribution: {np.bincount(y)}")
        print("-" * 50)

        model = train_model(X, y)

        results_df = test_on_logs(model, vectorizer, logs_path)
        results_csv_path = os.path.join(PROJECT_ROOT, 'analyzed_logs.csv')
        results_df.to_csv(results_csv_path, index=False)
        upload_to_s3(results_csv_path, 'analyzed_logs.csv')
        print("\nResults saved to S3: analyzed_logs.csv")

        # Save the model and vectorizer for future use
        model_path = os.path.join(PROJECT_ROOT, 'ids_model.joblib')
        vectorizer_path = os.path.join(PROJECT_ROOT, 'ids_vectorizer.joblib')
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        upload_to_s3(model_path, 'ids_model.joblib')
        upload_to_s3(vectorizer_path, 'ids_vectorizer.joblib')
        print("Model and vectorizer saved to S3.")


if __name__ == "__main__":
    main()
