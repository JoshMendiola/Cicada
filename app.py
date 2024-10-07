import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json
import re


def load_csic_data(file_path):
    df = pd.read_csv(file_path, encoding='latin1')
    df['full_path'] = df['Method'] + ' ' + df['URL']
    return df


def extract_features(row):
    features = {}
    features['path_length'] = len(str(row['URL']))
    features['num_params'] = str(row['URL']).count('=')

    # Extract the numeric value from the 'Content-Length' header
    if pd.notna(row['lenght']):
        content_length = row['lenght'].split(': ')[-1]
        features['body_length'] = int(content_length) if content_length.isdigit() else 0
    else:
        features['body_length'] = 0

    features['user_agent_length'] = len(str(row['User-Agent'])) if pd.notna(row['User-Agent']) else 0
    return pd.Series(features)


def is_suspicious(log):
    path = log['path']

    # Check for SQL injection attempts
    if re.search(r'(union|select|from|where)\s+.*\s+(union|select|from|where)', path, re.IGNORECASE):
        return True

    # Check for path traversal attempts
    if '..' in path or '%2e%2e' in path.lower():
        return True

    # Check for obvious script injection
    if re.search(r'<script.*?>.*?</script>', path, re.IGNORECASE):
        return True

    # Check for attempts to access sensitive files
    if re.search(r'/(passwd|shadow|etc/|wp-config\.php)', path, re.IGNORECASE):
        return True

    # Check for unusual file extensions that might indicate malicious activity
    if re.search(r'\.(php|asp|aspx|jsp|cgi)$', path, re.IGNORECASE):
        return True

    return False


def train_model(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X_scaled)

    return clf, scaler


def analyze_log(log, clf, scaler):
    if is_suspicious(log):
        return 1, 1.0  # Definitely suspicious

    features = extract_features_from_log(log)
    features_df = pd.DataFrame([features])
    features_scaled = scaler.transform(features_df)

    # -1 for outliers (potential attacks), 1 for inliers (normal)
    prediction = clf.predict(features_scaled)[0]
    score = clf.score_samples(features_scaled)[0]

    # Convert to probability-like score
    probability = 1 - (score - clf.offset_) / (np.max(clf.score_samples(features_scaled)) - clf.offset_)

    return -1 if prediction == -1 else 0, probability


def extract_features_from_log(log):
    features = {}
    features['path_length'] = len(str(log['path']))
    features['num_params'] = str(log['path']).count('=')
    features['body_length'] = len(str(log.get('body', '')))
    features['user_agent_length'] = len(str(log.get('headers', {}).get('User-Agent', '')))
    return features


def main(csic_file_path, custom_log_file_path):
    print("Loading CSIC data...")
    csic_df = load_csic_data(csic_file_path)

    print("Extracting features...")
    features_df = csic_df.apply(extract_features, axis=1)

    # Ensure all features are numeric
    for column in features_df.columns:
        features_df[column] = pd.to_numeric(features_df[column], errors='coerce')
    features_df = features_df.fillna(0)

    print("Training model...")
    model, scaler = train_model(features_df)

    print("Model training complete!")

    print("Analyzing custom logs...")
    with open(custom_log_file_path, 'r') as file:
        logs = json.load(file)

    results = []
    for log in logs:
        if log['type'] == 'REQUEST':
            prediction, probability = analyze_log(log, model, scaler)

            if prediction != 0:
                print("\nPotential attack detected. Full log entry:")
                print(json.dumps(log, indent=2))
                print(f"Attack probability: {probability:.2f}")

            results.append({
                'timestamp': log['timestamp'],
                'method': log['method'],
                'path': log['path'],
                'prediction': prediction,
                'probability': probability
            })

    results_df = pd.DataFrame(results)
    print("\nAnalysis summary:")
    print(results_df['prediction'].value_counts())

    # Optionally, save the analysis results
    results_df.to_csv('log_analysis_results.csv', index=False)


if __name__ == "__main__":
    csic_file_path = 'csic_database.csv'
    custom_log_file_path = 'logs.json'
    main(csic_file_path, custom_log_file_path)