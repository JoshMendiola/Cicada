import pandas as pd


def extract_features(data):
    features = pd.DataFrame({
        'method': data['method'],
        'has_body': data['body'].notna().astype(int),
        'header_count': data['headers'].apply(lambda x: len(x) if isinstance(x, dict) else 0),
        'has_query': data['query'].notna().astype(int),
        'content_type': data['headers'].apply(lambda x: 1 if 'content-type' in str(x).lower() else 0),
        'user_agent': data['headers'].apply(lambda x: 1 if 'user-agent' in str(x).lower() else 0),
        'body_length': data['body'].fillna('').str.len(),
        'path_depth': data['path'].str.count('/'),
        'has_sql_keywords': data['body'].fillna('').str.lower().str.contains(
            'select|from|where|union|insert|update|delete').astype(int),
        'has_script_tags': data['body'].fillna('').str.lower().str.contains('<script').astype(int)
    })

    return features
