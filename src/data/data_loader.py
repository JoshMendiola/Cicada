import json

import pandas as pd


def load_json_logs(file_path):
    with open(file_path, 'r') as f:
        try:
            logs = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return pd.DataFrame()

    if not logs:
        print("No valid JSON objects found in the file.")
        return pd.DataFrame()

    df = pd.DataFrame(logs)

    required_columns = {
        'timestamp': 'N/A', 'type': 'N/A', 'ip': 'N/A', 'method': 'UNKNOWN',
        'path': 'UNKNOWN', 'headers': '{}', 'body': None, 'is_attack': 0
    }

    for column, default_value in required_columns.items():
        if column not in df.columns:
            df[column] = default_value

    df['headers'] = df['headers'].apply(
        lambda x: x if isinstance(x, dict) else json.loads(x) if isinstance(x, str) else {})
    df['path'] = df['path'].fillna('UNKNOWN')
    df['query'] = df['path'].apply(lambda x: x.split('?')[1] if isinstance(x, str) and '?' in x else '')
    df['path'] = df['path'].apply(lambda x: x.split('?')[0] if isinstance(x, str) and '?' in x else x)

    print(f"Loaded {len(df)} log entries")
    print(f"Columns: {df.columns.tolist()}")
    if not df.empty:
        print(f"Sample entry:\n{df.iloc[0].to_dict()}")
    else:
        print("No entries to display.")

    return df
