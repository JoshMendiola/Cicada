import json
import os

import pandas as pd

from src.models.train import train_improved_model
from utils.feature_alignment import extract_features_consistent


def continuous_improvement_loop(model, vectorizer, preprocessor, all_feature_names,
                                onehot, initial_data_df, results_path):
    all_reviewed_data = pd.DataFrame()
    review_batch_size = 50
    min_attacks_per_batch = 10

    initial_feature_count = len(all_feature_names)

    while True:
        if len(all_reviewed_data) >= 100:
            print(f"Retraining model with {len(all_reviewed_data)} reviewed samples...")
            # Use same max_features as initial training
            model, vectorizer_new, preprocessor_new, all_feature_names_new, onehot_new = train_improved_model(
                all_reviewed_data,
                max_features=vectorizer.max_features  # Use same max_features
            )
            print("Model retrained.")

            # Verify feature dimensions match
            if len(all_feature_names_new) != initial_feature_count:
                print(
                    f"WARNING: Feature count mismatch. "
                    f"Expected {initial_feature_count}, got {len(all_feature_names_new)}")

        if os.path.exists(results_path):
            data_df = pd.read_csv(results_path)
            print("Loaded previous results for continued review.")
        else:
            print("No previous results found. Using initial data.")
            data_df = initial_data_df.copy()
            data_df['is_attack'] = -1
            data_df['predicted_attack'] = 0
            data_df['reviewed'] = 0

        if data_df.empty:
            print("No data available for review. Exiting continuous improvement loop.")
            return model, all_reviewed_data

        print("Making predictions...")
        X_combined = extract_features_consistent(data_df, vectorizer, preprocessor, all_feature_names, onehot)
        data_df['predicted_attack'] = model.predict(X_combined)

        data_df['predicted_attack'] = model.predict(X_combined)

        data_df, reviewed_count = manual_attack_review(data_df, limit=review_batch_size,
                                                       min_attacks=min_attacks_per_batch)

        all_reviewed_data = pd.concat([all_reviewed_data, data_df[data_df['reviewed'] == 1]], ignore_index=True)

        data_df.to_csv(results_path, index=False)
        print(f"Results saved to: {results_path}")

        if len(all_reviewed_data) >= 100:
            print(f"Retraining model with {len(all_reviewed_data)} reviewed samples...")
            model, _, _, _, preprocessor = train_improved_model(all_reviewed_data)
            print("Model retrained.")

        if input("Do you want to continue reviewing? (y/n): ").lower() != 'y':
            break

    return model, all_reviewed_data


def manual_attack_review(results_df, limit=50, min_attacks=10):
    print("\nManual Attack Review")
    print("====================")
    print("For each entry, enter 'y' if it's an attack, 'n' if it's not, or 'q' to quit.")

    attack_reviewed_count = 0
    total_reviewed = 0

    # Separate attacks and non-attacks as predicted by the model
    predicted_attacks = results_df[results_df['predicted_attack'] == 1]
    predicted_non_attacks = results_df[results_df['predicted_attack'] == 0]

    while total_reviewed < limit or attack_reviewed_count < min_attacks:
        # Prioritize reviewing predicted attacks if we haven't met the minimum
        if attack_reviewed_count < min_attacks and not predicted_attacks.empty:
            row = predicted_attacks.iloc[0]
            predicted_attacks = predicted_attacks.iloc[1:]
        elif not predicted_non_attacks.empty:
            row = predicted_non_attacks.iloc[0]
            predicted_non_attacks = predicted_non_attacks.iloc[1:]
        else:
            break  # No more entries to review

        if row.get('reviewed', 0) == 1 or row.get('type') == 'RESPONSE':
            continue

        print(f"\nEntry {total_reviewed + 1}")
        print("-----------------------")
        for column in row.index:
            if column not in ['reviewed', 'is_attack', 'predicted_attack']:
                value = row[column]
                if pd.isna(value) or value == 'N/A':
                    value = "N/A"
                elif column == 'headers' and isinstance(value, (str, dict)):
                    if isinstance(value, str):
                        try:
                            headers = json.loads(value)
                        except json.JSONDecodeError:
                            headers = {"Error": "Invalid JSON"}
                    else:
                        headers = value
                    print(f"{column.capitalize()}:")
                    for header, header_value in headers.items():
                        print(f"  {header}: {header_value}")
                    continue
                elif column == 'body' and value is not None:
                    value = f"{str(value)[:200]}..." if len(str(value)) > 200 else value
                print(f"{column.capitalize()}: {value}")

        print(f"Model prediction: {'Attack' if row.get('predicted_attack', 0) == 1 else 'Not Attack'}")

        while True:
            user_input = input("Is this an attack? (y/n/q): ").lower()
            if user_input in ['y', 'n', 'q']:
                break
            print("Invalid input. Please enter 'y', 'n', or 'q'.")

        if user_input == 'q':
            break

        results_df.loc[row.name, 'is_attack'] = (user_input == 'y')
        results_df.loc[row.name, 'reviewed'] = 1
        total_reviewed += 1

        if row.get('predicted_attack', 0) == 1:
            attack_reviewed_count += 1

    print("\nManual review session completed.")
    print(f"Total entries reviewed: {total_reviewed}")
    print(f"Model-identified attacks reviewed: {attack_reviewed_count}")
    return results_df, total_reviewed
