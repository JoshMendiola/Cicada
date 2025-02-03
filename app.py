import logging
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix

from src.continuous.improvement_loop import continuous_improvement_loop
from src.data.data_loader import load_json_logs
from src.models.train import train_improved_model
from utils.feature_alignment import extract_features_consistent

PROJECT_ROOT = os.getcwd()


def train_new_model(logs_path, model_dir):
    """Train a new model from scratch"""
    initial_logs_df = load_json_logs(logs_path)
    print("Training initial model...")
    model, vectorizer, preprocessor, all_feature_names, onehot = train_improved_model(
        initial_logs_df,
        max_features=48
    )
    save_model(model, vectorizer, preprocessor, all_feature_names, model_dir)
    return model, vectorizer, preprocessor, all_feature_names, onehot, initial_logs_df


def save_model(model, vectorizer, preprocessor, all_feature_names, model_dir):
    """Save model and its components"""
    print("\nSaving model and components...")
    model_info = {
        'model': model,
        'feature_names': all_feature_names,
        'n_features': len(all_feature_names),
        'vectorizer_max_features': vectorizer.max_features
    }
    joblib.dump(model_info, os.path.join(model_dir, 'model_info.joblib'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.joblib'))
    joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.joblib'))
    print("Model saved successfully.")


def evaluate_model(model, data, vectorizer, preprocessor, all_feature_names, onehot):
    """Evaluate model performance"""
    print("\nEvaluating model...")
    X_combined = extract_features_consistent(data, vectorizer, preprocessor, all_feature_names, onehot)
    logging.debug(f"Evaluation data shape: {X_combined.shape}")
    y = data['is_attack'].astype(int)
    y_pred = model.predict(X_combined)
    print(classification_report(y, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    return y_pred


def main():
    # Setup paths
    logs_path = os.path.join(PROJECT_ROOT, 'data', 'logs.json')
    results_path = os.path.join(PROJECT_ROOT, 'updated_log_analysis_results.csv')
    model_dir = os.path.join(PROJECT_ROOT, 'model')
    os.makedirs(model_dir, exist_ok=True)

    # Main menu
    while True:
        print("\nCicada Model Management")
        print("1. Train new model")
        print("2. Load and evaluate existing model")
        print("3. Start continuous improvement loop")
        print("4. Exit")

        choice = input("Select an option (1-4): ")

        if choice == '1':
            model, vectorizer, preprocessor, all_feature_names, onehot, initial_logs_df = train_new_model(logs_path,
                                                                                                          model_dir)
            evaluate_model(model, initial_logs_df, vectorizer, preprocessor, all_feature_names, onehot)

        elif choice == '2':
            try:
                # Load existing model
                model_info = joblib.load(os.path.join(model_dir, 'model_info.joblib'))
                vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.joblib'))
                preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.joblib'))
                initial_logs_df = load_json_logs(logs_path)
                evaluate_model(model_info['model'], initial_logs_df, vectorizer, preprocessor,
                               model_info['feature_names'], None)  # onehot handled by preprocessor
            except FileNotFoundError:
                print("No existing model found. Please train a new model first.")

        elif choice == '3':
            try:
                # Load existing model for improvement
                model_info = joblib.load(os.path.join(model_dir, 'model_info.joblib'))
                vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.joblib'))
                preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.joblib'))
                initial_logs_df = load_json_logs(logs_path)

                final_model, final_results = continuous_improvement_loop(
                    model_info['model'], vectorizer, preprocessor,
                    model_info['feature_names'], None, initial_logs_df, results_path
                )
                save_model(final_model, vectorizer, preprocessor, model_info['feature_names'], model_dir)
                evaluate_model(final_model, final_results, vectorizer, preprocessor,
                               model_info['feature_names'], None)
            except FileNotFoundError:
                print("No existing model found. Please train a new model first.")

        elif choice == '4':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please select 1-4.")


if __name__ == "__main__":
    main()
