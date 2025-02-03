from sklearn.metrics import classification_report, confusion_matrix


def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        if name in ['Isolation Forest', 'One-Class SVM']:
            y_pred = (model.predict(X_test) == -1).astype(int)
        else:
            y_pred = model.predict(X_test)

        print(f"\n{name} Evaluation:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))