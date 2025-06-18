# app/model_evaluation.py

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

def evaluate_model(model, test_data):
    """
    Avalia o modelo MLP com várias métricas.
    Assume que os dados de teste já estão pré-processados e escalados.
    """
    X_test, y_test = test_data

    # Obter probabilidades
    y_proba = model.predict(X_test).ravel()

    # Converter para classes (0 ou 1)
    predictions = (y_proba > 0.5).astype("int32")

    print("✅ Avaliação concluída!")
    print(classification_report(y_test, predictions))
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, predictions))
    print(f"AUC: {roc_auc_score(y_test, y_proba)}")

    return predictions, y_proba
