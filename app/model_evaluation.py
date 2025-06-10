# app/model_evaluation.py

from sklearn.metrics import classification_report

def evaluate_model(model, features, labels):
    """
    Avalia o modelo com um relatório de classificação.
    """
    predictions = model.predict(features)
    report = classification_report(labels, predictions, output_dict=False)
    print("✅ Avaliação concluída!")
    return report
