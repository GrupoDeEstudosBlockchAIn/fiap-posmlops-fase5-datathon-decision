# app/model_training.py

from sklearn.ensemble import RandomForestClassifier

def train_model(features, labels):
    """
    Treina um modelo RandomForest com hiperparâmetros padrão.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, labels)
    print("✅ Modelo treinado com sucesso!")
    return model
