import pandas as pd
from app.feature_engineering import engineer_features

def test_engineer_features():
    data = pd.DataFrame({
        'idade': [30, 40],
        'experiencia': [5, 10],
        'salario': [5000, 10000],
        'cargo': ['Dev', 'Analyst'],
        'target': [1, 0]
    })

    features, labels = engineer_features(data)

    # Testa se as features estão com a quantidade correta de linhas
    assert features.shape[0] == labels.shape[0]

    # Testa se não tem missing nas features
    assert features.isnull().sum().sum() == 0

    # Testa se o target só tem 0 ou 1
    assert set(labels.unique()).issubset({0, 1})
