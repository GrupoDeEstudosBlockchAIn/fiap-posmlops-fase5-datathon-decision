import pandas as pd
from app.feature_engineering import FeatureEngineer

def test_feature_engineer():
    df = pd.DataFrame({
        'cv': ['Desenvolvedor backend', 'Analista de dados'],
        'nivel_ingles': ['avançado', 'intermediário'],
        'area_atuacao': ['TI', 'Dados']
    })

    fe = FeatureEngineer()
    X = fe.fit_transform(df)

    assert X.shape[0] == 2
    assert X.shape[1] > 0
