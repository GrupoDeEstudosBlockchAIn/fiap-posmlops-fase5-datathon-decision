# app/feature_engineering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def engineer_features(df: pd.DataFrame):
    """
    Realiza engenharia de features:
    - Seleção de features numéricas e categóricas
    - Escalonamento
    - Encoding
    - Validação de target
    """

    # ✅ Verificar presença do target
    if 'contratado' not in df.columns:
        raise ValueError("❌ Target 'contratado' não encontrado no DataFrame!")

    labels = df['contratado']

    # ✅ Selecionar features relevantes
    numeric_features = ['idade', 'experiencia', 'salario', 'avaliacao_entrevista', 'engajamento', 'fit_cultural']
    categorical_features = ['cargo', 'status']

    # ✅ Tratamento de valores ausentes nas features
    df[numeric_features] = df[numeric_features].fillna(0)
    df[categorical_features] = df[categorical_features].fillna('desconhecido')

    # ✅ Construção de Pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # ✅ Aplicar transformações
    X_transformed = preprocessor.fit_transform(df)

    # ✅ Salvar pipeline para uso futuro (ex: na API)
    joblib.dump(preprocessor, 'app/preprocessor.joblib')

    # ✅ Salvar features transformadas para debug
    transformed_df = pd.DataFrame(X_transformed)
    transformed_df.to_csv('app/features_transformed.csv', index=False)

    logging.info("✅ Engenharia de features finalizada e pipeline salvo.")

    return X_transformed, labels
