import pandas as pd
import joblib
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Configuração de logging
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, texto_col='cv', cat_cols=['nivel_ingles', 'area_atuacao'], max_features=300):
        self.texto_col = texto_col
        self.cat_cols = cat_cols
        self.max_features = max_features
        self.pipeline = None

    def fit(self, df):
        logger.info("Iniciando treino da engenharia de features...")
        text_vectorizer = TfidfVectorizer(max_features=self.max_features)
        cat_encoder = OneHotEncoder(handle_unknown='ignore')

        self.pipeline = ColumnTransformer([
            ('tfidf', text_vectorizer, self.texto_col),
            ('cat', cat_encoder, self.cat_cols)
        ])

        self.pipeline.fit(df)
        joblib.dump(self.pipeline, 'models/feature_pipeline.pkl')
        logger.info("Pipeline de features treinada e salva com sucesso em 'models/feature_pipeline.pkl'.")

    def transform(self, df):
        if self.pipeline is None:
            logger.info("Carregando pipeline de features já treinada...")
            self.pipeline = joblib.load('models/feature_pipeline.pkl')

        logger.info("Transformando dados com pipeline de features...")
        return self.pipeline.transform(df)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

if __name__ == "__main__":
    logger.info("Iniciando extração de features...")

    df = pd.read_csv("data/dataset_processado.csv")

    df['cv'] = df['cv'].fillna("").astype(str)
    df['nivel_ingles'] = df['nivel_ingles'].fillna("Desconhecido").str.lower()
    df['area_atuacao'] = df['area_atuacao'].fillna("Indefinido").str.lower()

    fe = FeatureEngineer()
    X = fe.fit_transform(df)

    joblib.dump(X, "data/features_treinamento.pkl")
    logger.info("Features extraídas e salvas com sucesso em 'data/features_treinamento.pkl'")
