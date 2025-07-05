import os
import pandas as pd
import joblib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from app.utils.constants import DATASET_PATH, FEATURES_PATH, FEATURE_PIPELINE_PATH
input_path = DATASET_PATH
output_path = FEATURES_PATH
feature_pipeline_path = FEATURE_PIPELINE_PATH

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, texto_col='cv', cat_cols=['nivel_ingles', 'area_atuacao'], max_features=300):
        self.texto_col = texto_col
        self.cat_cols = cat_cols
        self.max_features = max_features
        self.pipeline = None

    def fit(self, df):
        logger.info("Iniciando treino da engenharia de features...")

        # Tratamento preventivo de NaN
        df[self.texto_col] = df[self.texto_col].fillna("").astype(str)
        for col in self.cat_cols:
            df[col] = df[col].fillna("desconhecido").astype(str).str.lower()

        text_vectorizer = TfidfVectorizer(max_features=self.max_features)
        cat_encoder = OneHotEncoder(handle_unknown='ignore')

        self.pipeline = ColumnTransformer([
            ('tfidf', text_vectorizer, self.texto_col),
            ('cat', cat_encoder, self.cat_cols)
        ])

        self.pipeline.fit(df)

        joblib.dump(self.pipeline, feature_pipeline_path)
        logger.info(f"Pipeline de features treinada e salva com sucesso em '{feature_pipeline_path}'.")

    def transform(self, df):
        if self.pipeline is None:
            logger.info("Carregando pipeline de features já treinada...")
            self.pipeline = joblib.load(feature_pipeline_path)

        logger.info("Transformando dados com pipeline de features...")

        # Garantia extra de consistência
        df[self.texto_col] = df[self.texto_col].fillna("").astype(str)
        for col in self.cat_cols:
            df[col] = df[col].fillna("desconhecido").astype(str).str.lower()

        return self.pipeline.transform(df)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

if __name__ == "__main__":
    logger.info("Iniciando extração de features...")

    df = pd.read_csv(input_path)

    fe = FeatureEngineer()
    X = fe.fit_transform(df)

    joblib.dump(X, output_path)
    logger.info(f"Features extraídas e salvas com sucesso em '{output_path}'")
