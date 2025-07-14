import pandas as pd
import re
import unidecode
from sklearn.base import BaseEstimator, TransformerMixin

from app.etl.data_collector import coletar_dados
from app.utils.constants import DATASET_PATH
from app.utils.logging_config import setup_logging

output_path = DATASET_PATH

# Configuração de logging
logger = setup_logging(__name__)

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, colunas_texto=['cv']):
        self.colunas_texto = colunas_texto

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        for col in self.colunas_texto:
            logger.info(f"Pré-processando coluna de texto: {col}")
            df[col] = df[col].fillna("").astype(str).apply(self.limpar_texto)

        df['nivel_ingles'] = df['nivel_ingles'].fillna("Desconhecido").astype(str).str.lower()
        df['area_atuacao'] = df['area_atuacao'].fillna("Indefinido").astype(str).str.lower()

        logger.info("Texto e colunas categóricas normalizados com sucesso.")
        return df

    @staticmethod
    def limpar_texto(texto):
        texto = texto.lower()
        texto = unidecode.unidecode(texto)
        texto = re.sub(r'[^a-z\s]', ' ', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        return texto

if __name__ == "__main__":
    logger.info("Iniciando pré-processamento de dados...")

    df = coletar_dados()

    if df.empty:
        logger.warning("Nenhum dado foi carregado para pré-processamento.")
    else:
        preprocessor = DataPreprocessor()
        df_limpo = preprocessor.fit_transform(df)

        df_limpo.to_csv(output_path, index=False)
        logger.info(f"Dados pré-processados e salvos com sucesso em '{output_path}'")
