import pandas as pd
import re
import unidecode
from sklearn.base import BaseEstimator, TransformerMixin
import logging

# Configuração de logging
logger = logging.getLogger(__name__)

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, colunas_texto=['cv']):
        self.colunas_texto = colunas_texto

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        for col in self.colunas_texto:
            logger.info(f"Pré-processando coluna de texto: {col}")
            df[col] = df[col].fillna("").apply(self.limpar_texto)

        df['nivel_ingles'] = df['nivel_ingles'].fillna("Desconhecido").str.lower()
        df['area_atuacao'] = df['area_atuacao'].fillna("Indefinido").str.lower()

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
    from data_collector import coletar_dados

    logger.info("Iniciando pré-processamento de dados...")

    df = coletar_dados(
        "data/applicants/applicants.json",
        "data/prospects/prospects.json",
        "data/vagas/vagas.json"
    )

    if df.empty:
        logger.warning("Nenhum dado foi carregado para pré-processamento.")
    else:
        preprocessor = DataPreprocessor()
        df_limpo = preprocessor.fit_transform(df)

        df_limpo.to_csv("data/dataset_processado.csv", index=False)
        logger.info("Dados pré-processados e salvos com sucesso em 'data/dataset_processado.csv'")
