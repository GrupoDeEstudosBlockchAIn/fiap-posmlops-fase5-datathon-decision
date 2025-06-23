import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

class FeatureEngineer:
    def __init__(self, texto_col='cv', cat_cols=['nivel_ingles', 'area_atuacao'], max_features=300):
        self.texto_col = texto_col
        self.cat_cols = cat_cols
        self.max_features = max_features
        self.pipeline = None

    def fit(self, df):
        text_vectorizer = TfidfVectorizer(max_features=self.max_features)
        cat_encoder = OneHotEncoder(handle_unknown='ignore')

        self.pipeline = ColumnTransformer([
            ('tfidf', text_vectorizer, self.texto_col),
            ('cat', cat_encoder, self.cat_cols)
        ])

        self.pipeline.fit(df)
        joblib.dump(self.pipeline, 'models/feature_pipeline.pkl')
        print("✅ Engenharia de features treinada e salva com sucesso!")

    def transform(self, df):
        if self.pipeline is None:
            self.pipeline = joblib.load('models/feature_pipeline.pkl')

        return self.pipeline.transform(df)

if __name__ == "__main__":
    print("🔄 Iniciando extração de features...")
    df = pd.read_csv("data/dataset_processado.csv")

    # Prevenção extra: remove NaNs da coluna 'cv'
    df['cv'] = df['cv'].fillna("").astype(str)

    # (opcional) também garante que colunas categóricas não tenham NaN
    df['nivel_ingles'] = df['nivel_ingles'].fillna("Desconhecido").str.lower()
    df['area_atuacao'] = df['area_atuacao'].fillna("Indefinido").str.lower()

    fe = FeatureEngineer()
    fe.fit(df)
    X = fe.transform(df)

    joblib.dump(X, "data/features_treinamento.pkl")
    print("✅ Features extraídas e salvas com sucesso!")
