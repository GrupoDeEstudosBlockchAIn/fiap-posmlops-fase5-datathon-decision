import pandas as pd
import joblib
import logging
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from app.feature_engineering import FeatureEngineer
from app.metric_report import gerar_metric_report
from app.model_utils import construir_dataframe_supervisionado

# Configuração de logging
logger = logging.getLogger(__name__)

# Caminhos dos arquivos
FEATURES_PATH = 'data/features_treinamento.pkl'
MODEL_PATH = 'models/model.pkl'

def treinar_modelo(df=None):
    logger.info("Iniciando processo de treinamento do modelo...")

    if df is None:
        df = construir_dataframe_supervisionado()
        logger.info(f"Dados carregados: {len(df)} candidatos")
    else:
        logger.info(f"Usando DataFrame fornecido com {len(df)} registros")

    X = df[['cv', 'nivel_ingles', 'area_atuacao']]
    y = df['match']

    # Engenharia de features
    logger.info("Executando engenharia de features...")
    fe = FeatureEngineer(texto_col='cv', cat_cols=['nivel_ingles', 'area_atuacao'])
    X_transformed = fe.fit_transform(X)
    joblib.dump(X_transformed, FEATURES_PATH)
    logger.info(f"Features salvas em '{FEATURES_PATH}'")

    # Separação dos dados
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    logger.info(f"Split dos dados: {X_train.shape[0]} treino | {X_test.shape[0]} teste")

    # SMOTE
    logger.info("Aplicando SMOTE para balanceamento de classes...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"SMOTE aplicado. Dados balanceados: {X_train_resampled.shape[0]} registros")

    # Cálculo de peso para classe minoritária
    weight = (len(y_train_resampled) - sum(y_train_resampled)) / sum(y_train_resampled)
    logger.info(f"Peso para classe positiva (scale_pos_weight): {weight:.2f}")

    # Modelo
    logger.info("Treinando modelo XGBoost...")
    model = XGBClassifier(
        eval_metric='logloss',
        n_estimators=100,
        random_state=42,
        scale_pos_weight=weight
    )
    model.fit(X_train_resampled, y_train_resampled)
    logger.info("Modelo treinado com sucesso.")

    # Avaliação
    logger.info("Avaliando modelo...")
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = 0.4
    y_pred = (y_proba >= threshold).astype(int)
    logger.info("Classification report:\n" + classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    logger.info(f"Modelo salvo em '{MODEL_PATH}'")

    gerar_metric_report()
    logger.info("Relatório de métricas gerado.")

if __name__ == '__main__':
    treinar_modelo()
