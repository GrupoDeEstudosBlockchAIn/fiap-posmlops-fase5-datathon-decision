import pandas as pd
import joblib
import logging
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from app.model.feature_engineering import FeatureEngineer
from app.report.metric_report import gerar_metric_report
from app.utils.model_utils import construir_dataframe_supervisionado
from app.utils.constants import FEATURES_PATH, MODEL_PATH
from app.utils.logging_config import setup_logging

logger = setup_logging(__name__)

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

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    logger.info(f"Split dos dados: {X_train.shape[0]} treino | {X_test.shape[0]} teste")
    logger.info(f"Distribuição das classes no treino:\n{y_train.value_counts()}")

    # SMOTE se possível
    if y_train.nunique() < 2:
        logger.warning("A variável target possui apenas uma classe. SMOTE ignorado.")
        X_train_resampled = X_train
        y_train_resampled = y_train
    else:
        logger.info("Aplicando SMOTE para balanceamento de classes...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        logger.info(f"SMOTE aplicado. Total: {len(X_train_resampled)} registros")

    # Calcula base_score
    base_score = y_train_resampled.mean()
    if not 0 < base_score < 1:
        base_score = 0.5
        logger.warning(f"Base score inválido. Valor ajustado para {base_score}")

    # Treinamento
    logger.info("Treinando modelo XGBoost...")
    model = XGBClassifier(
        eval_metric='logloss',
        n_estimators=100,
        random_state=42,
        scale_pos_weight=1.0,
        base_score=base_score
    )
    model.fit(X_train_resampled, y_train_resampled)
    logger.info("Modelo treinado com sucesso.")

    # Avaliação
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.4).astype(int)
    logger.info("Classification report:\n" + classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    logger.info(f"Modelo salvo em '{MODEL_PATH}'")

    gerar_metric_report()
    logger.info("Relatório de métricas gerado.")

if __name__ == '__main__':
    treinar_modelo()
