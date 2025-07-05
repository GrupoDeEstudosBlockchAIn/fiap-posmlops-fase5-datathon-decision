import os
import logging
import uvicorn
import pandas as pd

from app.utils.constants import DATASET_PATH, FEATURES_PATH
from app.etl.data_preprocessing import DataPreprocessor
from app.etl.data_collector import coletar_dados
from app.model.feature_engineering import FeatureEngineer
from app.model.model_training import treinar_modelo

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verificar_e_executar_pipeline():
    # Etapa 1: Pré-processamento
    if not os.path.exists(DATASET_PATH):
        logger.warning(f"{DATASET_PATH} não encontrado. Executando pré-processamento...")

        df = coletar_dados()
        if df.empty:
            logger.error("Pipeline abortada: Nenhum dado coletado.")
            return

        preprocessor = DataPreprocessor()
        df_limpo = preprocessor.fit_transform(df)

        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        df_limpo.to_csv(DATASET_PATH, index=False)
        logger.info(f"Pré-processamento concluído. Arquivo salvo em '{DATASET_PATH}'")
    else:
        logger.info("Pré-processamento já executado. Dados encontrados.")

    # Etapa 2: Engenharia de features
    if not os.path.exists(FEATURES_PATH):
        logger.warning(f"{FEATURES_PATH} não encontrado. Executando engenharia de features...")

        df = pd.read_csv(DATASET_PATH)
        fe = FeatureEngineer()
        X = fe.fit_transform(df)

        os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
        import joblib
        joblib.dump(X, FEATURES_PATH)
        logger.info(f"Engenharia de features concluída. Arquivo salvo em '{FEATURES_PATH}'")
    else:
        logger.info("Engenharia de features já executada. Arquivo encontrado.")

    # Etapa 3: Treinamento de modelo
    logger.info("Executando etapa de treinamento de modelo...")
    df = pd.read_csv(DATASET_PATH)
    df['match'] = df.get('match', 0)  # garante a coluna para o modelo supervisionado
    treinar_modelo(df)

if __name__ == "__main__":
    logger.info("Inicializando pipeline do Decision AI (modo direto)...")
    verificar_e_executar_pipeline()
    logger.info("Iniciando servidor FastAPI...")
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000)
