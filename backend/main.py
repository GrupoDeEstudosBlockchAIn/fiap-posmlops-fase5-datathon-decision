import os
import subprocess
import uvicorn
import sys
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/model.pkl"
FEATURES_PATH = "models/feature_pipeline.pkl"
DATASET_PATH = "data/dataset_processado.csv"

PYTHON_EXECUTABLE = sys.executable

def run_script(script_path):
    logger.info(f"Executando script: {script_path}")
    result = subprocess.run(
        [PYTHON_EXECUTABLE, script_path],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.getcwd()}
    )

    if result.returncode != 0:
        logger.error(f"Erro ao executar {script_path}:\n{result.stderr}")
        exit(1)
    else:
        logger.info(f"{script_path} executado com sucesso.")

def verificar_e_executar_pipeline():
    if not os.path.exists(DATASET_PATH):
        logger.warning(f"{DATASET_PATH} não encontrado. Executando pré-processamento.")
        run_script("app/data_preprocessing.py")
    
    if not os.path.exists(FEATURES_PATH):
        logger.warning(f"{FEATURES_PATH} não encontrado. Executando engenharia de features.")
        run_script("app/feature_engineering.py")
    
    run_script("app/model_training.py")

if __name__ == "__main__":
    logger.info("Inicializando pipeline do Decision AI...")
    verificar_e_executar_pipeline()
    logger.info("Iniciando servidor FastAPI...")
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000)
