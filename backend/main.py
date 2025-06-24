# main.py
import os
import subprocess
import uvicorn
import sys

# Caminhos esperados
MODEL_PATH = "models/model.pkl"
FEATURES_PATH = "models/feature_pipeline.pkl"
DATASET_PATH = "data/dataset_processado.csv"

PYTHON_EXECUTABLE = sys.executable

def run_script(script_path):
    print(f"Executando: {script_path}")
    result = subprocess.run(
        [PYTHON_EXECUTABLE, script_path],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.getcwd()}
    )

    if result.returncode != 0:
        print(f"Erro ao executar {script_path}:\n{result.stderr}")
        exit(1)
    else:
        print(f"{script_path} executado com sucesso.\n")

def verificar_e_executar_pipeline():
    if not os.path.exists(DATASET_PATH):
        run_script("app/data_preprocessing.py")
    
    if not os.path.exists(FEATURES_PATH):
        run_script("app/feature_engineering.py")
    
    if not os.path.exists(MODEL_PATH):
        run_script("app/model_training.py")

if __name__ == "__main__":
    print("Inicializando pipeline do Decision AI...")

    # Verifica e executa etapas se necessário
    verificar_e_executar_pipeline()

    # Sobe o servidor FastAPI
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
