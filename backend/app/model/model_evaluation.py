import joblib
import pandas as pd
from app.model.feature_engineering import FeatureEngineer
from app.utils.constants import FEATURE_PIPELINE_PATH, MODEL_PATH
from app.utils.logging_config import setup_logging

model_path = MODEL_PATH
feature_pipeline_path = FEATURE_PIPELINE_PATH

# Configuração de logging
logger = setup_logging(__name__)

# Carrega modelo
logger.info("Carregando modelo e pipeline de features...")
model = joblib.load(model_path)
logger.info("Modelo carregado com sucesso.")

fe = FeatureEngineer()
fe.pipeline = joblib.load(feature_pipeline_path)
logger.info("Pipeline de engenharia de features carregado com sucesso.")

def prever_match(dados_candidato: dict) -> dict:
    logger.info("Realizando predição para candidato...")
    df = pd.DataFrame([dados_candidato])

    # Limpeza
    df['cv'] = df['cv'].fillna("").astype(str)
    df['nivel_ingles'] = df['nivel_ingles'].fillna("Desconhecido").str.lower()
    df['area_atuacao'] = df['area_atuacao'].fillna("Indefinido").str.lower()

    X = fe.transform(df)
    proba = model.predict_proba(X)[0][1]
    match = bool(proba > 0.5)

    logger.info(f"Resultado: Score={proba:.4f} | Match={match}")
    return {
        "match": match,
        "score": round(proba, 2),
        "perfil_recomendado": "Match Técnico" if match else "Não compatível"
    }

# Teste local
if __name__ == '__main__':
    exemplo = {
        "cv": "Desenvolvedor Python com experiência em APIs e bancos de dados.",
        "nivel_ingles": "Avançado",
        "area_atuacao": "TI"
    }
    resultado = prever_match(exemplo)
    logger.info(f"Resultado da Inferência: {resultado}")
