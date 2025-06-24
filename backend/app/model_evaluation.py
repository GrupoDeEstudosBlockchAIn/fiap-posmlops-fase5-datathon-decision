import joblib
import pandas as pd
from app.feature_engineering import FeatureEngineer

# Caminhos dos arquivos salvos
MODEL_PATH = 'models/model.pkl'
PIPELINE_PATH = 'models/feature_pipeline.pkl'

# Carrega modelo treinado
model = joblib.load(MODEL_PATH)
print("Modelo carregado com sucesso.")

# Carrega pipeline de engenharia de features já treinado
fe = FeatureEngineer()
# NÃO executa fit ou fit_transform — apenas garante que self.pipeline seja carregado
fe.pipeline = joblib.load(PIPELINE_PATH)

def prever_match(dados_candidato: dict) -> dict:
    """
    Recebe um dicionário com dados do candidato e retorna o score de match.
    """
    df = pd.DataFrame([dados_candidato])

    # Preenche valores ausentes e normaliza strings (repete o tratamento feito no fit)
    df['cv'] = df['cv'].fillna("").astype(str)
    df['nivel_ingles'] = df['nivel_ingles'].fillna("Desconhecido").str.lower()
    df['area_atuacao'] = df['area_atuacao'].fillna("Indefinido").str.lower()

    X = fe.transform(df)
    proba = model.predict_proba(X)[0][1]  # Probabilidade da classe 1 (match)
    match = bool(proba > 0.5)

    return {
        "match": match,
        "score": round(proba, 2),
        "perfil_recomendado": "Match Técnico" if match else "Não compatível"
    }

# Exemplo de uso local (remover ou comentar em produção)
if __name__ == '__main__':
    exemplo = {
        "cv": "Desenvolvedor Python com experiência em APIs e bancos de dados.",
        "nivel_ingles": "Avançado",
        "area_atuacao": "TI"
    }
    resultado = prever_match(exemplo)
    print("Resultado da Inferência:", resultado)
