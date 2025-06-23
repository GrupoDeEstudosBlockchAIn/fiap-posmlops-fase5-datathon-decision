import joblib
import pandas as pd
from app.feature_engineering import FeatureEngineer

# Caminhos dos arquivos salvos
MODEL_PATH = 'models/model.pkl'
FEATURES_PATH = 'data/features_treinamento.pkl'

# Carrega objetos
model = joblib.load(MODEL_PATH)
print("✅ Modelo carregado com sucesso.")

# Para garantir compatibilidade de vetorização
X_train = joblib.load(FEATURES_PATH)
fe = FeatureEngineer()
fe.fit_transform(pd.DataFrame(columns=['cv', 'nivel_ingles', 'area_atuacao']))  # Dummy fit
fe.vectorizer = X_train.vectorizer
fe.ohe = X_train.ohe

def prever_match(dados_candidato: dict) -> dict:
    """
    Recebe um dicionário com dados do candidato e retorna o score de match.
    """
    df = pd.DataFrame([dados_candidato])
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
    print("\n🔍 Resultado da Inferência:", resultado)
