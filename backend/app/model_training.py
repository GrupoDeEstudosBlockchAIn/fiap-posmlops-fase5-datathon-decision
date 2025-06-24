import json
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from feature_engineering import FeatureEngineer  # Corrigido: sem 'app.'

# Caminhos das bases
APPLICANTS_PATH = 'data/applicants/applicants.json'
PROSPECTS_PATH = 'data/prospects/prospects.json'

# Caminho de salvamento
FEATURES_PATH = 'data/features_treinamento.pkl'
MODEL_PATH = 'models/model.pkl'

# Função auxiliar para identificar se o candidato foi contratado
def construir_dataframe_supervisionado():
    with open(APPLICANTS_PATH, 'r', encoding='utf-8') as f:
        applicants = json.load(f)

    with open(PROSPECTS_PATH, 'r', encoding='utf-8') as f:
        prospects = json.load(f)

    # Mapeia códigos de candidatos contratados
    contratados = set()
    for vaga in prospects.values():
        for prospect in vaga.get('prospects', []):
            if prospect.get('situacao_candidado', '').lower() == 'contratado pela decision':
                contratados.add(prospect['codigo'])

    dados = []
    for codigo, dados_app in applicants.items():
        nome = dados_app.get('infos_basicas', {}).get('nome', '')
        cv = dados_app.get('cv_pt', '')
        nivel_ingles = dados_app.get('formacao_e_idiomas', {}).get('nivel_ingles', '')
        area_atuacao = dados_app.get('informacoes_profissionais', {}).get('area_atuacao', '')

        match = 1 if codigo in contratados else 0

        dados.append({
            'codigo': codigo,
            'nome': nome,
            'cv': cv,
            'nivel_ingles': nivel_ingles,
            'area_atuacao': area_atuacao,
            'match': match
        })

    return pd.DataFrame(dados)


def treinar_modelo():

    print("Iniciando treinamento do modelo...")

    df = construir_dataframe_supervisionado()

    print("Dados carregados com", len(df), "candidatos")

    X = df[['cv', 'nivel_ingles', 'area_atuacao']]
    y = df['match']

    # Engenharia de features (corrigido: nomes dos parâmetros)
    fe = FeatureEngineer(texto_col='cv', cat_cols=['nivel_ingles', 'area_atuacao'])
    X_transformed = fe.fit_transform(X)

    joblib.dump(X_transformed, FEATURES_PATH)
    print("Features salvas em", FEATURES_PATH)

    # Treinamento do modelo
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Avaliação do Modelo:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print("Modelo salvo em", MODEL_PATH)


if __name__ == '__main__':
    treinar_modelo()
