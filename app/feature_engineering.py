# app/feature_engineering.py

import pandas as pd

def engineer_features(df: pd.DataFrame):
    """
    Realiza a engenharia de features a partir do DataFrame pré-processado.
    Exemplo genérico: cria variáveis dummies e extrai a coluna target.
    """
    # 🔹 Definindo a coluna alvo (exemplo genérico: 'contratado' como 1 ou 0)
    if 'contratado' in df.columns:
        labels = df['contratado']
    else:
        labels = pd.Series([0] * len(df))  # Placeholder

    # 🔹 Selecionando algumas features numéricas
    features = df[['idade', 'experiencia', 'salario']].copy()

    # 🔹 Transformação de variáveis categóricas em dummies (exemplo)
    if 'cargo' in df.columns:
        dummies = pd.get_dummies(df['cargo'], prefix='cargo')
        features = pd.concat([features, dummies], axis=1)

    # 🔹 Lidando com nulos (simples)
    features.fillna(0, inplace=True)

    print("✅ Engenharia de features concluída!")
    return features, labels
