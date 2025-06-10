# app/data_preprocessing.py

import pandas as pd

def preprocess_data(applicants_df: pd.DataFrame,
                     prospects_df: pd.DataFrame,
                     vagas_df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza o pré-processamento básico dos dados:
    - Remove duplicatas
    - Trata valores nulos
    - Faz merges necessários (exemplo genérico)
    """

    # 🔹 Removendo duplicatas
    applicants_df = applicants_df.drop_duplicates()
    prospects_df = prospects_df.drop_duplicates()
    vagas_df = vagas_df.drop_duplicates()

    # 🔹 Preenchendo valores nulos com valores padrão (exemplo genérico)
    applicants_df.fillna({'nome': 'Desconhecido', 'idade': applicants_df['idade'].median()}, inplace=True)
    prospects_df.fillna({'status': 'não informado'}, inplace=True)
    vagas_df.fillna({'salario': vagas_df['salario'].median()}, inplace=True)

    # 🔹 Exemplo de merge: vinculando applicants a vagas
    merged_df = applicants_df.merge(vagas_df, left_on='vaga_id', right_on='id', how='left')

    # 🔹 Exemplo de merge com prospects (ajuste conforme dados reais)
    merged_df = merged_df.merge(prospects_df, left_on='prospect_id', right_on='id', how='left')

    print("✅ Pré-processamento concluído!")
    return merged_df
