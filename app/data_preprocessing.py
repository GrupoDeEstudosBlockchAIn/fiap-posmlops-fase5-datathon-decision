# app/data_preprocessing.py

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess_data(applicants_df: pd.DataFrame,
                    prospects_df: pd.DataFrame,
                    vagas_df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza o pré-processamento dos dados para o modelo de recrutamento Decision.
    Inclui validação de schema, remoção de duplicatas, tratamento de nulos,
    padronização de tipos e merges.
    """

    # ✅ Validação de Schema (verificar se colunas obrigatórias existem)
    expected_applicants_cols = ['id', 'nome', 'idade', 'experiencia', 'vaga_id', 'prospect_id', 'avaliacao_entrevista', 'engajamento', 'fit_cultural']
    expected_prospects_cols = ['id', 'status']
    expected_vagas_cols = ['id', 'cargo', 'salario']

    for df, name, expected_cols in zip(
        [applicants_df, prospects_df, vagas_df],
        ['Applicants', 'Prospects', 'Vagas'],
        [expected_applicants_cols, expected_prospects_cols, expected_vagas_cols]
    ):
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            logging.warning(f"[{name}] - Colunas faltando: {missing_cols}")

    # ✅ Remover duplicatas
    applicants_df = applicants_df.drop_duplicates()
    prospects_df = prospects_df.drop_duplicates()
    vagas_df = vagas_df.drop_duplicates()

    # ✅ Conversão de tipos
    numeric_cols = ['idade', 'experiencia', 'salario', 'avaliacao_entrevista', 'engajamento', 'fit_cultural']
    for col in numeric_cols:
        for df in [applicants_df, vagas_df]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # ✅ Tratamento de Nulos
    applicants_df['idade'].fillna(applicants_df['idade'].median(), inplace=True)
    applicants_df['experiencia'].fillna(0, inplace=True)
    applicants_df['avaliacao_entrevista'].fillna(0, inplace=True)
    applicants_df['engajamento'].fillna(0, inplace=True)
    applicants_df['fit_cultural'].fillna(0, inplace=True)

    vagas_df['salario'].fillna(vagas_df['salario'].median(), inplace=True)
    prospects_df['status'].fillna('não informado', inplace=True)

    # ✅ Merges
    merged_df = applicants_df.merge(vagas_df, left_on='vaga_id', right_on='id', how='left', suffixes=('', '_vaga'))
    merged_df = merged_df.merge(prospects_df, left_on='prospect_id', right_on='id', how='left', suffixes=('', '_prospect'))

    logging.info("✅ Pré-processamento concluído com sucesso.")
    return merged_df
