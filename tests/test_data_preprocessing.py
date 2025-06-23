import pandas as pd
from app.data_preprocessing import preprocess_data

def test_preprocess_data():
    # Criar DataFrames fictícios de exemplo
    applicants_df = pd.DataFrame({'id': [1, 2], 'idade': [25, None], 'experiencia': [3, 5]})
    prospects_df = pd.DataFrame({'id': [1, 2], 'vaga_id': [10, 20]})
    vagas_df = pd.DataFrame({'vaga_id': [10, 20], 'cargo': ['Dev', 'Analyst']})

    result = preprocess_data(applicants_df, prospects_df, vagas_df)

    # Testa se retornou um DataFrame
    assert isinstance(result, pd.DataFrame)

    # Testa se removeu nulos
    assert result.isnull().sum().sum() == 0
