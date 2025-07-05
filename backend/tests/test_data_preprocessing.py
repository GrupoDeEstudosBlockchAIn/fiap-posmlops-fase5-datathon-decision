import pandas as pd
from app.etl.data_preprocessing import DataPreprocessor

def test_preprocess_data():
    df = pd.DataFrame({
        'cv': ['Sou desenvolvedor', None],
        'nivel_ingles': ['Avançado', None],
        'area_atuacao': ['TI', None]
    })

    processor = DataPreprocessor()
    result = processor.fit_transform(df)

    assert isinstance(result, pd.DataFrame)
    assert result.isnull().sum().sum() == 0
