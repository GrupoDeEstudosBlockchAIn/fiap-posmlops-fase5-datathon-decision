# app/data_collector.py

import os
import pandas as pd

class DataCollector:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir

    def load_applicants_data(self):
        """
        Carrega os dados de applicants a partir de um arquivo JSON.
        """
        path = os.path.join(self.data_dir, 'applicants', 'applicants.json')
        return self._load_json_as_df(path)

    def load_prospects_data(self):
        """
        Carrega os dados de prospects a partir de um arquivo JSON.
        """
        path = os.path.join(self.data_dir, 'prospects', 'prospects.json')
        return self._load_json_as_df(path)

    def load_vagas_data(self):
        """
        Carrega os dados de vagas a partir de um arquivo JSON.
        """
        path = os.path.join(self.data_dir, 'vagas', 'vagas.json')
        return self._load_json_as_df(path)

    def _load_json_as_df(self, filepath):
        """
        Função auxiliar para carregar um JSON como DataFrame.
        """
        try:
            df = pd.read_json(filepath)
            print(f"✅ Dados carregados de: {filepath}")
            return df
        except Exception as e:
            print(f"❌ Erro ao carregar {filepath}: {e}")
            return pd.DataFrame()
