import pandas as pd
from app.model.model_training import treinar_modelo

def test_model_training():
    # Criando dados fake com pelo menos 10 amostras balanceadas
    df = pd.DataFrame({
        'cv': [
            'Python backend', 'Engenheiro de dados', 'Frontend Vue.js',
            'Analista de dados', 'DevOps AWS', 'Cientista de Dados',
            'Tester QA', 'Engenheiro ML', 'Sysadmin Linux', 'Arquiteto de Soluções'
        ],
        'nivel_ingles': ['avançado', 'básico', 'intermediário', 'básico', 'avançado',
                         'avançado', 'básico', 'intermediário', 'básico', 'avançado'],
        'area_atuacao': ['TI', 'Dados', 'TI', 'Dados', 'Infraestrutura',
                         'Dados', 'TI', 'IA', 'Infraestrutura', 'TI'],
        'match': [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    })

    # Chamando a função com os dados fake
    treinar_modelo(df)
