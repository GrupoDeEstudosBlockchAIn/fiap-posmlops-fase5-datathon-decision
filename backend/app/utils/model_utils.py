import json
import pandas as pd
import logging
from app.etl.backblaze_loader import download_json_from_backblaze
from app.utils.constants import CONST_DATABASE_APPLICANTS, CONST_DATABASE_PROSPECTS

# Configuração de logging
logger = logging.getLogger(__name__)

def construir_dataframe_supervisionado(): 
    logger.info("Iniciando construção do dataset supervisionado...")

    try:
        applicants = download_json_from_backblaze(CONST_DATABASE_APPLICANTS)
        logger.info(f"Carregado: {CONST_DATABASE_APPLICANTS}")

        prospects = download_json_from_backblaze(CONST_DATABASE_PROSPECTS)
        logger.info(f"Carregado: {CONST_DATABASE_PROSPECTS}")

    except Exception as e:
        logger.exception("Erro ao carregar arquivos JSON")
        raise e

    contratados = set()
    for vaga in prospects.values():
        for prospect in vaga.get('prospects', []):
            situacao = prospect.get('situacao_candidado', '').strip().lower()
            if 'contratado' in situacao:
                contratados.add(prospect['codigo'])

    logger.info(f"Total de candidatos contratados identificados: {len(contratados)}")

    dados = []
    ignorados = 0

    for codigo, dados_app in applicants.items():
        try:
            nome = dados_app.get('infos_basicas', {}).get('nome', '')
            cv = dados_app.get('cv_pt', '')
            nivel_ingles = dados_app.get('formacao_e_idiomas', {}).get('nivel_ingles', '')
            area_atuacao = dados_app.get('informacoes_profissionais', {}).get('area_atuacao', '')

            if not nome or not cv:
                ignorados += 1
                logger.warning(f"Ignorando candidato sem nome ou CV (código: {codigo})")
                continue

            match = 1 if codigo in contratados else 0

            dados.append({
                'codigo': codigo,
                'nome': nome,
                'cv': cv,
                'nivel_ingles': nivel_ingles,
                'area_atuacao': area_atuacao,
                'match': match
            })
        except Exception as e:
            ignorados += 1
            logger.warning(f"Candidato ignorado por erro de leitura (código: {codigo}): {str(e)}")

    df = pd.DataFrame(dados)
    logger.info(f"Dataset supervisionado construído: {len(df)} candidatos | Ignorados: {ignorados}")

    # Verificação da diversidade da variável target
    match_counts = df['match'].value_counts().to_dict()
    logger.info(f"Distribuição da coluna 'match': {match_counts}")

    # Caso nenhuma classe positiva, adiciona exemplos sintéticos
    if match_counts.get(1, 0) == 0:
        logger.warning("Nenhum candidato contratado encontrado. Adicionando exemplos sintéticos para classe positiva.")
        exemplos_sinteticos = pd.DataFrame([
            {
                'codigo': 'sintetico1',
                'nome': 'Exemplo Sintético 1',
                'cv': 'Engenheiro de dados com experiência em pipelines ETL e Spark',
                'nivel_ingles': 'avançado',
                'area_atuacao': 'dados',
                'match': 1
            },
            {
                'codigo': 'sintetico2',
                'nome': 'Exemplo Sintético 2',
                'cv': 'Cientista de dados com experiência em modelagem estatística e Python',
                'nivel_ingles': 'intermediário',
                'area_atuacao': 'ciência de dados',
                'match': 1
            }
        ])
        df = pd.concat([df, exemplos_sinteticos], ignore_index=True)

    return df
