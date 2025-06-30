import json
import pandas as pd
import logging

# Configuração de logging
logger = logging.getLogger(__name__)

APPLICANTS_PATH = 'data/applicants/applicants.json'
PROSPECTS_PATH = 'data/prospects/prospects.json'

def construir_dataframe_supervisionado(): 
    logger.info("Iniciando construção do dataset supervisionado...")

    try:
        with open(APPLICANTS_PATH, 'r', encoding='utf-8') as f:
            applicants = json.load(f)
        logger.info(f"Carregado: {APPLICANTS_PATH}")

        with open(PROSPECTS_PATH, 'r', encoding='utf-8') as f:
            prospects = json.load(f)
        logger.info(f"Carregado: {PROSPECTS_PATH}")
    except Exception as e:
        logger.exception("Erro ao carregar arquivos JSON")
        raise e

    contratados = set()
    for vaga in prospects.values():
        for prospect in vaga.get('prospects', []):
            if prospect.get('situacao_candidado', '').lower() == 'contratado pela decision':
                contratados.add(prospect['codigo'])

    dados = []
    ignorados = 0

    for codigo, dados_app in applicants.items():
        try:
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
        except Exception as e:
            ignorados += 1
            logger.warning(f"Candidato ignorado por erro de leitura (código: {codigo}): {str(e)}")

    logger.info(f"Dataset supervisionado construído: {len(dados)} candidatos | Ignorados: {ignorados}")
    return pd.DataFrame(dados)
