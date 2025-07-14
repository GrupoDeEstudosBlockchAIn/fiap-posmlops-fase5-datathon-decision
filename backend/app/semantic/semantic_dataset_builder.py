import json
import logging
import pandas as pd
from app.semantic.semantic_matcher import avaliar_match
from app.utils.logging_config import setup_logging

# Configuração de logging
logger = setup_logging(__name__)

def carregar_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.exception(f"Erro ao carregar arquivo JSON: {path}")
        raise e

def construir_dataset_semantico(applicants_path, prospects_path, vagas_path, output_path='data/dataset_semantico.csv'):
    logger.info("Construindo dataset supervisionado com embeddings semânticos...")

    applicants = carregar_json(applicants_path)
    prospects = carregar_json(prospects_path)
    vagas = carregar_json(vagas_path)

    registros = []

    for id_processo, dados_processo in prospects.items():
        if dados_processo.get('status_final', '') == 'Contratado':
            id_candidato = dados_processo.get('id_candidato')
            id_vaga = dados_processo.get('id_vaga')

            candidato = applicants.get(id_candidato)
            vaga = vagas.get(id_vaga)

            if not candidato or not vaga:
                logger.warning(f"Dados ausentes para processo {id_processo}")
                continue

            cv = candidato.get('cv', '')
            ingles = candidato.get('nivel_ingles', 'Desconhecido')
            area = candidato.get('area_atuacao', 'Indefinida')

            perfil_vaga = vaga.get('perfil_vaga', {})
            titulo = vaga.get('informacoes_basicas', {}).get('titulo_vaga', '')
            atividades = perfil_vaga.get('principais_atividades', '')
            competencias = perfil_vaga.get('competencia_tecnicas_e_comportamentais', '')
            descricao_vaga = f"{titulo}. {atividades} {competencias}"

            try:
                resultado = avaliar_match(cv, ingles, area, descricao_vaga)
                registros.append({
                    "cv": cv,
                    "nivel_ingles": ingles,
                    "area_atuacao": area,
                    "descricao_vaga": descricao_vaga,
                    "score": resultado['score'],
                    "match": resultado['match'],
                    "perfil_recomendado": resultado['perfil_recomendado']
                })
            except Exception:
                logger.warning(f"Falha ao processar match para processo {id_processo}")

    df = pd.DataFrame(registros)
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset salvo com sucesso em '{output_path}' com {len(df)} registros.")
