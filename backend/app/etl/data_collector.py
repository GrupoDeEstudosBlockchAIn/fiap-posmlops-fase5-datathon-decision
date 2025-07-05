import json
import pandas as pd
from pathlib import Path
import logging
from app.etl.backblaze_loader import download_json_from_backblaze
from app.utils.constants import CONST_DATABASE_APPLICANTS, CONST_DATABASE_PROSPECTS, CONST_DATABASE_VAGAS

# Configuração de logging
logger = logging.getLogger(__name__)

def coletar_dados():
    try:
        logger.info("Iniciando coleta de dados...")

        applicants = download_json_from_backblaze(CONST_DATABASE_APPLICANTS)
        prospects = download_json_from_backblaze(CONST_DATABASE_PROSPECTS)

        dados = []
        total_candidatos = 0
        ignorados = 0

        for vaga_id, vaga_data in prospects.items():
            for candidato in vaga_data.get("prospects", []):
                codigo_candidato = candidato.get("codigo")
                status = candidato.get("situacao_candidado", "")
                contratado = 1 if "contratado" in status.lower() else 0

                dados_applicant = applicants.get(codigo_candidato, {})
                if not dados_applicant:
                    ignorados += 1
                    continue

                cv = dados_applicant.get("cv_pt", "")
                ingles = dados_applicant.get("formacao_e_idiomas", {}).get("nivel_ingles", "")
                area = dados_applicant.get("informacoes_profissionais", {}).get("area_atuacao", "")
                nome = dados_applicant.get("informacoes_pessoais", {}).get("nome", "")

                dados.append({
                    "codigo": codigo_candidato,
                    "nome": nome,
                    "nivel_ingles": ingles,
                    "area_atuacao": area,
                    "cv": cv,
                    "contratado": contratado
                })
                total_candidatos += 1

        logger.info(f"Dados coletados com sucesso! Total: {total_candidatos} | Ignorados: {ignorados}")
        return pd.DataFrame(dados)

    except FileNotFoundError as e:
        logger.error(f"Erro ao carregar arquivos JSON: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.exception("Erro inesperado durante a coleta de dados.")
        return pd.DataFrame()
