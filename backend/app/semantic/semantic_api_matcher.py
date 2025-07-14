import logging
from typing import List
from sentence_transformers import SentenceTransformer, util
from app.etl.backblaze_loader import download_json_from_backblaze
from app.utils.logging_config import setup_logging

# Configuração de logging
logger = setup_logging(__name__)

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("Modelo SBERT carregado com sucesso.")

def carregar_vaga_por_id(id_vaga):
    try:
        vagas = download_json_from_backblaze("vagas.json")
        vaga = vagas.get(id_vaga)
        if not vaga:
            raise ValueError(f"ID da vaga '{id_vaga}' não encontrado.")
        return vaga
    except Exception as e:
        logger.exception("Erro ao carregar vaga pelo ID.")
        raise e

def construir_texto_vaga(vaga_dict):
    perfil = vaga_dict.get('perfil_vaga', {})
    titulo = vaga_dict.get('informacoes_basicas', {}).get('titulo_vaga', '')
    atividades = perfil.get('principais_atividades', '')
    competencias = perfil.get('competencia_tecnicas_e_comportamentais', '')
    ingles = perfil.get('nivel_ingles', '')
    area = perfil.get('areas_atuacao', '')
    return f"{titulo}. Área: {area}. Inglês: {ingles}. {atividades} {competencias}"

def construir_texto_candidato(candidato):
    return f"{candidato.nome}. Área: {candidato.area_atuacao}. Inglês: {candidato.nivel_ingles}. {candidato.cv}"

def processar_match_sbert(candidato):
    logger.info(f"Processando match para o candidato '{candidato.nome}' e vaga '{candidato.id_vaga}'")
    try:
        vaga = carregar_vaga_por_id(candidato.id_vaga)
        texto_vaga = construir_texto_vaga(vaga)
        texto_candidato = construir_texto_candidato(candidato)

        embedding_vaga = sbert_model.encode(texto_vaga, convert_to_tensor=True)
        embedding_candidato = sbert_model.encode(texto_candidato, convert_to_tensor=True)

        score = float(util.cos_sim(embedding_candidato, embedding_vaga)[0])
        logger.info(f"Similaridade calculada: {score:.4f}")

        if score >= 0.6:
            perfil = "Match Técnico"
            match = True
        elif score >= 0.3:
            perfil = "Compatível"
            match = True
        else:
            perfil = "Não Compatível"
            match = False

        return {
            "match": match,
            "score": round(score, 2),
            "perfil_recomendado": perfil
        }
    except Exception as e:
        logger.exception("Erro no processamento do match SBERT")
        raise e

def rankear_candidatos(id_vaga, candidatos: List):
    logger.info(f"Gerando ranking para a vaga '{id_vaga}' com {len(candidatos)} candidatos...")
    try:
        vaga = carregar_vaga_por_id(id_vaga)
        texto_vaga = construir_texto_vaga(vaga)
        embedding_vaga = sbert_model.encode(texto_vaga, convert_to_tensor=True)

        resultados = []

        for candidato in candidatos:
            texto_candidato = construir_texto_candidato(candidato)
            embedding_candidato = sbert_model.encode(texto_candidato, convert_to_tensor=True)
            score = float(util.cos_sim(embedding_candidato, embedding_vaga)[0])

            if score >= 0.6:
                perfil = "Match Técnico"
            elif score >= 0.3:
                perfil = "Compatível"
            else:
                perfil = "Não Compatível"

            resultados.append({
                "nome": candidato.nome,
                "score": round(score, 2),
                "perfil_recomendado": perfil
            })

        resultados_ordenados = sorted(resultados, key=lambda x: x["score"], reverse=True)
        logger.info("Ranking gerado com sucesso.")
        return resultados_ordenados
    except Exception as e:
        logger.exception("Erro ao rankear candidatos.")
        raise e
