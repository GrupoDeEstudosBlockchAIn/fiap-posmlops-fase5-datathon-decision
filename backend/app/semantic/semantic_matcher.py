import logging
from sentence_transformers import SentenceTransformer, util
from app.utils.logging_config import setup_logging

# Configuração de logging
logger = setup_logging(__name__)

# Modelo SBERT
model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("Modelo SBERT carregado com sucesso.")

def gerar_texto_padronizado(cv: str, nivel_ingles: str, area_atuacao: str) -> str:
    texto = f"{cv}. Área de atuação: {area_atuacao}. Nível de inglês: {nivel_ingles}."
    return texto

def calcular_similaridade(texto1: str, texto2: str) -> float:
    logger.debug("Calculando similaridade entre textos...")
    emb1 = model.encode(texto1, convert_to_tensor=True)
    emb2 = model.encode(texto2, convert_to_tensor=True)
    score = float(util.cos_sim(emb1, emb2)[0])
    logger.info(f"Similaridade calculada: {score:.4f}")
    return score

def avaliar_match(cv_candidato: str, ingles: str, area: str, descricao_vaga: str) -> dict:
    logger.info("Avaliando compatibilidade entre candidato e vaga...")
    try:
        texto_candidato = gerar_texto_padronizado(cv_candidato, ingles, area)
        score = calcular_similaridade(texto_candidato, descricao_vaga)

        if score >= 0.6:
            match = True
            perfil = "Match Técnico"
        elif score >= 0.3:
            match = True
            perfil = "Compatível"
        else:
            match = False
            perfil = "Não Compatível"

        resultado = {
            "match": match,
            "score": round(score, 2),
            "perfil_recomendado": perfil
        }
        logger.info(f"Resultado: {resultado}")
        return resultado
    except Exception as e:
        logger.exception("Erro ao avaliar match semântico.")
        raise e
