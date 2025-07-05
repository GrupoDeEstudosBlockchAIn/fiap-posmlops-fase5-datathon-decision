import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from app.semantic.semantic_api_matcher import processar_match_sbert

# Configuração de logging
logger = logging.getLogger(__name__)

app = FastAPI(title="Decision SBERT API", version="2.0")

class CandidatoRequest(BaseModel):
    id_vaga: str
    nome: str
    cv: str
    nivel_ingles: str
    area_atuacao: str

class ResultadoMatch(BaseModel):
    match: bool
    score: float
    perfil_recomendado: str

class CandidatoRanking(BaseModel):
    nome: str
    cv: str
    nivel_ingles: str
    area_atuacao: str

class RankingRequest(BaseModel):
    id_vaga: str
    candidatos: List[CandidatoRanking]

class RankingOutput(BaseModel):
    nome: str
    score: float
    perfil_recomendado: str

@app.post("/match", response_model=ResultadoMatch)
def obter_match_semantico(candidato: CandidatoRequest):
    try:
        logger.info(f"Recebida solicitação de /match para candidato: {candidato.nome}")
        resultado = processar_match_sbert(candidato)
        logger.info(f"Match processado com sucesso para: {candidato.nome} | Score: {resultado['score']}")
        return resultado
    except Exception as e:
        logger.error(f"Erro ao processar match para {candidato.nome}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rank", response_model=List[RankingOutput])
def obter_ranking_candidatos(request: RankingRequest):
    try:
        logger.info(f"Recebida solicitação de /rank para vaga: {request.id_vaga} com {len(request.candidatos)} candidatos.")
        from app.semantic.semantic_api_matcher import rankear_candidatos
        resultado = rankear_candidatos(request.id_vaga, request.candidatos)
        logger.info(f"Ranking gerado com sucesso para vaga: {request.id_vaga}")
        return resultado
    except Exception as e:
        logger.error(f"Erro ao gerar ranking: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
