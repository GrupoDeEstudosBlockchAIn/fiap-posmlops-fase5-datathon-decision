from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model_evaluation import prever_match

app = FastAPI(title="Decision IA API", version="1.0")

class CandidatoInput(BaseModel):
    cv: str
    nivel_ingles: str
    area_atuacao: str

class ResultadoOutput(BaseModel):
    match: bool
    score: float
    perfil_recomendado: str

@app.post("/predict", response_model=ResultadoOutput)
def prever_candidato(dados: CandidatoInput):
    try:
        resultado = prever_match(dados.dict())
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
