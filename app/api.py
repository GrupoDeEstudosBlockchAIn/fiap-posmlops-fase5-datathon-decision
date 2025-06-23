# app/api.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
import os

# 🔹 Inicializando FastAPI
app = FastAPI(title="Decision Recruitment AI", description="API para predição de contratação - Datathon Decision", version="1.0")

# 🔹 Caminhos dos arquivos salvos
MODEL_PATH = 'saved_model/mlp_model.h5'
SCALER_PATH = 'saved_model/scaler.pkl'

# 🔹 Carregando o modelo e o scaler
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# 🔹 Classe para validar o input JSON
class CandidateData(BaseModel):
    idade: float
    experiencia: float
    salario: float
    cargo_Analista: int = 0
    cargo_Desenvolvedor: int = 0
    cargo_Gerente: int = 0
    cargo_Outros: int = 0

@app.get("/")
def root():
    return {"message": "API de Recrutamento - Decision Datathon - Status: Online"}

@app.post("/predict")
def predict(data: CandidateData):
    # 🔹 Convertendo input para numpy array
    input_data = np.array([[ 
        data.idade,
        data.experiencia,
        data.salario,
        data.cargo_Analista,
        data.cargo_Desenvolvedor,
        data.cargo_Gerente,
        data.cargo_Outros
    ]])

    # 🔹 Escalando os dados
    input_scaled = scaler.transform(input_data)

    # 🔹 Fazendo a predição
    prediction = model.predict(input_scaled)[0][0]

    # 🔹 Definindo o resultado final como 0 ou 1
    predicted_class = int(prediction >= 0.5)

    return {
        "probabilidade_de_contratacao": float(prediction),
        "classe_predita": predicted_class
    }
