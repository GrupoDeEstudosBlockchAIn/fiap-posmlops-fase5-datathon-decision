# Use uma imagem base oficial do Python
FROM python:3.10-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia os arquivos de requisitos (dependências)
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante da aplicação (código fonte + modelo treinado + scaler)
COPY . .

# Expondo a porta que o Uvicorn irá usar
EXPOSE 8000

# Comando para rodar a API FastAPI usando Uvicorn
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
