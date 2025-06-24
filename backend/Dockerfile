# Etapa 1 — Imagem base
FROM python:3.10-slim

# Etapa 2 — Definir diretório de trabalho
WORKDIR /app

# Etapa 3 — Copiar arquivos para dentro do container
COPY . /app

# Etapa 4 — Instalar dependências
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Etapa 5 — Variável para decidir se roda API ou App
ENV MODE=api

# Etapa 6 — Porta padrão (API = 8000, Streamlit = 8501)
EXPOSE 8000
EXPOSE 8501

# Etapa 7 — Comando de inicialização condicional
CMD ["sh", "-c", "if [ \"$MODE\" = 'api' ]; then uvicorn app.api:app --host 0.0.0.0 --port 8000; else streamlit run streamlit_app.py --server.port 8501 --server.enableCORS false; fi"]

