# Etapa base: imagem oficial do Python
FROM python:3.10-slim

# Define diretório de trabalho no container
WORKDIR /app

# Copia arquivos de dependência
COPY requirements.txt .

# Instala as dependências do projeto
RUN pip install --no-cache-dir -r requirements.txt

# Copia todos os arquivos do backend para o container
COPY . .

# Expõe a porta padrão do FastAPI
EXPOSE 8000

# Comando para rodar a aplicação
CMD ["python", "main.py"]
