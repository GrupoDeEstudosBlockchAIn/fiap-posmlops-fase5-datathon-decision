FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --retries=5 CMD curl --silent --fail http://127.0.0.1:8000/docs || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
