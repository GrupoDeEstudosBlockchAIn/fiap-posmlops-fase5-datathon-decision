FROM python:3.10-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=480s --retries=10 \
  CMD curl --fail http://localhost:8000/docs || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
