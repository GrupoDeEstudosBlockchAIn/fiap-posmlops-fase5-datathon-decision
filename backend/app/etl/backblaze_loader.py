# app/etl/backblaze_loader.py
import requests
import logging
import time

logger = logging.getLogger(__name__)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def download_json_from_backblaze(filename: str, retries: int = 3, delay: int = 10):
    base_url = "https://f005.backblazeb2.com/file/vagas-data-api/"
    url = base_url + filename
    logger.info(f"Baixando JSON da URL: {url}")

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Tentativa {attempt} falhou ao baixar ou carregar JSON '{filename}': {e}")
            if attempt < retries:
                logger.info(f"Aguardando {delay} segundos antes de tentar novamente...")
                time.sleep(delay)
            else:
                raise
