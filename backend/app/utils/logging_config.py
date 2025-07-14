import logging

def setup_logging(name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Evita múltiplos handlers duplicados
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

        # Apenas log para o terminal (stdout), que a Railway captura automaticamente
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
