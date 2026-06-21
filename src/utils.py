import time
from loguru import logger
import os

# Configure structural runtime tracking logs
LOG_FILE_PATH = os.path.join(
    os.path.dirname(__file__), "../data/processed/pipeline.log"
)

logger.add(
    LOG_FILE_PATH,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
    rotation="10 MB",
)


def log_execution_time(func):
    """
    Decorator matrix to record how fast parts of our RAG architecture process chunks.
    Essential profile indicator for production audits.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Execution started for task pipeline: '{func.__name__}'")

        result = func(*args, **kwargs)

        elapsed_time = time.time() - start_time
        logger.info(
            f"Finished pipeline step '{func.__name__}' inside {elapsed_time:.4f} seconds."
        )
        return result

    return wrapper
