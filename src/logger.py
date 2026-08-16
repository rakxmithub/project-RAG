import logging
import os
from datetime import datetime


# ==============================================
# Create Logs Directory
# ==============================================

LOG_DIR = "logs"

os.makedirs(
    LOG_DIR,
    exist_ok=True
)


# ==============================================
# Log File
# ==============================================

LOG_FILE = os.path.join(
    LOG_DIR,
    "rag.log"
)


# ==============================================
# Logger
# ==============================================

logger = logging.getLogger(
    "rag_system"
)

logger.setLevel(
    logging.INFO
)


# Prevent duplicate handlers
if not logger.handlers:

    # ------------------------------------------
    # File Handler
    # ------------------------------------------

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )

    # ------------------------------------------
    # Console Handler
    # ------------------------------------------

    console_handler = logging.StreamHandler()


    # ------------------------------------------
    # Formatter
    # ------------------------------------------

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler.setFormatter(
        formatter
    )

    console_handler.setFormatter(
        formatter
    )


    # ------------------------------------------
    # Add Handlers
    # ------------------------------------------

    logger.addHandler(
        file_handler
    )

    logger.addHandler(
        console_handler
    )


# ==============================================
# Helper Functions
# ==============================================

def log_info(message):

    logger.info(
        message
    )


def log_warning(message):

    logger.warning(
        message
    )


def log_error(message):

    logger.error(
        message
    )


def log_query(question):

    logger.info(
        f"QUERY | {question}"
    )


def log_rewritten_query(query):

    logger.info(
        f"REWRITTEN_QUERY | {query}"
    )


def log_retrieval(count):

    logger.info(
        f"RETRIEVAL | documents={count}"
    )


def log_reranking(count):

    logger.info(
        f"RERANKING | top_documents={count}"
    )


def log_answer(answer):

    logger.info(
        f"ANSWER | length={len(answer)}"
    )