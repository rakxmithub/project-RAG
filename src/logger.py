import logging
import os


# ============================================================
# LOG DIRECTORY
# ============================================================

LOG_DIR = "logs"

os.makedirs(
    LOG_DIR,
    exist_ok=True
)


LOG_FILE = os.path.join(
    LOG_DIR,
    "rag.log"
)


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(
    "rag_system"
)

logger.setLevel(
    logging.INFO
)

logger.propagate = False


if not logger.handlers:

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        file_handler
    )


# ============================================================
# BASIC
# ============================================================

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


def log_exception(stage, error):

    logger.exception(
        f"ERROR | stage={stage} | error={error}"
    )


# ============================================================
# QUERY
# ============================================================

def log_query(question):

    logger.info(
        f"QUERY | question={question}"
    )


def log_rewritten_query(
    rewritten_question
):

    logger.info(
        f"QUERY_REWRITE | rewritten={rewritten_question}"
    )


# ============================================================
# RETRIEVAL
# ============================================================

def log_retrieval(
    count
):

    logger.info(
        f"RETRIEVAL | documents={count}"
    )


# ============================================================
# RERANKING
# ============================================================

def log_reranking(
    count
):

    logger.info(
        f"RERANKING | documents={count}"
    )


# ============================================================
# GENERATION
# ============================================================

def log_answer(answer):

    logger.info(
        f"GENERATION | answer_length={len(answer)}"
    )


# ============================================================
# LATENCY
# ============================================================

def log_latency(
    stage,
    latency
):

    logger.info(
        f"LATENCY | stage={stage} | time={latency:.3f}s"
    )


# ============================================================
# REQUEST
# ============================================================

def log_request(
    question,
    rewritten_question,
    retrieved_count,
    reranked_count,
    latency
):

    logger.info(
        "REQUEST | "
        f"question={question} | "
        f"rewritten={rewritten_question} | "
        f"retrieved={retrieved_count} | "
        f"reranked={reranked_count} | "
        f"latency={latency:.3f}s"
    )


# ============================================================
# READ LOGS
# ============================================================

def read_logs():

    if not os.path.exists(
        LOG_FILE
    ):

        return []

    try:

        with open(
            LOG_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return file.readlines()

    except Exception:

        return []


# ============================================================
# CLEAR LOGS
# ============================================================

def clear_logs():

    try:

        with open(
            LOG_FILE,
            "w",
            encoding="utf-8"
        ):

            pass

    except Exception as e:

        logger.error(
            f"Could not clear logs: {e}"
        )


# ============================================================
# LOG PATH
# ============================================================

def get_log_file():

    return LOG_FILE