from langfuse.callback import CallbackHandler
from loguru import logger

from src.config import settings


def create_langfuse_handler():
    """
    Create a CallbackHandler for Langfuse.
    """
    langfuse_handler = CallbackHandler(
        public_key=settings.LANGFUSE_PUBLIC_KEY.get_secret_value(),
        secret_key=settings.LANGFUSE_SECRET_KEY.get_secret_value(),
        host=settings.LANGFUSE_HOST,
        # debug=True,
        # trace_name="parenting-chatbot",
    )
    try:
        langfuse_handler.auth_check()
        logger.info("Authenticated with langfuse_handler successfully.")
    except Exception as e:
        logger.error(
            "Failed to authenticate with langfuse_handler. Running without callback."
        )
    return langfuse_handler
