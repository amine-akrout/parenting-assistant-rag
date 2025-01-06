from langfuse.callback import CallbackHandler

from config import settings


def create_langfuse_handler():
    """
    Create a CallbackHandler for Langfuse.
    """
    langfuse_handler = CallbackHandler(
        public_key=settings.LANGFUSE_PUBLIC_KEY.get_secret_value(),
        secret_key=settings.LANGFUSE_SECRET_KEY.get_secret_value(),
        host=settings.LANGFUSE_HOST.get_secret_value(),
    )
    return langfuse_handler
