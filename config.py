"""Configuration settings for the project."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Open AI API settings
    OPENAI_API_KEY: SecretStr | None = None

    # Lanfuse settings
    LANGFUSE_SECRET_KEY: SecretStr | None = None
    LANGFUSE_PUBLIC_KEY: SecretStr | None = None
    LANGFUSE_HOST: SecretStr | None = None

    # Data settings
    RAW_DATA_PATH: str = "data/QueryResults.csv"
    PROCESSED_DATA_PATH: str = "preprocessed_parenting_data.csv"

    # Embeddings settings
    # EMBEDDINGS_MODEL_NAME: str = "BAAI/llm-embedder"
    EMBEDDINGS_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    FAISS_INDEX_PATH: str = "faiss_index.faiss"

    # Logging settings
    LOGGING_LEVEL: str = "INFO"
    LOGGING_FILE: str = "preprocessing.log"


settings = Settings()
