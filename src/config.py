"""Configuration settings for the project."""

import os
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project settings."""

    model_config = SettingsConfigDict(
        env_file="./.env", env_file_encoding="utf-8", extra="allow"
    )

    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    INDEX_DIR: Path = DATA_DIR / "indexes"

    # Open AI API settings
    OPENAI_API_KEY: SecretStr | None = None

    # Lanfuse settings
    LANGFUSE_SECRET_KEY: SecretStr | None = None
    LANGFUSE_PUBLIC_KEY: SecretStr | None = None
    LANGFUSE_HOST: SecretStr | None = None

    # Data settings
    RAW_DATA_PATH: str = str(DATA_DIR / "QueryResults.csv")
    PROCESSED_DATA_PATH: str = str(DATA_DIR / "ProcessedData.csv")

    # Embeddings settings
    # EMBEDDINGS_MODEL_NAME: str = "BAAI/llm-embedder"
    EMBEDDINGS_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    # CROSS_ENCODER_MODEL_NAME: str = "BAAI/bge-reranker-base"
    CROSS_ENCODER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # LLM settings
    LLM_MODEL_NAME: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0
    LLM_MAX_TOKENS: int = 100

    FAISS_INDEX_PATH: str = str(INDEX_DIR / "faiss_index.faiss")
    BM25_INDEX_PATH: str = str(INDEX_DIR / "bm25_index.pkl")

    # Guadrail settings
    GUARDRAIL_SETTINGS_DIR: str = str(BASE_DIR / "src" / "core" / "guardrail")

    FAISS_TOP_K: int = 5
    BM25_TOP_K: int = 5

    RETTRIEVER_TOP_K: int = 10
    RETRIEVER_WEIGHTS: list[float] = [0.5, 0.5]

    COMPRESSOR_TOP_K: int = 3

    # Logging settings
    LOGGING_LEVEL: str = "INFO"
    LOGGING_FILE: str = str(BASE_DIR / "logs" / "preprocessing.log")

    # Ensure that the data directory exists
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.INDEX_DIR, exist_ok=True)
        os.makedirs(self.BASE_DIR / "logs", exist_ok=True)


settings = Settings()
