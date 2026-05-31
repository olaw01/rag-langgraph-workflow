from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # OpenAI
    openai_api_key: str
    model_name: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Data
    docs_dir: str = "./data/docs"
    chroma_dir: str = "./data/chroma"

    # Retrieval / verification
    retriever_k: int = 4
    verify_threshold: float = 0.7
    max_iterations: int = 3

    # Logging
    log_level: str = "INFO"