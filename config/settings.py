from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    default_llm_provider: str = "openai"
    default_model: str = "gpt-4o"
    default_embedding_model: str = "text-embedding-3-small"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"

    # Memory
    memory_decay_rate: float = 0.05
    consolidation_interval_seconds: int = 3600
    max_episodic_memories: int = 10000
    activation_threshold: float = 0.3

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
