"""LLM provider abstraction layer."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from config.settings import settings


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Get configured LLM instance."""
    provider = provider or settings.default_llm_provider
    model = model or settings.default_model

    if provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=settings.openai_api_key,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=settings.anthropic_api_key,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_embeddings(model: str | None = None) -> Embeddings:
    """Get configured embeddings instance."""
    model = model or settings.default_embedding_model
    return OpenAIEmbeddings(
        model=model,
        api_key=settings.openai_api_key,
    )
