"""FastAPI dependencies — DI для RAG и БД."""

from typing import Annotated

from fastapi import Depends

from src.core.config import get_settings
from src.services.rag_service import RAGService


_rag_service: RAGService | None = None
_embed_model = None


def get_embed_model():
    """Возвращает embedding model для индексации."""
    global _embed_model
    if _embed_model is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        settings = get_settings()
        _embed_model = HuggingFaceEmbedding(
            model_name=settings.embedding_model,
            token=settings.huggingface_hub_token,
        )
    return _embed_model


def get_rag_service() -> RAGService:
    """Возвращает singleton RAG сервиса."""
    global _rag_service
    if _rag_service is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
        from src.llm.deepseek_llm import DeepSeekLLM
        from src.llm.openrouter_llm import OpenRouterLLM

        settings = get_settings()
        embed_model = HuggingFaceEmbedding(
            model_name=settings.embedding_model,
            token=settings.huggingface_hub_token,
        )
        provider = settings.llm_provider.lower()
        if provider == "deepseek":
            if not settings.deepseek_api_key:
                raise ValueError(
                    "DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek"
                )
            llm = DeepSeekLLM(
                model_name=settings.llm_model,
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                num_output=settings.llm_num_output,
                timeout_sec=settings.llm_timeout_sec,
            )
        elif provider == "openrouter":
            if not settings.openrouter_api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter"
                )
            llm = OpenRouterLLM(
                model_name=settings.llm_model,
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
                num_output=settings.llm_num_output,
                timeout_sec=settings.llm_timeout_sec,
            )
        elif provider == "hf":
            llm = HuggingFaceInferenceAPI(
                model_name=settings.llm_model,
                token=settings.huggingface_hub_token,
                num_output=settings.llm_num_output,
            )
        else:
            raise ValueError(
                "LLM_PROVIDER must be one of: deepseek, openrouter, hf"
            )
        _rag_service = RAGService(embed_model=embed_model, llm=llm)
    return _rag_service


RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
