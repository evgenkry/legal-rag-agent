"""Конфигурация приложения."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения из переменных окружения."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # dv
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db"
    database_url_sync: str = "postgresql://postgres:postgres@localhost:5432/rag_db"

    # telegram
    telegram_bot_token: str = ""

    # hugging face 
    huggingface_hub_token: Optional[str] = None

    # models
    llm_provider: str = "deepseek"  # deepseek | openrouter | hf
    embedding_model: str = "intfloat/multilingual-e5-large"
    llm_model: str = "deepseek-chat"
    llm_model_fallback: str = "Qwen/Qwen2.5-32B-Instruct"
    # LlamaIndex HuggingFaceInferenceAPI default num_output=256 — мало для развёрнутых ответов (обрыв ответа)
    llm_num_output: int = 2048
    reranker_model: str = "qilowoq/bge-reranker-v2-m3-en-ru"
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    # RAG
    retrieval_top_k: int = 10
    rerank_top_k: int = 5
    embed_dim: int = 1024  # multilingual-e5-large
    context_window_size: int = 8
    max_context_tokens_est: int = 8000
    approx_chars_per_token: float = 3.5

    # explicit hybrid retrieval (sparse FTS + dense pgvector + RRF)
    rag_retrieval_mode: str = "llamaindex_hybrid"  # "explicit_hybrid" | "llamaindex_hybrid"
    sparse_top_k: int = 20
    dense_top_k: int = 20
    fused_top_k: int = 20
    rrf_k: int = 60
    
    # retrieval metadata policy
    retrieval_policy_top_k: int = 6
    law_boost: float = 0.2
    law_min: int = 2
    explanation_min: int = 2

    # app
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # admin
    admin_api_key: Optional[str] = None

    # RAG relevance gate (reranker score threshold; refuse if max_score < threshold)
    rag_relevance_threshold: float = 0.0

    # подмешивание полного текста FAQ при попадании retrieval-чанка
    enable_faq_pair_expansion: bool = True

    # абляции
    enable_query_rewrite: bool = True
    enable_reranker: bool = True
    enable_reference_expansion: bool = True
    hybrid_mode: str = "full"  # full | dense_only | sparse_only | llamaindex

    # LLM-агент (если True — запускается LegalQueryAgent; иначе см. enable_query_rewrite)
    enable_legal_query_agent: bool = True
    enable_answer_verifier: bool = True
    enable_rule_based_verifier: bool = True

    # timeouts / retries
    llm_timeout_sec: float = 10.0
    retriever_timeout_sec: float = 10.0
    llm_retry_attempts: int = 3

    # redis cache / rate limit
    redis_url: str = "redis://localhost:6379/0"
    enable_redis_cache: bool = True
    answer_cache_ttl_sec: int = 300
    retrieval_cache_ttl_sec: int = 120
    enable_rate_limit: bool = True
    rate_limit_per_minute: int = 100

    @property
    def pgvector_url(self) -> str:
        """URL для pgvector/LlamaIndex (синхронный psycopg2)."""
        return self.database_url_sync.replace("postgresql+asyncpg://", "postgresql://")


@lru_cache
def get_settings() -> Settings:
    """Возвращает кэшированный экземпляр настроек."""
    return Settings()
