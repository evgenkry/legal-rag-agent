"""Основной RAG-сервис"""

import asyncio
import logging
from typing import Optional

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM

from src.core.config import get_settings
from src.dialog.orchestrator import ConversationOrchestrator
from src.infra.cache import cache_backend
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import get_retriever
from src.services.interaction_logger import log_interaction

logger = logging.getLogger(__name__)


class RAGService:
    """Единая точка доступа к RAG"""

    def __init__(
        self,
        embed_model: BaseEmbedding,
        llm: LLM,
    ):
        retriever = get_retriever(embed_model)
        self._pipeline = RAGPipeline(embed_model=embed_model, llm=llm, retriever=retriever)
        self._orchestrator = ConversationOrchestrator(self._pipeline, llm)

    async def query(
        self,
        question: str,
        user_id: Optional[str] = None,
        chat_history: Optional[list[dict]] = None,
        interaction_type: str = "query",
        trace_id: Optional[str] = None,
    ) -> dict:
        """Выполняет диалог + RAG и логирует взаимодействие"""
        settings = get_settings()
        cache_key = cache_backend.make_key("answer", question.strip(), str(chat_history or []))
        if settings.enable_redis_cache:
            cached = await cache_backend.get_json(cache_key)
            if cached:
                if user_id:
                    log_interaction(
                        user_id=user_id,
                        interaction_type=interaction_type,
                        question=question,
                        retrieved_chunk_ids=cached.get("retrieved_chunk_ids", []),
                        used_chunk_ids=cached.get("chunks_used", []),
                        sources_found=cached.get("sources_found", False),
                        llm_latency_ms=None,
                        retriever_latency_ms=None,
                        model_used=None,
                        trace_id=trace_id,
                        refusal_reason=None,
                        answer_full_text=cached.get("answer"),
                        agent_outcome=cached.get("agent_outcome"),
                        agent_trace_json={"from_cache": True},
                        from_cache=True,
                    )
                else:
                    logger.info(
                        "RAG cache hit (no user_id); key=%s",
                        cache_key[:24],
                    )
                return cached

        result = None
        for attempt in range(settings.llm_retry_attempts):
            try:
                result = await self._orchestrator.run(
                    question=question,
                    chat_history=chat_history,
                    trace_id=trace_id,
                )
                break
            except Exception as e:
                logger.warning("RAGService attempt %d failed: %s", attempt + 1, e)
                if attempt < settings.llm_retry_attempts - 1:
                    await asyncio.sleep(0.1 * (attempt + 1))
        if result is None:
            result = {
                "answer": "Недостаточно данных",
                "citations": [],
                "sources_found": False,
                "chunks_used": [],
                "retrieved_chunk_ids": [],
                "context_texts": [],
                "latency_ms": None,
                "agent_outcome": "fallback_error",
            }

        if user_id:
            log_interaction(
                user_id=user_id,
                interaction_type=interaction_type,
                question=question,
                retrieved_chunk_ids=result.get("retrieved_chunk_ids", []),
                used_chunk_ids=result.get("chunks_used", []),
                sources_found=result.get("sources_found", False),
                llm_latency_ms=result.get("llm_latency_ms"),
                retriever_latency_ms=result.get("retriever_latency_ms"),
                model_used=get_settings().llm_model,
                trace_id=trace_id,
                rewritten_query=result.get("rewritten_query"),
                reranked_chunk_ids=result.get("reranked_chunk_ids"),
                stage_timings=result.get("stage_timings"),
                top_scores=result.get("top_scores"),
                refusal_reason=result.get("refusal_reason"),
                answer_full_text=result.get("answer_full_text"),
                candidates_json=result.get("candidates_json"),
                llm_context_json=result.get("llm_context_json"),
                agent_outcome=result.get("agent_outcome"),
                agent_trace_json=result.get("agent_trace"),
            )

        response = {
            "answer": result["answer"],
            "citations": result["citations"],
            "sources_found": result["sources_found"],
            "chunks_used": result["chunks_used"],
            "retrieved_chunk_ids": result.get("retrieved_chunk_ids", []),
            "context_texts": result.get("context_texts", []),
            "latency_ms": result.get("latency_ms"),
            "agent_outcome": result.get("agent_outcome"),
        }
        if settings.enable_redis_cache:
            await cache_backend.set_json(cache_key, response, ttl_sec=settings.answer_cache_ttl_sec)
        return response
