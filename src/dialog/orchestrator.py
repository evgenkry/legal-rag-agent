"""Оркестратор: безопасность → агент запроса → RAG + верификатор"""

import time
from typing import Any, Optional

from src.agent.legal_query_agent import LegalQueryAgent
from src.core.config import get_settings
from src.dialog.safety_gate import UNSAFE_REFUSAL, is_unsafe_question
from src.rag.pipeline import RAGPipeline


class ConversationOrchestrator:
    def __init__(self, pipeline: RAGPipeline, llm):
        self._pipeline = pipeline
        self._legal_agent = LegalQueryAgent(llm)

    async def run(
        self,
        question: str,
        chat_history: Optional[list[dict]] = None,
        trace_id: Optional[str] = None,
    ) -> dict:
        settings = get_settings()
        agent_trace: dict[str, Any] = {}

        if is_unsafe_question(question):
            start = time.perf_counter()
            latency = (time.perf_counter() - start) * 1000
            msg = UNSAFE_REFUSAL
            return {
                "answer": msg,
                "answer_full_text": msg,
                "citations": [],
                "sources_found": False,
                "chunks_used": [],
                "retrieved_chunk_ids": [],
                "reranked_chunk_ids": [],
                "context_texts": [],
                "candidates_json": [],
                "llm_context_json": [],
                "retriever_latency_ms": 0.0,
                "llm_latency_ms": 0.0,
                "latency_ms": latency,
                "refusal_reason": "unsafe",
                "max_score": None,
                "trace_id": trace_id,
                "rewritten_query": None,
                "stage_timings": {},
                "top_scores": [],
                "agent_outcome": "refused_unsafe",
                "agent_trace": agent_trace,
            }

        retrieval_query: Optional[str] = None
        if settings.enable_legal_query_agent:
            qa_start = time.perf_counter()
            qnorm = await self._legal_agent.normalize(question)
            agent_trace["query_agent"] = qnorm.model_dump()
            retrieval_query = qnorm.legal_query
            agent_trace["legal_query_agent_ms"] = (time.perf_counter() - qa_start) * 1000

        rk = settings.rerank_top_k

        result = await self._pipeline.query(
            question=question,
            chat_history=chat_history,
            trace_id=trace_id,
            rerank_top_k=rk,
            retrieval_query=retrieval_query,
            run_verifier=settings.enable_answer_verifier,
        )

        if vt := result.pop("verifier_trace", None):
            agent_trace["verifier"] = vt
        result["agent_trace"] = agent_trace

        if lq_ms := agent_trace.get("legal_query_agent_ms"):
            st = dict(result.get("stage_timings") or {})
            st["legal_query_agent_ms"] = lq_ms
            result["stage_timings"] = st

        # Итоговый исход для логов
        if result.get("refusal_reason") == "unsafe":
            outcome = "refused_unsafe"
        elif result.get("refusal_reason") == "low_relevance":
            outcome = "refused_low_relevance"
        elif result.get("refusal_reason") == "no_results":
            outcome = "refused_no_results"
        elif result.get("refusal_reason") == "verifier_no_substance":
            outcome = "refused_verifier"
        else:
            outcome = "answered"
        result["agent_outcome"] = outcome

        return result
