"""RAG-пайплайн"""

import logging
import time
from typing import Optional

from src.agent.answer_verifier import AnswerVerifier
from src.agent.rule_based_verifier import RuleBasedVerifier
from src.core.config import get_settings
from src.infra.cache import cache_backend
from src.rag.context_builder import ContextBuilder
from src.rag.generator import Generator
from src.rag.pipeline_steps import (
    GenerationStep,
    NormalizeQueryStep,
    PipelineContext,
    PipelineExecutor,
    RerankStep,
    RetrievalStep,
    VerificationStep,
)
from src.rag.query_rewriter import QueryRewriter
from src.rag.reranker import Reranker
from src.rag.retriever import HybridRetriever, get_retriever

logger = logging.getLogger(__name__)
NO_RESULTS_REFUSAL = (
    "В базе знаний не найдено подходящих фрагментов по Вашему запросу. "
    "Попробуйте переформулировать или дополнить вопрос"
)


class RAGPipeline:
    """Шаги RAG: Normalize -> Retrieve -> Rerank -> Generate -> Verify"""

    def __init__(
        self,
        embed_model,
        llm,
        query_rewriter: Optional[QueryRewriter] = None,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[Reranker] = None,
        generator: Optional[Generator] = None,
        answer_verifier: Optional[AnswerVerifier] = None,
    ):
        settings = get_settings()
        self._embed_model = embed_model
        self._llm = llm

        self._query_rewriter = query_rewriter or QueryRewriter(llm)
        self._query_rewriter.set_llm(llm)

        self._retriever = retriever or get_retriever(embed_model)
        self._fallback_retriever = HybridRetriever(embed_model)
        self._context_builder = ContextBuilder(get_index=lambda: self._retriever.get_index())
        self._reranker = reranker or Reranker(
            model_name=settings.reranker_model,
            top_n=settings.rerank_top_k,
        )
        self._generator = generator or Generator(llm)
        self._generator.set_llm(llm)
        self._answer_verifier = answer_verifier or AnswerVerifier(llm)
        self._answer_verifier.set_llm(llm)

        self._executor = PipelineExecutor(
            [
                NormalizeQueryStep(self._query_rewriter),
                RetrievalStep(self._retriever, self._fallback_retriever, cache_backend),
                RerankStep(self._reranker),
                GenerationStep(
                    self._generator,
                    self._context_builder,
                    get_index=lambda: self._retriever.get_index(),
                ),
                VerificationStep(self._answer_verifier, RuleBasedVerifier()),
            ]
        )

    async def query(
        self,
        question: str,
        chat_history: Optional[list[dict]] = None,
        use_rewrite: bool = True,
        trace_id: Optional[str] = None,
        rerank_top_k: Optional[int] = None,
        retrieval_query: Optional[str] = None,
        run_verifier: bool = True,
    ) -> dict:
        """
        Выполняет RAG-запрос.
        retrieval_query: если задан (например, агентом), то используется для поиска вместо внутреннего rewrite
        """
        start = time.perf_counter()
        settings = get_settings()
        context = PipelineContext(
            user_query=question,
            retrieval_query=(retrieval_query if use_rewrite else None),
            chat_history=chat_history or [],
            trace_id=trace_id,
            rerank_top_k=rerank_top_k if rerank_top_k is not None else settings.rerank_top_k,
            run_verifier=run_verifier,
        )
        context = await self._executor.run(context)
        stage_timings = context.metadata.get("stage_timings", {})
        nodes = context.reranked_docs
        retrieved_chunk_ids = context.metadata.get("retrieved_chunk_ids", [])
        reranked_chunk_ids = context.metadata.get("reranked_chunk_ids", [])
        candidates_after_rerank = context.metadata.get("candidates_json", [])

        threshold = settings.rag_relevance_threshold
        max_score = None
        if nodes:
            scores = [getattr(n, "score", None) for n in nodes]
            valid_scores = [s for s in scores if s is not None]
            max_score = max(valid_scores) if valid_scores else None

        if max_score is not None and max_score < threshold:
            refusal = NO_RESULTS_REFUSAL
            return {
                "answer": refusal,
                "answer_full_text": refusal,
                "citations": [],
                "sources_found": False,
                "chunks_used": [],
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "reranked_chunk_ids": reranked_chunk_ids,
                "context_texts": [n.node.get_content() for n in nodes] if nodes else [],
                "candidates_json": candidates_after_rerank,
                "llm_context_json": context.metadata.get("llm_context_json", []),
                "retriever_latency_ms": stage_timings.get("RetrievalStep_ms", 0.0),
                "llm_latency_ms": 0,
                "latency_ms": (time.perf_counter() - start) * 1000,
                "refusal_reason": "low_relevance",
                "max_score": max_score,
                "trace_id": trace_id,
                "rewritten_query": context.normalized_query,
                "stage_timings": stage_timings,
                "top_scores": [getattr(n, "score", None) for n in nodes],
            }
        elif not nodes and threshold > float("-inf"):
            refusal = NO_RESULTS_REFUSAL
            return {
                "answer": refusal,
                "answer_full_text": refusal,
                "citations": [],
                "sources_found": False,
                "chunks_used": [],
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "reranked_chunk_ids": [],
                "context_texts": [],
                "candidates_json": [],
                "llm_context_json": [],
                "retriever_latency_ms": stage_timings.get("RetrievalStep_ms", 0.0),
                "llm_latency_ms": 0,
                "latency_ms": (time.perf_counter() - start) * 1000,
                "refusal_reason": "no_results",
                "max_score": None,
                "trace_id": trace_id,
                "rewritten_query": context.normalized_query,
                "stage_timings": stage_timings,
                "top_scores": [],
            }
        answer_text = context.answer_text or ""
        citations = context.citations
        if citations:
            numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(citations, 1))
            answer = answer_text.rstrip() + "\n\nИсточники\n" + numbered
        else:
            answer = answer_text

        context_texts = [n.node.get_content() for n in nodes] if nodes else []
        top_scores = [getattr(n, "score", None) for n in nodes]
        total_latency = (time.perf_counter() - start) * 1000
        refusal_reason = context.metadata.get("refusal_reason")
        llm_context_json = context.metadata.get("llm_context_json", [])
        chunks_used = context.metadata.get("chunks_used", [])
        sources_found = bool(context.metadata.get("sources_found", False))

        out = {
            "answer": answer,
            "answer_full_text": answer,
            "citations": citations,
            "sources_found": sources_found,
            "chunks_used": chunks_used,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "reranked_chunk_ids": reranked_chunk_ids,
            "context_texts": context_texts,
            "candidates_json": candidates_after_rerank,
            "llm_context_json": llm_context_json,
            "retriever_latency_ms": stage_timings.get("RetrievalStep_ms", 0.0),
            "llm_latency_ms": (
                stage_timings.get("GenerationStep_ms", 0.0)
                + stage_timings.get("VerificationStep_ms", 0.0)
            ),
            "latency_ms": total_latency,
            "refusal_reason": refusal_reason,
            "max_score": max_score,
            "trace_id": trace_id,
            "rewritten_query": context.normalized_query,
            "stage_timings": stage_timings,
            "top_scores": top_scores,
        }
        if vt := context.metadata.get("verifier_trace"):
            out["verifier_trace"] = vt
        return out
