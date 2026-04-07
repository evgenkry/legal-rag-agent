"""Шаги пайплайна"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.agent.answer_verifier import VERIFIER_REFUSAL, AnswerVerifier
from src.agent.rule_based_verifier import RuleBasedVerifier
from src.core.config import get_settings
from src.rag.candidate_snapshot import (
    nodes_to_candidate_records,
    nodes_to_llm_context_records,
)
from src.rag.context_builder import ContextBuilder
from src.rag.faq_context import expand_faq_context_nodes
from src.rag.generator import Generator
from src.rag.query_rewriter import QueryRewriter
from src.rag.retrieval_policy import apply_law_explanation_policy
from src.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class PipelineContext(BaseModel):
    user_query: str
    normalized_query: str | None = None
    retrieval_query: str | None = None
    chat_history: list[dict] = Field(default_factory=list)
    trace_id: str | None = None
    rerank_top_k: int | None = None
    run_verifier: bool = True

    retrieved_docs: list = Field(default_factory=list)
    reranked_docs: list = Field(default_factory=list)
    answer: str | None = None
    answer_text: str | None = None
    citations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineStep(ABC):
    @abstractmethod
    async def run(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError


class PipelineExecutor:
    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps

    async def run(self, context: PipelineContext) -> PipelineContext:
        for step in self.steps:
            started = time.perf_counter()
            context = await step.run(context)
            timings = context.metadata.setdefault("stage_timings", {})
            timings[f"{step.__class__.__name__}_ms"] = (time.perf_counter() - started) * 1000
        return context


class NormalizeQueryStep(PipelineStep):
    def __init__(self, query_rewriter: QueryRewriter):
        self._query_rewriter = query_rewriter

    async def run(self, context: PipelineContext) -> PipelineContext:
        settings = get_settings()
        if context.retrieval_query is not None:
            context.normalized_query = (context.retrieval_query or "").strip() or context.user_query
            return context
        if settings.enable_query_rewrite:
            timeout = settings.llm_timeout_sec
            try:
                context.normalized_query = await asyncio.wait_for(
                    self._query_rewriter.rewrite(context.user_query), timeout=timeout
                )
            except Exception:
                context.normalized_query = context.user_query
        else:
            context.normalized_query = context.user_query
        return context


class RetrievalStep(PipelineStep):
    def __init__(
        self,
        retriever,
        fallback_retriever: HybridRetriever,
        cache_backend=None,
    ):
        self._retriever = retriever
        self._fallback_retriever = fallback_retriever
        self._cache = cache_backend

    async def run(self, context: PipelineContext) -> PipelineContext:
        settings = get_settings()
        query = context.normalized_query or context.user_query
        cache_key = None
        if self._cache:
            cache_key = self._cache.make_key("retrieval", query, str(context.rerank_top_k or 0))
            cached = await self._cache.get_json(cache_key)
            if cached and isinstance(cached.get("node_ids"), list):
                context.metadata["retrieval_cache_hit"] = True

        extra: dict[str, Any] = {}
        try:
            if hasattr(self._retriever, "retrieve_async"):
                context.retrieved_docs, extra = await asyncio.wait_for(
                    self._retriever.retrieve_async(query, trace_id=context.trace_id),
                    timeout=settings.retriever_timeout_sec,
                )
            else:
                context.retrieved_docs = await asyncio.wait_for(
                    asyncio.to_thread(self._retriever.retrieve, query),
                    timeout=settings.retriever_timeout_sec,
                )
        except Exception as e:
            logger.warning("Primary retriever failed, fallback enabled: %s", e)
            extra["fallback_used"] = True
            context.retrieved_docs = await asyncio.wait_for(
                asyncio.to_thread(self._fallback_retriever.retrieve, query),
                timeout=settings.retriever_timeout_sec,
            )
        if extra:
            context.metadata.setdefault("stage_timings", {}).update(extra)
        context.metadata["retrieved_chunk_ids"] = [
            getattr(n.node, "node_id", None) or getattr(n.node, "id_", "")
            for n in context.retrieved_docs
        ]
        if self._cache and cache_key:
            await self._cache.set_json(
                cache_key,
                {"node_ids": context.metadata["retrieved_chunk_ids"]},
                ttl_sec=settings.retrieval_cache_ttl_sec,
            )
        return context


class RerankStep(PipelineStep):
    def __init__(self, reranker):
        self._reranker = reranker

    async def run(self, context: PipelineContext) -> PipelineContext:
        settings = get_settings()
        nodes = context.retrieved_docs
        nodes = apply_law_explanation_policy(
            nodes,
            top_k=settings.retrieval_policy_top_k,
            law_boost=settings.law_boost,
            law_min=settings.law_min,
            explanation_min=settings.explanation_min,
        )
        rk = context.rerank_top_k if context.rerank_top_k is not None else settings.rerank_top_k
        if settings.enable_reranker:
            nodes = self._reranker.rerank(context.normalized_query or context.user_query, nodes, top_n=rk)
        else:
            nodes = nodes[:rk] if len(nodes) > rk else nodes
        context.reranked_docs = nodes
        context.metadata["reranked_chunk_ids"] = [
            getattr(n.node, "node_id", None) or getattr(n.node, "id_", "") for n in nodes
        ]
        context.metadata["candidates_json"] = nodes_to_candidate_records(nodes)
        return context


class GenerationStep(PipelineStep):
    def __init__(self, generator: Generator, context_builder: ContextBuilder, get_index):
        self._generator = generator
        self._context_builder = context_builder
        self._get_index = get_index

    async def run(self, context: PipelineContext) -> PipelineContext:
        settings = get_settings()
        nodes = context.reranked_docs
        if settings.enable_faq_pair_expansion:
            nodes = expand_faq_context_nodes(nodes, self._get_index)
        if settings.enable_reference_expansion:
            nodes = self._context_builder.expand(nodes)
        context.reranked_docs = nodes
        context.metadata["llm_context_json"] = nodes_to_llm_context_records(nodes)
        context.metadata["chunks_used"] = [
            getattr(n.node, "node_id", None) or getattr(n.node, "id_", "")
            for n in nodes
        ]
        context.metadata["sources_found"] = bool(nodes)
        context.answer_text, context.citations = await self._generator.generate(
            question=context.user_query,
            nodes=nodes,
            chat_history=context.chat_history,
        )
        return context


class VerificationStep(PipelineStep):
    def __init__(
        self,
        answer_verifier: AnswerVerifier,
        rule_based_verifier: RuleBasedVerifier,
    ):
        self._answer_verifier = answer_verifier
        self._rule_based_verifier = rule_based_verifier

    async def run(self, context: PipelineContext) -> PipelineContext:
        settings = get_settings()
        answer_text = (context.answer_text or "").strip()
        nodes = context.reranked_docs
        if not context.run_verifier or not settings.enable_answer_verifier or not nodes or not answer_text:
            return context

        if settings.enable_rule_based_verifier:
            rb = self._rule_based_verifier.verify(answer_text, nodes)
            if not rb.passed:
                logger.info("RuleBasedVerifier reject reason=%s", rb.reason)
                context.metadata["refusal_reason"] = rb.reason or "rule_reject"
                context.answer_text = VERIFIER_REFUSAL
                context.answer = VERIFIER_REFUSAL
                context.citations = []
                context.metadata["sources_found"] = False
                context.metadata["verifier_trace"] = {"claims": [], "substantive": False}
                return context

        vres, body_or_refusal = await self._answer_verifier.verify(
            context.user_query, answer_text, nodes
        )
        context.metadata["verifier_trace"] = {
            "claims": [c.model_dump() for c in vres.claims],
            "substantive": vres.substantive,
            "revised_answer_body": vres.revised_answer_body,
        }
        if body_or_refusal == VERIFIER_REFUSAL or not vres.substantive:
            context.metadata["refusal_reason"] = "verifier_no_substance"
            context.answer_text = VERIFIER_REFUSAL
            context.answer = VERIFIER_REFUSAL
            context.citations = []
            context.metadata["sources_found"] = False
            return context
        context.answer_text = body_or_refusal
        return context
