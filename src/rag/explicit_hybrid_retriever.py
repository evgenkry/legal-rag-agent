"""Explicit hybrid retriever: sparse (FTS) + dense (pgvector) + RRF fusion"""

import asyncio
import logging
import time
from typing import Optional

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import NodeWithScore, TextNode

from src.core.config import get_settings
from src.rag.dense_retriever import dense_retrieve
from src.rag.rrf import FusedResult, rrf_fuse
from src.rag.sparse_retriever import sparse_retrieve

logger = logging.getLogger(__name__)


class ExplicitHybridRetriever:
    """
    Явный hybrid retrieval: sparse (FTS) + dense (pgvector) -> RRF -> candidates.
    Возвращает NodeWithScore для совместимости с reranker и ContextBuilder
    """

    def __init__(
        self,
        embed_model: BaseEmbedding,
        sparse_top_k: Optional[int] = None,
        dense_top_k: Optional[int] = None,
        fused_top_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
    ):
        s = get_settings()
        self._embed_model = embed_model
        self._sparse_top_k = sparse_top_k or s.sparse_top_k
        self._dense_top_k = dense_top_k or s.dense_top_k
        self._fused_top_k = fused_top_k or s.fused_top_k
        self._rrf_k = rrf_k or s.rrf_k

    def get_index(self):
        """Для совместимости с ContextBuilder — возвращает VectorStoreIndex"""
        from src.knowledge.indexer import get_vector_store
        from llama_index.core import VectorStoreIndex
        vs = get_vector_store()
        return VectorStoreIndex.from_vector_store(
            vector_store=vs,
            embed_model=self._embed_model,
        )

    async def retrieve_async(
        self,
        query: str,
        top_k: Optional[int] = None,
        trace_id: Optional[str] = None,
    ) -> tuple[list[NodeWithScore], dict[str, float]]:
        """
        Async retrieval. Возвращает (nodes, stage_timings).
        stage_timings: sparse_ms, dense_ms, fuse_ms
        """
        t0 = time.perf_counter()
        stage_timings: dict[str, float] = {}
        mode = get_settings().hybrid_mode

        async def _sparse():
            return await sparse_retrieve(
                query, top_k=self._sparse_top_k, trace_id=trace_id
            )

        async def _dense():
            return await dense_retrieve(
                query,
                embed_model=self._embed_model,
                top_k=self._dense_top_k,
                trace_id=trace_id,
            )

        fused: list[FusedResult]

        if mode == "dense_only":
            dense_list = await _dense()
            t_parallel_done = time.perf_counter()
            parallel_ms = (t_parallel_done - t0) * 1000
            stage_timings["sparse_ms"] = 0.0
            stage_timings["dense_ms"] = parallel_ms
            fused = [
                FusedResult(
                    chunk_id=cid,
                    fused_score=sc,
                    provenance="dense",
                    text=t,
                    metadata=m,
                    sparse_score=None,
                    dense_score=sc,
                )
                for cid, t, m, sc in dense_list[: self._fused_top_k]
            ]
        elif mode == "sparse_only":
            sparse_list = await _sparse()
            t_parallel_done = time.perf_counter()
            parallel_ms = (t_parallel_done - t0) * 1000
            stage_timings["sparse_ms"] = parallel_ms
            stage_timings["dense_ms"] = 0.0
            fused = [
                FusedResult(
                    chunk_id=cid,
                    fused_score=sc,
                    provenance="sparse",
                    text=t,
                    metadata=m,
                    sparse_score=sc,
                    dense_score=None,
                )
                for cid, t, m, sc in sparse_list[: self._fused_top_k]
            ]
        else:
            sparse_list, dense_list = await asyncio.gather(_sparse(), _dense())
            t_parallel_done = time.perf_counter()
            parallel_ms = (t_parallel_done - t0) * 1000
            stage_timings["sparse_ms"] = parallel_ms
            stage_timings["dense_ms"] = parallel_ms
            fused = rrf_fuse(
                sparse_list, dense_list, k=self._rrf_k, top_n=self._fused_top_k
            )

        stage_timings["fuse_ms"] = (time.perf_counter() - t_parallel_done) * 1000

        logger.info(
            "explicit_hybrid mode=%s trace_id=%s fused=%d top_fused=%s provenance=%s",
            mode,
            trace_id or "n/a",
            len(fused),
            [f.chunk_id for f in fused[:5]],
            [f.provenance for f in fused[:5]],
        )

        nodes = _fused_to_nodes(fused)
        return nodes, stage_timings


def _fused_to_nodes(fused: list[FusedResult]) -> list[NodeWithScore]:
    """Преобразует FusedResult в NodeWithScore для reranker"""
    out = []
    for f in fused:
        meta = dict(f.metadata) if f.metadata else {}
        meta["retrieval_sparse_score"] = f.sparse_score
        meta["retrieval_dense_score"] = f.dense_score
        meta["retrieval_rrf_score"] = f.fused_score
        meta["retrieval_provenance"] = f.provenance
        node = TextNode(text=f.text, metadata=meta)
        node.id_ = f.chunk_id
        out.append(NodeWithScore(node=node, score=f.fused_score))
    return out
