"""Hybrid Retriever: BM25 + pgvector, RRF"""

import asyncio
import logging
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from src.core.config import get_settings
from src.knowledge.indexer import get_vector_store

logger = logging.getLogger(__name__)


def get_retriever(embed_model: BaseEmbedding):
    """retriever по RAG_RETRIEVAL_MODE и hybrid_mode"""
    from src.rag.explicit_hybrid_retriever import ExplicitHybridRetriever

    settings = get_settings()
    if settings.hybrid_mode == "llamaindex":
        return HybridRetriever(embed_model)
    if settings.rag_retrieval_mode == "explicit_hybrid":
        return ExplicitHybridRetriever(embed_model=embed_model)
    return HybridRetriever(embed_model)


class HybridRetriever:
    """Гибридный поиск: semantic + full-text (pgvector hybrid)"""

    def __init__(
        self,
        embed_model,
        top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
    ):
        settings = get_settings()
        self._top_k = top_k or settings.retrieval_top_k
        self._sparse_top_k = sparse_top_k or self._top_k
        self._embed_model = embed_model
        self._index: Optional[VectorStoreIndex] = None

    def _get_index(self) -> VectorStoreIndex:
        if self._index is None:
            vector_store = get_vector_store()
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self._embed_model,
            )
        return self._index

    def get_index(self) -> VectorStoreIndex:
        """Доступ к индексу (для ContextBuilder)"""
        return self._get_index()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[NodeWithScore]:
        """Гибридный поиск (vector + sparse в PGVectorStore)"""
        index = self._get_index()
        k = top_k or self._top_k

        retriever = index.as_retriever(
            similarity_top_k=k,
            vector_store_query_mode="hybrid",
            sparse_top_k=self._sparse_top_k,
        )

        bundle = QueryBundle(query_str=query)
        nodes = retriever.retrieve(bundle)
        return nodes

    async def retrieve_async(
        self,
        query: str,
        top_k: Optional[int] = None,
        trace_id: Optional[str] = None,
    ) -> tuple[list[NodeWithScore], dict[str, float]]:
        del trace_id
        nodes = await asyncio.to_thread(self.retrieve, query, top_k)
        return nodes, {}
