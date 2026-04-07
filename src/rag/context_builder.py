"""Расширение контекста связанными статьями по метаданным"""

import logging
from typing import Callable, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter

logger = logging.getLogger(__name__)

# ограничение на кол-во дополнительных чанков, чтобы не раздувать контекст
MAX_REFERENCE_CHUNKS = 5


class ContextBuilder:
    """Добавляет чанки связанных статей (references) в контекст"""

    def __init__(self, get_index: Callable[[], VectorStoreIndex]):
        self._get_index = get_index

    def _get_referenced_articles(self, nodes: list[NodeWithScore]) -> set[str]:
        """Собирает номера статей из references метаданных"""
        refs = set()
        for n in nodes:
            meta = n.node.metadata or {}
            refs.update(meta.get("references", []))
            refs.add(meta.get("article", ""))
        return {r for r in refs if r and str(r).replace(".", "").isdigit()}

    def _get_existing_articles(self, nodes: list[NodeWithScore]) -> set[str]:
        """Номера статей, уже присутствующих в nodes"""
        return {
            str(m.get("article", ""))
            for n in nodes
            for m in [n.node.metadata or {}]
            if m.get("article")
        }

    def expand(
        self,
        nodes: list[NodeWithScore],
        max_extra: int = MAX_REFERENCE_CHUNKS,
    ) -> list[NodeWithScore]:
        """
        Расширяет контекст чанками связанных статей (references).
        Возвращает исходные nodes + дополнительные по ссылкам, убираем дкубликаты
        """
        if not nodes:
            return nodes

        index = self._get_index()
        ref_articles = self._get_referenced_articles(nodes)
        existing_articles = self._get_existing_articles(nodes)
        to_fetch = ref_articles - existing_articles

        if not to_fetch:
            return nodes

        seen_ids: set[str] = set()
        for n in nodes:
            nid = getattr(n.node, "node_id", id(n.node))
            if nid:
                seen_ids.add(str(nid))

        extra: list[NodeWithScore] = []
        for article in list(to_fetch)[:max_extra]:
            try:
                filters = MetadataFilters(
                    filters=[MetadataFilter(key="article", value=article)]
                )
                retriever = index.as_retriever(
                    similarity_top_k=1,
                    filters=filters,
                )
                bundle = QueryBundle(query_str=f"статья {article}")
                found = retriever.retrieve(bundle)
                for f in found:
                    nid = getattr(f.node, "node_id", None)
                    if nid and str(nid) not in seen_ids:
                        seen_ids.add(str(nid))
                        extra.append(f)
                        if len(extra) >= max_extra:
                            break
            except Exception as e:
                logger.debug("ContextBuilder: fetch article %s failed: %s", article, e)
            if len(extra) >= max_extra:
                break

        result = list(nodes) + extra
        logger.debug(
            "ContextBuilder: expanded %d -> %d nodes (refs: %s)",
            len(nodes),
            len(result),
            list(to_fetch)[:5],
        )
        return result
