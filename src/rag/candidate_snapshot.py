"""Сериализация кандидатов и контекста LLM для логирования и анализа"""

from typing import Any

from llama_index.core.schema import NodeWithScore


def _chunk_id(node) -> str:
    return str(
        getattr(node, "node_id", None)
        or getattr(node, "id_", "")
        or ""
    )


def nodes_to_candidate_records(nodes: list[NodeWithScore]) -> list[dict[str, Any]]:
    """Кандидаты после retrieval + rerank с доступными скорами"""
    out: list[dict[str, Any]] = []
    for n in nodes:
        meta = dict(n.node.metadata or {})
        rid = _chunk_id(n.node)
        out.append(
            {
                "chunk_id": rid,
                "article": meta.get("article"),
                "faq_id": meta.get("faq_id"),
                "source_url": meta.get("source_url"),
                "source": meta.get("source"),
                "chunk_role": meta.get("chunk_role"),
                "full_citation": meta.get("full_citation"),
                "sparse_score": meta.get("retrieval_sparse_score"),
                "dense_score": meta.get("retrieval_dense_score"),
                "rrf_score": meta.get("retrieval_rrf_score"),
                "rerank_score": meta.get("rerank_score"),
                "node_score": getattr(n, "score", None),
            }
        )
    return out


def nodes_to_llm_context_records(nodes: list[NodeWithScore]) -> list[dict[str, Any]]:
    """Чанки, фактически переданные в LLM после расширения контекста"""
    out: list[dict[str, Any]] = []
    for n in nodes:
        meta = dict(n.node.metadata or {})
        rid = _chunk_id(n.node)
        out.append(
            {
                "chunk_id": rid,
                "article": meta.get("article"),
                "faq_id": meta.get("faq_id"),
                "source_url": meta.get("source_url"),
                "source": meta.get("source"),
                "chunk_role": meta.get("chunk_role"),
                "full_citation": meta.get("full_citation"),
            }
        )
    return out
