"""Подмешивание полного ответа Роструда в контекст, если выжимка из ответа (вопрос + начало ответа) попали в retrieval"""

import logging
from typing import Callable

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

logger = logging.getLogger(__name__)


def expand_faq_context_nodes(
    nodes: list[NodeWithScore],
    get_index: Callable[[], VectorStoreIndex],
) -> list[NodeWithScore]:
    """
    Для каждого чанка с chunk_role=retrieval и faq_id подгружает chunk_role=context
    с тем же faq_id, если его ещё нет в списке. Дедупликация по node id
    """
    if not nodes:
        return nodes

    seen_ids: set[str] = set()
    for n in nodes:
        nid = getattr(n.node, "node_id", None) or getattr(n.node, "id_", None)
        if nid:
            seen_ids.add(str(nid))

    existing_faq_context: set[str] = set()
    for n in nodes:
        meta = n.node.metadata or {}
        if meta.get("chunk_role") == "context" and meta.get("faq_id"):
            existing_faq_context.add(str(meta["faq_id"]))

    faq_ids_for_merge: list[str] = []
    anchor_scores: dict[str, float] = {}
    for n in nodes:
        meta = n.node.metadata or {}
        if meta.get("chunk_role") != "retrieval":
            continue
        fid = meta.get("faq_id")
        if not fid:
            continue
        fid_s = str(fid)
        if fid_s in existing_faq_context:
            continue
        if fid_s not in faq_ids_for_merge:
            faq_ids_for_merge.append(fid_s)
        sc = getattr(n, "score", None)
        if sc is not None:
            anchor_scores[fid_s] = max(anchor_scores.get(fid_s, float("-inf")), float(sc))

    if not faq_ids_for_merge:
        return nodes

    index = get_index()
    extra: list[NodeWithScore] = []

    for fid in faq_ids_for_merge:
        try:
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(key="faq_id", value=fid),
                    MetadataFilter(key="chunk_role", value="context"),
                ]
            )
            retriever = index.as_retriever(similarity_top_k=2, filters=filters)
            found = retriever.retrieve(QueryBundle(query_str=f"faq {fid}"))
            for f in found:
                nid = getattr(f.node, "node_id", None) or getattr(f.node, "id_", None)
                if nid and str(nid) in seen_ids:
                    continue
                if nid:
                    seen_ids.add(str(nid))
                base_sc = anchor_scores.get(fid, 0.0)
                extra.append(
                    NodeWithScore(
                        node=f.node,
                        score=float(base_sc) * 0.99,
                    )
                )
                break
        except Exception as e:
            logger.debug("expand_faq_context: faq_id=%s failed: %s", fid, e)

    if not extra:
        return nodes
    return list(nodes) + extra
