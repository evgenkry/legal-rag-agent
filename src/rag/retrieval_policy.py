"""Политика retrieval на основе юридичееской силы документов (закон или разъяснения Роструда)"""

from typing import Any


def resolve_source_type(metadata: dict[str, Any] | None) -> str:
    """Maps existing source metadata to target source_type axis."""
    if not isinstance(metadata, dict):
        return "explanation"
    source_type = str(metadata.get("source_type") or "").strip().lower()
    if source_type in {"law", "explanation"}:
        return source_type
    source = str(metadata.get("source") or "").strip().lower()
    if source == "tkrf":
        return "law"
    return "explanation"


def apply_law_explanation_policy(
    results: list,
    *,
    top_k: int,
    law_boost: float,
    law_min: int,
    explanation_min: int,
) -> list:
    """
    Ранжирование документов с учетом юридической силы
    """
    if not results:
        return []

    for doc in results:
        meta = getattr(getattr(doc, "node", None), "metadata", {}) or {}
        source_type = resolve_source_type(meta)
        meta["source_type"] = source_type
        setattr(getattr(doc, "node", None), "metadata", meta)
        score = getattr(doc, "score", 0.0) or 0.0
        if source_type == "law":
            setattr(doc, "score", float(score) + law_boost)

    law_docs = [
        d for d in results if resolve_source_type(getattr(d.node, "metadata", {})) == "law"
    ]
    explanation_docs = [
        d
        for d in results
        if resolve_source_type(getattr(d.node, "metadata", {})) == "explanation"
    ]
    law_docs.sort(key=lambda x: getattr(x, "score", 0.0) or 0.0, reverse=True)
    explanation_docs.sort(key=lambda x: getattr(x, "score", 0.0) or 0.0, reverse=True)

    if not law_docs:
        return explanation_docs[:top_k]
    if not explanation_docs:
        return law_docs[:top_k]

    final_docs = law_docs[:law_min] + explanation_docs[:explanation_min]
    remaining = sorted(
        results, key=lambda x: getattr(x, "score", 0.0) or 0.0, reverse=True
    )
    for doc in remaining:
        if doc not in final_docs:
            final_docs.append(doc)
        if len(final_docs) >= top_k:
            break
    return final_docs[:top_k]
