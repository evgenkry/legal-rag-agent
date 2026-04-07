"""Reciprocal Rank Fusion (RRF)"""

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

Provenance = Literal["sparse", "dense", "both"]


@dataclass
class FusedResult:
    """Результат после RRF: chunk_id, fused_score, provenance, исходные скоры веток"""

    chunk_id: str
    fused_score: float
    provenance: Provenance
    text: str
    metadata: dict
    sparse_score: float | None = None
    dense_score: float | None = None


def rrf_fuse(
    sparse_list: list[tuple[str, str, dict, float]],
    dense_list: list[tuple[str, str, dict, float]],
    k: int = 60,
    top_n: int = 20,
) -> list[FusedResult]:
    """
    Слияние sparse и dense списков через Reciprocal Rank Fusion.
    Входные списки: [(chunk_id, text, metadata, score), ...]
    """
    scores: dict[str, float] = {}
    texts: dict[str, str] = {}
    metadata_by_id: dict[str, dict] = {}
    seen_in: dict[str, set[str]] = {}  # chunk_id -> {"sparse", "dense"}
    sparse_scores: dict[str, float] = {}
    dense_scores: dict[str, float] = {}

    for rank, (cid, text, meta, sc) in enumerate(sparse_list, start=1):
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        texts[cid] = text
        metadata_by_id[cid] = meta
        seen_in.setdefault(cid, set()).add("sparse")
        sparse_scores[cid] = float(sc)

    for rank, (cid, text, meta, sc) in enumerate(dense_list, start=1):
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        texts[cid] = text
        metadata_by_id[cid] = meta
        seen_in.setdefault(cid, set()).add("dense")
        dense_scores[cid] = float(sc)

    def provenance(s: set[str]) -> Provenance:
        if "sparse" in s and "dense" in s:
            return "both"
        if "sparse" in s:
            return "sparse"
        return "dense"

    fused = [
        FusedResult(
            chunk_id=cid,
            fused_score=sc,
            provenance=provenance(seen_in[cid]),
            text=texts[cid],
            metadata=metadata_by_id[cid],
            sparse_score=sparse_scores.get(cid),
            dense_score=dense_scores.get(cid),
        )
        for cid, sc in sorted(scores.items(), key=lambda x: -x[1])
    ][:top_n]

    return fused
