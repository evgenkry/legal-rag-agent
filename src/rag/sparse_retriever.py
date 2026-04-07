"""Sparse retriever via Postgres Full-Text Search"""

import json
import logging
from typing import Any, Optional

from src.core.config import get_settings
from src.db.pool import get_pool
from src.knowledge.indexer import INDEX_TABLE

logger = logging.getLogger(__name__)


async def sparse_retrieve(
    query: str,
    top_k: Optional[int] = None,
    trace_id: Optional[str] = None,
) -> list[tuple[str, str, dict[str, Any], float]]:
    """
    Sparse retrieval через Postgres FTS.
    Возвращает список (chunk_id, text, metadata, score_sparse).
    score_sparse = ts_rank_cd
    """
    settings = get_settings()
    k = top_k or settings.sparse_top_k
    pool = get_pool()
    if not pool:
        raise RuntimeError("Asyncpg pool not initialized")

    # plainto_tsquery для пользовательского запроса
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT node_id, text, metadata_,
                   ts_rank_cd(text_search_tsv, plainto_tsquery('russian', $1)) AS score_sparse
            FROM """
            + INDEX_TABLE
            + """
            WHERE text_search_tsv @@ plainto_tsquery('russian', $1)
            ORDER BY score_sparse DESC
            LIMIT $2
            """,
            query,
            k,
        )

    result = []
    for r in rows:
        node_id = r["node_id"] or str(r.get("id", ""))
        text = r["text"] or ""
        meta = r.get("metadata_") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta) if meta else {}
            except Exception:
                meta = {}
        score = float(r.get("score_sparse", 0))
        result.append((node_id, text, meta, score))

    logger.info(
        "sparse_retrieve trace_id=%s top_k=%d got=%d top_ids=%s scores=%s",
        trace_id or "n/a",
        k,
        len(result),
        [x[0] for x in result[:5]],
        [round(x[3], 4) for x in result[:5]],
    )
    return result
