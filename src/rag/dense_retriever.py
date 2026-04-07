"""Dense retriever via pgvector (cosine similarity)"""

import asyncio
import logging
from typing import Any, Optional

from llama_index.core.embeddings import BaseEmbedding

from src.core.config import get_settings
from src.db.pool import get_pool
from src.knowledge.indexer import INDEX_TABLE

logger = logging.getLogger(__name__)


async def dense_retrieve(
    query: str,
    embed_model: BaseEmbedding,
    top_k: Optional[int] = None,
    trace_id: Optional[str] = None,
) -> list[tuple[str, str, dict[str, Any], float]]:
    """
    Dense retrieval через pgvector.
    Возвращает список (chunk_id, text, metadata, score_dense).
    score_dense = 1 - distance (cosine distance)
    """
    settings = get_settings()
    k = top_k or settings.dense_top_k
    pool = get_pool()
    if not pool:
        raise RuntimeError("Asyncpg pool not initialized")

    emb = await asyncio.to_thread(embed_model.get_query_embedding, query)
    emb_str = "[" + ",".join(str(x) for x in emb) + "]"

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT node_id, text, metadata_, 1 - (embedding <=> $1::vector) AS score_dense
            FROM """ + INDEX_TABLE + """
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            emb_str,
            k,
        )

    result = []
    for r in rows:
        node_id = r["node_id"] or str(r.get("id", ""))
        text = r["text"] or ""
        meta = r.get("metadata_") or {}
        if isinstance(meta, str):
            import json

            try:
                meta = json.loads(meta) if meta else {}
            except Exception:
                meta = {}
        score = float(r.get("score_dense", 0))
        result.append((node_id, text, meta, score))

    logger.info(
        "dense_retrieve trace_id=%s top_k=%d got=%d top_ids=%s scores=%s",
        trace_id or "n/a",
        k,
        len(result),
        [x[0] for x in result[:5]],
        [round(x[3], 4) for x in result[:5]],
    )
    return result
