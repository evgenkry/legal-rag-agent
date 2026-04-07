"""Таблица и операции для логов взаимодействий"""

import asyncio
import json
import logging
from datetime import datetime, timezone

import asyncpg

from src.db.pool import get_pool

logger = logging.getLogger(__name__)

INSERT_TIMEOUT = 5.0
MAX_RETRIES = 2

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS interaction_logs (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    interaction_type VARCHAR(50) NOT NULL,
    question TEXT NOT NULL,
    retrieved_chunk_ids JSONB NOT NULL DEFAULT '[]',
    reranked_chunk_ids JSONB NOT NULL DEFAULT '[]',
    used_chunk_ids JSONB NOT NULL DEFAULT '[]',
    sources_found BOOLEAN NOT NULL DEFAULT FALSE,
    llm_latency_ms FLOAT,
    retriever_latency_ms FLOAT,
    model_used VARCHAR(255),
    trace_id VARCHAR(64),
    rewritten_query TEXT,
    stage_timings_json JSONB,
    top_scores_json JSONB,
    refusal_reason VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS ix_interaction_logs_user_id ON interaction_logs(user_id);
CREATE INDEX IF NOT EXISTS ix_interaction_logs_timestamp ON interaction_logs(timestamp);
CREATE INDEX IF NOT EXISTS ix_interaction_logs_interaction_type ON interaction_logs(interaction_type);
CREATE INDEX IF NOT EXISTS ix_interaction_logs_trace_id ON interaction_logs(trace_id);
"""

_ALTER_ADD_COLUMNS = [
    "ALTER TABLE interaction_logs ADD COLUMN IF NOT EXISTS reranked_chunk_ids JSONB NOT NULL DEFAULT '[]'",
    "ALTER TABLE interaction_logs ADD COLUMN IF NOT EXISTS trace_id VARCHAR(64)",
    "ALTER TABLE interaction_logs ADD COLUMN IF NOT EXISTS rewritten_query TEXT",
    "ALTER TABLE interaction_logs ADD COLUMN IF NOT EXISTS stage_timings_json JSONB",
    "ALTER TABLE interaction_logs ADD COLUMN IF NOT EXISTS top_scores_json JSONB",
    "ALTER TABLE interaction_logs ADD COLUMN IF NOT EXISTS refusal_reason VARCHAR(100)",
    "ALTER TABLE interaction_logs ADD COLUMN IF NOT EXISTS answer_full_text TEXT",
    "ALTER TABLE interaction_logs ADD COLUMN IF NOT EXISTS candidates_json JSONB",
    "ALTER TABLE interaction_logs ADD COLUMN IF NOT EXISTS llm_context_json JSONB",
    "ALTER TABLE interaction_logs ADD COLUMN IF NOT EXISTS agent_outcome VARCHAR(40)",
    "ALTER TABLE interaction_logs ADD COLUMN IF NOT EXISTS agent_trace_json JSONB",
]


async def _ensure_table(conn: asyncpg.Connection) -> None:
    await conn.execute(_CREATE_TABLE_SQL)
    for sql in _ALTER_ADD_COLUMNS:
        try:
            await conn.execute(sql)
        except Exception:
            pass


async def insert_interaction_log_async(
    user_id: str,
    timestamp: str,
    interaction_type: str,
    question: str,
    retrieved_chunk_ids: list[str],
    used_chunk_ids: list[str],
    sources_found: bool = False,
    llm_latency_ms: float | None = None,
    retriever_latency_ms: float | None = None,
    model_used: str | None = None,
    trace_id: str | None = None,
    rewritten_query: str | None = None,
    stage_timings_json: dict | None = None,
    top_scores_json: dict | list | None = None,
    refusal_reason: str | None = None,
    reranked_chunk_ids: list[str] | None = None,
    answer_full_text: str | None = None,
    candidates_json: list | dict | None = None,
    llm_context_json: list | dict | None = None,
    agent_outcome: str | None = None,
    agent_trace_json: dict | list | None = None,
) -> None:
    """Асинхронная вставка лога в PostgreSQL через connection pool."""
    pool = get_pool()
    if pool is None:
        logger.warning("Log pool not available, skipping DB insert")
        return

    reranked = reranked_chunk_ids or []
    # asyncpg ожидает datetime для timestamptz, не строку
    if isinstance(timestamp, str):
        try:
            ts_value = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            ts_value = datetime.now(timezone.utc)
        if ts_value.tzinfo is None:
            ts_value = ts_value.replace(tzinfo=timezone.utc)
    else:
        ts_value = timestamp

    for attempt in range(MAX_RETRIES):
        try:
            async with pool.acquire() as conn:
                await _ensure_table(conn)
                await asyncio.wait_for(
                    conn.execute(
                        """
                        INSERT INTO interaction_logs (
                            user_id, timestamp, interaction_type, question,
                            retrieved_chunk_ids, reranked_chunk_ids, used_chunk_ids,
                            sources_found, llm_latency_ms, retriever_latency_ms, model_used,
                            trace_id, rewritten_query, stage_timings_json, top_scores_json, refusal_reason,
                            answer_full_text, candidates_json, llm_context_json,
                            agent_outcome, agent_trace_json
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb, $15::jsonb, $16,
                            $17, $18::jsonb, $19::jsonb, $20, $21::jsonb
                        )
                        """,
                        user_id[:255],
                        ts_value,
                        interaction_type[:50],
                        question[:10000],
                        json.dumps(retrieved_chunk_ids),
                        json.dumps(reranked),
                        json.dumps(used_chunk_ids),
                        sources_found,
                        llm_latency_ms,
                        retriever_latency_ms,
                        model_used[:255] if model_used else None,
                        trace_id[:64] if trace_id else None,
                        rewritten_query[:5000] if rewritten_query else None,
                        json.dumps(stage_timings_json) if stage_timings_json else "null",
                        json.dumps(top_scores_json) if top_scores_json else "null",
                        refusal_reason[:100] if refusal_reason else None,
                        (answer_full_text[:50000] if answer_full_text else None),
                        json.dumps(candidates_json) if candidates_json is not None else "null",
                        json.dumps(llm_context_json) if llm_context_json is not None else "null",
                        agent_outcome[:40] if agent_outcome else None,
                        json.dumps(agent_trace_json) if agent_trace_json is not None else None,
                    ),
                timeout=INSERT_TIMEOUT,
            )
            return
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning("Insert log retry %d: %s", attempt + 1, e)
            else:
                logger.warning("Failed to insert interaction log into DB: %s", e)
