"""Логирование взаимодействий"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.db.interaction_logs import insert_interaction_log_async

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


def log_interaction(
    user_id: str,
    interaction_type: str,
    question: str,
    retrieved_chunk_ids: list[str],
    used_chunk_ids: list[str],
    sources_found: bool = False,
    llm_latency_ms: Optional[float] = None,
    retriever_latency_ms: Optional[float] = None,
    model_used: Optional[str] = None,
    trace_id: Optional[str] = None,
    rewritten_query: Optional[str] = None,
    reranked_chunk_ids: Optional[list[str]] = None,
    stage_timings: Optional[dict[str, float]] = None,
    top_scores: Optional[list[float]] = None,
    refusal_reason: Optional[str] = None,
    answer_full_text: Optional[str] = None,
    candidates_json: Optional[list] = None,
    llm_context_json: Optional[list] = None,
    agent_outcome: Optional[str] = None,
    agent_trace_json: Optional[dict] = None,
    from_cache: bool = False,
) -> None:
    """Логирует взаимодействие в файл, stdout и PostgreSQL"""
    timestamp = datetime.utcnow().isoformat()
    entry: dict[str, Any] = {
        "user_id": user_id,
        "timestamp": timestamp,
        "interaction_type": interaction_type,
        "question": question[:500],
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "used_chunk_ids": used_chunk_ids,
        "sources_found": sources_found,
        "llm_latency_ms": llm_latency_ms,
        "retriever_latency_ms": retriever_latency_ms,
        "model_used": model_used,
    }
    if trace_id:
        entry["trace_id"] = trace_id
    if refusal_reason:
        entry["refusal_reason"] = refusal_reason
    if agent_outcome:
        entry["agent_outcome"] = agent_outcome
    if from_cache:
        entry["from_cache"] = True

    log_msg = (
        f"INTERACTION | user={user_id} | type={interaction_type} | "
        f"q={question[:80]}... | retrieved={len(retrieved_chunk_ids)} | "
        f"used={len(used_chunk_ids)} | found={sources_found}"
    )
    if trace_id:
        log_msg += f" | trace={trace_id[:8]}"
    if refusal_reason:
        log_msg += f" | refusal={refusal_reason}"
    if from_cache:
        log_msg += " | cache_hit"
    logger.info(log_msg)

    # для резервного хранения
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "interactions.log"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("Failed to write interaction log to file: %s", e)

    # postgresql
    async def _write_db() -> None:
        try:
            await insert_interaction_log_async(
                user_id=user_id,
                timestamp=timestamp,
                interaction_type=interaction_type,
                question=question[:10000],
                retrieved_chunk_ids=retrieved_chunk_ids,
                used_chunk_ids=used_chunk_ids,
                sources_found=sources_found,
                llm_latency_ms=llm_latency_ms,
                retriever_latency_ms=retriever_latency_ms,
                model_used=model_used,
                trace_id=trace_id,
                rewritten_query=rewritten_query,
                reranked_chunk_ids=reranked_chunk_ids,
                stage_timings_json=stage_timings,
                top_scores_json=top_scores,
                refusal_reason=refusal_reason,
                answer_full_text=answer_full_text,
                candidates_json=candidates_json,
                llm_context_json=llm_context_json,
                agent_outcome=agent_outcome,
                agent_trace_json=agent_trace_json,
            )
        except Exception as e:
            logger.warning("Async log write failed: %s", e)

    try:
        asyncio.get_running_loop()
        asyncio.create_task(_write_db())
    except RuntimeError:
        asyncio.run(_write_db())
