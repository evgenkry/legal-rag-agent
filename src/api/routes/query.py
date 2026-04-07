"""POST /query — RAG запрос."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.dependencies import RAGServiceDep
from src.core.config import get_settings
from src.infra.rate_limit import rate_limiter
from src.models.query import QueryRequest
from src.models.response import QueryResponse

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    rag: RAGServiceDep,
    request: Request,
) -> QueryResponse:
    """Выполняет RAG-запрос по вопросу."""
    settings = get_settings()
    user_key = body.user_id or (request.client.host if request.client else "anonymous")
    if settings.enable_rate_limit:
        allowed = await rate_limiter.check_and_hit(
            user_key, limit=settings.rate_limit_per_minute, period_sec=60
        )
        if not allowed:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

    trace_id = body.trace_id or str(uuid.uuid4())
    result = await rag.query(
        question=body.question,
        user_id=body.user_id,
        chat_history=body.chat_history,
        interaction_type="clarify" if not body.reset_context else "query",
        trace_id=trace_id,
    )
    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        sources_found=result["sources_found"],
        chunks_used=result["chunks_used"],
    )
