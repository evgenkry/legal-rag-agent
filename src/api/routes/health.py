"""GET /health — проверка работоспособности."""

from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health() -> dict:
    """Health check."""
    return {"status": "ok", "service": "rag-assistant"}
