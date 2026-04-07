"""Connection pool для asyncpg (логирование взаимодействий)."""

import logging
from typing import Optional

import asyncpg

from src.core.config import get_settings

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None


def _get_dsn() -> str:
    settings = get_settings()
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://")


async def init_pool(min_size: int = 1, max_size: int = 5) -> Optional[asyncpg.Pool]:
    """Инициализирует connection pool. Возвращает pool или None при ошибке."""
    global _pool
    if _pool is not None:
        return _pool
    try:
        dsn = _get_dsn()
        _pool = await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size, command_timeout=5)
        logger.info("Asyncpg pool initialized (min=%d, max=%d)", min_size, max_size)
        return _pool
    except Exception as e:
        logger.warning("Failed to init asyncpg pool: %s", e)
        return None


async def close_pool() -> None:
    """Закрывает connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Asyncpg pool closed")


def get_pool() -> Optional[asyncpg.Pool]:
    """Возвращает текущий pool или None."""
    return _pool
