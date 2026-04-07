"""Точка входа приложения"""

import asyncio
import logging
import sys

from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.api.routes import health, query, admin
from src.core.config import get_settings
from src.core.logging_config import setup_logging

settings = get_settings()
setup_logging(level=settings.log_level)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan: startup/shutdown"""
    from src.db.pool import init_pool, close_pool
    from src.infra.cache import cache_backend
    from src.infra.rate_limit import rate_limiter

    await init_pool()
    await cache_backend.connect()
    await rate_limiter.connect()
    yield
    await rate_limiter.close()
    await cache_backend.close()
    await close_pool()


app = FastAPI(
    title="LLM-агент для правовых справок",
    description="Modular RAG с гибридным поиском",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router, prefix="")
app.include_router(query.router, prefix="")
app.include_router(admin.router, prefix="")


def run_bot():
    """Запускает Telegram бота"""
    from src.bot.bot import run_polling
    run_polling()


def run_api():
    """Запускает FastAPI"""
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    # запускаем API по умолчанию, бот — отдельно или через env
    run_api()
