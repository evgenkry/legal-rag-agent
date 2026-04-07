"""Инициализация aiogram бота"""

import asyncio
import logging

from aiogram import Bot, Dispatcher

from src.api.dependencies import get_rag_service
from src.bot.handlers import router, set_rag_service
from src.core.config import get_settings

logger = logging.getLogger(__name__)


def run_polling() -> None:
    """Запускает бота в режиме long polling."""
    settings = get_settings()
    if not settings.telegram_bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN is required for bot")

    async def main():
        from src.db.pool import init_pool, close_pool
        await init_pool()
        try:
            rag = get_rag_service()
            set_rag_service(rag)
            bot = Bot(token=settings.telegram_bot_token)
            dp = Dispatcher()
            dp.include_router(router)
            await dp.start_polling(bot)
        finally:
            await close_pool()

    asyncio.run(main())
