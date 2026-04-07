#!/usr/bin/env python3
"""Запуск Telegram-бота"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import get_settings
from src.core.logging_config import setup_logging

setup_logging(level=get_settings().log_level)

from src.bot.bot import run_polling

if __name__ == "__main__":
    run_polling()
