"""Кнопки в чате Телеграм-бота для пользователей"""

from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton


def get_answer_keyboard() -> InlineKeyboardMarkup:
    """Кнопки после ответа: Задать новый вопрос / Уточнить вопрос."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(
                text="🆕 Задать новый вопрос (сброс контекста)",
                callback_data="new_question",
            ),
        ],
        [
            InlineKeyboardButton(
                text="💬 Уточнить вопрос",
                callback_data="clarify",
            ),
        ],
    ])
