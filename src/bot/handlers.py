"""Обработчики команд и сообщений в Телеграм-боте"""

import logging
import uuid

from aiogram import F, Router
from aiogram.types import CallbackQuery, Message

from src.bot.keyboards import get_answer_keyboard
from src.core.config import get_settings

logger = logging.getLogger(__name__)

router = Router()

# RAG service — инициализируется в bot.py
_rag_service = None


def set_rag_service(svc):
    global _rag_service
    _rag_service = svc


# Контекст диалога: user_id -> list[{role, content}]
_chat_context: dict[str, list[dict]] = {}


def _get_context_limit() -> int:
    return get_settings().context_window_size


def _estimate_dialog_tokens(messages: list[dict]) -> int:
    s = get_settings()
    total = sum(len(m.get("content", "") or "") for m in messages)
    return int(total / s.approx_chars_per_token) if s.approx_chars_per_token else total


def _get_context(user_id: str) -> list[dict]:
    return _chat_context.get(user_id, [])


def _add_to_context(user_id: str, role: str, content: str) -> None:
    limit = _get_context_limit()
    if user_id not in _chat_context:
        _chat_context[user_id] = []
    ctx = _chat_context[user_id]
    ctx.append({"role": role, "content": content})
    if len(ctx) > limit:
        _chat_context[user_id] = ctx[-limit:]


def _reset_context(user_id: str) -> None:
    _chat_context[user_id] = []


@router.message(F.text == "/start")
async def cmd_start(message: Message) -> None:
    """Приветствие, назначение, возможности."""
    welcome = (
        "👋 Здравствуйте!\n\n"
        "Я — LLM-агент для правовых справок. Специализируюсь на росийском трудовом праве. "
        "Отвечаю на вопросы на основе Трудового кодекса и разъяснений Роструда.\n\n"
        "💡 Задайте свой вопрос — я постараюсь дать чёткий ответ со ссылками на нормы ТК РФ."
    )
    await message.answer(welcome)


@router.callback_query(F.data == "new_question")
async def callback_new_question(callback: CallbackQuery) -> None:
    """Сброс контекста."""
    _reset_context(str(callback.from_user.id))
    await callback.answer("Контекст сброшен. Задайте новый вопрос.")
    await callback.message.answer("Контекст сброшен. Можете задать новый вопрос. ✨")


@router.callback_query(F.data == "clarify")
async def callback_clarify(callback: CallbackQuery) -> None:
    """Уточнение — контекст сохраняется."""
    await callback.answer("Уточните, пожалуйста, Ваш вопрос.")


@router.message(F.text)
async def handle_question(message: Message) -> None:
    """Обработка вопроса."""
    if _rag_service is None:
        await message.answer("Сервис временно недоступен. Попробуйте позже.")
        return
    user_id = str(message.from_user.id)
    question = message.text.strip()

    if not question:
        await message.answer("Пожалуйста, введите вопрос.")
        return

    settings = get_settings()
    chat_history = _get_context(user_id)
    tentative = chat_history + [{"role": "user", "content": question}]
    if _estimate_dialog_tokens(tentative) > settings.max_context_tokens_est:
        await message.answer(
            "К сожалению, объём переписки превышает допустимый лимит контекста. "
            "Нажмите «Задать новый вопрос (сброс контекста)» и начните заново "
            "или сократите историю диалога.",
            reply_markup=get_answer_keyboard(),
        )
        return

    # явное сообщение + индикатор набора
    await message.answer(
        "✅ Получил Ваш вопрос, анализирую материалы и формулирую ответ… ⏳"
    )
    await message.bot.send_chat_action(message.chat.id, "typing")

    trace_id = str(uuid.uuid4())
    try:
        result = await _rag_service.query(
            question=question,
            user_id=user_id,
            chat_history=chat_history if chat_history else None,
            interaction_type="clarify" if chat_history else "query",
            trace_id=trace_id,
        )

        answer = result["answer"]

        _add_to_context(user_id, "user", question)
        _add_to_context(user_id, "assistant", answer)

        await message.answer(answer, reply_markup=get_answer_keyboard())
    except Exception as e:
        logger.exception("RAG error: %s", e)
        await message.answer(
            "😔 Произошла ошибка при обработке вопроса. Попробуйте ещё раз или уточните формулировку.",
            reply_markup=get_answer_keyboard(),
        )
