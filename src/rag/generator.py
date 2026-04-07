"""LLM-генерация ответа с промптом"""

import asyncio
import logging
from typing import Optional

from llama_index.core.llms import ChatMessage, LLM, MessageRole

from src.core.config import get_settings
from src.rag.context_format import format_nodes_for_context
from src.rag.source_format import build_formatted_citations

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — дружелюбный юридический ассистент по трудовому праву РФ.
Отвечай на основе переданного контекста (ТК РФ и разъяснения Роструда).

Стиль ответа по существу (как у опытного юриста):
- Связный развёрнутый текст; в **теле** ответа указывай нормы естественным языком: «ч. 3 ст. 136 ТК РФ», «ст. 140 ТК» и т.п.
- Где уместна правоприменительная оценка — используй формулировки вроде «скорее всего», «можно говорить о…», «оснований нет», не выдавай выдуманные статьи.
- **Не дублируй** формулировку вопроса пользователя и не начинай с «Вы спросили…».

Структура:
1) Сначала краткий вывод по существу (1–3 предложения).
2) Затем развёрнутое обоснование с отсылками к нормам в тексте.

Тон: профессиональный, тёплый, с уместными эмодзи; обращение на «Вы».
Если в контексте нет релевантной информации — честно скажи об этом и предложи уточнить вопрос."""

USER_PROMPT = """Контекст (фрагменты ТК РФ и разъяснений):
{context}

Вопрос пользователя (не цитируй его дословно в ответе):
{question}

Сформулируй ответ по структуре из системной инструкции (краткий вывод, затем обоснование с нормами в тексте):"""

NO_CONTEXT_PROMPT = """В предоставленном контексте нет релевантной информации для ответа на вопрос.
Честно сообщи пользователю, что ответ не найден в базе знаний, и предложи уточнить вопрос."""


class Generator:
    """Генерация ответа через LLM"""

    def __init__(self, llm: Optional[LLM] = None):
        self._llm = llm

    def set_llm(self, llm: LLM) -> None:
        self._llm = llm

    def _format_context(self, nodes: list) -> str:
        return format_nodes_for_context(nodes)

    def _sources_from_nodes(self, nodes: list) -> list[str]:
        """Источники для блока «Источники» и поля citations (см. source_format)"""
        return build_formatted_citations(nodes)

    async def generate(
        self,
        question: str,
        nodes: list,
        chat_history: Optional[list[dict]] = None,
    ) -> tuple[str, list[str]]:
        """
        Генерирует ответ. Возвращает (answer_text, citations).
        citations — отформатированный список источников (см. build_formatted_citations)
        """
        if not self._llm:
            return "Ошибка: LLM не настроен.", []

        context = self._format_context(nodes)
        has_context = bool(context.strip())

        if not has_context:
            prompt = NO_CONTEXT_PROMPT
        else:
            prompt = USER_PROMPT.format(context=context, question=question)

        messages = [ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)]
        if chat_history:
            limit = get_settings().context_window_size
            for m in chat_history[-limit:]:
                role = MessageRole.USER if m.get("role") == "user" else MessageRole.ASSISTANT
                messages.append(ChatMessage(role=role, content=m.get("content", "")))
        messages.append(ChatMessage(role=MessageRole.USER, content=prompt))

        settings = get_settings()
        last_error: Exception | None = None
        for attempt in range(settings.llm_retry_attempts):
            try:
                response = await asyncio.wait_for(
                    self._llm.achat(messages), timeout=settings.llm_timeout_sec
                )
                msg = getattr(response, "message", response)
                answer_text = str(getattr(msg, "content", response) or "").strip()
                citations = self._sources_from_nodes(nodes)
                return answer_text, citations
            except Exception as e:
                last_error = e
                logger.warning("Generation attempt %d failed: %s", attempt + 1, e)
        logger.exception("Generation failed after retries: %s", last_error)
        return "Недостаточно данных", []
