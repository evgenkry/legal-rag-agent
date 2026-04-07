"""Query Rewriting — переформулировка на юридический язык"""

import logging
from typing import Optional

from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """Переформулируй следующий вопрос пользователя на формальный юридический язык, 
на котором написан Трудовой кодекс Российской Федерации. 
Используй термины ТК РФ: "работник", "работодатель", "трудовой договор", "отпуск", "увольнение" и т.п.
Сохрани суть вопроса. Ответь только переформулированным вопросом, без пояснений.

Вопрос пользователя: {question}

Переформулированный вопрос:"""


class QueryRewriter:
    """Переформулирует запрос на юридический язык"""

    def __init__(self, llm: Optional[LLM] = None):
        self._llm = llm

    def set_llm(self, llm: LLM) -> None:
        self._llm = llm

    async def rewrite(self, query: str) -> str:
        """Переформулирует запрос. При ошибке возвращает оригинал"""
        if not self._llm:
            return query

        try:
            response = await self._llm.acomplete(
                REWRITE_PROMPT.format(question=query)
            )
            rewritten = str(response).strip()
            if rewritten and len(rewritten) > 5:
                logger.debug("Query rewritten: %s -> %s", query[:50], rewritten[:50])
                return rewritten
        except Exception as e:
            logger.warning("Query rewrite failed: %s", e)
        return query

    def rewrite_sync(self, query: str) -> str:
        """Синхронная версия (fallback)"""
        if not self._llm:
            return query
        try:
            response = self._llm.complete(REWRITE_PROMPT.format(question=query))
            rewritten = str(response).strip()
            return rewritten if rewritten else query
        except Exception as e:
            logger.warning("Query rewrite failed: %s", e)
            return query
