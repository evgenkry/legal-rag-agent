"""LLM-агент: решение о переписывании вопроса на юридический язык."""

import asyncio
import logging
from typing import Optional

from llama_index.core.llms import ChatMessage, LLM, MessageRole

from src.agent.json_llm import parse_llm_json
from src.agent.schemas import LegalQueryAgentResult
from src.core.config import get_settings

logger = logging.getLogger(__name__)

LEGAL_QUERY_SYSTEM = """Ты помощник для поиска по Трудовому кодексу РФ. Отвечай ТОЛЬКО одним JSON-объектом без пояснений и без markdown.

Схема JSON:
{
  "needs_rewrite": true/false,
  "preserved_facts": ["кратко перечисли неизменяемые факты: кто, что случилось, сроки, статусы"],
  "legal_query": "строка, которую использовать для полнотекстового и семантического поиска по базе знаний",
  "rationale": "одно короткое предложение"
}

Правила:
- Если вопрос уже сформулирован формально и близко к языку ТК РФ, поставь needs_rewrite=false и legal_query скопируй из вопроса пользователя дословно.
- Если вопрос бытовой или разговорный, поставь needs_rewrite=true и сформулируй legal_query так, как задал бы вопрос юрист (термины: работник, работодатель, трудовой договор, увольнение, отпуск и т.д.), НЕ добавляя новых фактов и НЕ делая юридических выводов.
- preserved_facts должны отражать только то, что явно сказано пользователем.
- legal_query — одна строка или короткий абзац без JSON внутри."""

LEGAL_QUERY_USER = """Вопрос пользователя:
{question}

JSON:"""


class LegalQueryAgent:
    """Адаптивное переписывание запроса для retrieval."""

    def __init__(self, llm: Optional[LLM] = None):
        self._llm = llm

    def set_llm(self, llm: LLM) -> None:
        self._llm = llm

    async def normalize(self, question: str) -> LegalQueryAgentResult:
        """Возвращает строку для поиска; при сбое — исходный вопрос."""
        q = (question or "").strip()
        if not q or not self._llm:
            return LegalQueryAgentResult(
                needs_rewrite=False,
                preserved_facts=[],
                legal_query=q,
                rationale="no_llm_or_empty",
            )

        prompt = LEGAL_QUERY_USER.format(question=q)
        settings = get_settings()
        try:
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=LEGAL_QUERY_SYSTEM),
                ChatMessage(role=MessageRole.USER, content=prompt),
            ]
            raw = ""
            for attempt in range(settings.llm_retry_attempts):
                try:
                    response = await asyncio.wait_for(
                        self._llm.achat(messages),
                        timeout=settings.llm_timeout_sec,
                    )
                    msg = getattr(response, "message", response)
                    raw = str(getattr(msg, "content", response) or "").strip()
                    break
                except Exception as e:
                    logger.warning("LegalQueryAgent attempt %d failed: %s", attempt + 1, e)
            if not raw:
                raise RuntimeError("legal_query_agent_empty_response")
        except Exception as e:
            logger.warning("LegalQueryAgent LLM call failed: %s", e)
            return LegalQueryAgentResult(
                needs_rewrite=False,
                preserved_facts=[],
                legal_query=q,
                rationale="llm_error",
            )

        parsed = parse_llm_json(raw, LegalQueryAgentResult)
        if parsed is None:
            return LegalQueryAgentResult(
                needs_rewrite=False,
                preserved_facts=[],
                legal_query=q,
                rationale="json_parse_failed",
            )

        lq = (parsed.legal_query or "").strip()
        if not lq:
            lq = q
        if not parsed.needs_rewrite:
            lq = q
        return LegalQueryAgentResult(
            needs_rewrite=parsed.needs_rewrite,
            preserved_facts=parsed.preserved_facts or [],
            legal_query=lq,
            rationale=(parsed.rationale or "").strip() or "ok",
        )
