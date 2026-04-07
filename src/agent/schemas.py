"""Pydantic-схемы для LLM-агента."""

from typing import Optional

from pydantic import BaseModel, Field


class LegalQueryAgentResult(BaseModel):
    """Решение агента нормализации запроса."""

    needs_rewrite: bool = Field(description="Нужно ли переписать вопрос для retrieval")
    preserved_facts: list[str] = Field(
        default_factory=list,
        description="Извлечённые факты: субъекты, сроки, обстоятельства",
    )
    legal_query: str = Field(description="Строка запроса для поиска по базе знаний")
    rationale: str = Field(default="", description="Краткое обоснование для логов")


class ClaimRecord(BaseModel):
    """Одно утверждение из ответа и статус опоры на контекст."""

    text: str
    supported: bool
    evidence_chunk_indices: list[int] = Field(
        default_factory=list,
        description="Номера фрагментов [1..n], на которых основано утверждение; пусто если не поддержано",
    )


class AnswerVerifierResult(BaseModel):
    """Итог верификации черновика ответа."""

    claims: list[ClaimRecord] = Field(default_factory=list)
    revised_answer_body: str = Field(
        default="",
        description="Текст ответа после удаления/смягчения неподдержанных утверждений",
    )
    substantive: bool = Field(
        description="True, если остаётся содержательный ответ по вопросу пользователя"
    )
