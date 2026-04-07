"""Модели запросов"""

from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Запрос на RAG-ответ"""

    question: str = Field(..., min_length=1, description="Вопрос пользователя")
    user_id: Optional[str] = None
    chat_history: Optional[list[dict[str, str]]] = None
    reset_context: bool = False
    trace_id: Optional[str] = None
