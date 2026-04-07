"""Модели ответов"""

from typing import Optional

from pydantic import BaseModel, Field


class QueryResponse(BaseModel):
    """Ответ на RAG-запрос"""

    answer: str = Field(..., description="Ответ модели")
    citations: list[str] = Field(default_factory=list, description="Цитаты в формате 'ст. N ТК'")
    sources_found: bool = Field(..., description="Найдены ли релевантные источники")
    chunks_used: list[str] = Field(default_factory=list, description="ID чанков, использованных для ответа")


class RAGResponse(BaseModel):
    """Полный ответ RAG pipeline"""

    answer: str
    citations: list[str] = []
    sources_found: bool
    chunks_used: list[str] = []
    retrieved_chunk_ids: list[str] = []
    latency_ms: Optional[float] = None
