"""Извлечение JSON из ответа LLM"""

import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def extract_json_object(raw: str) -> dict | None:
    """Достаёт первый валидный JSON-объект из текста (в т.ч. обёрнутый в ```json)."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    while start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : i + 1]
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def parse_llm_json(raw: str, model: type[T]) -> T | None:
    """Парсит ответ LLM в Pydantic-модель."""
    data = extract_json_object(raw)
    if data is None:
        logger.warning("LLM JSON parse failed, raw prefix: %s", raw[:200])
        return None
    try:
        return model.model_validate(data)
    except Exception as e:
        logger.warning("Pydantic validate failed: %s", e)
        return None
