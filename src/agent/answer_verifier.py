"""Верификатор утверждений ответа по чанкам контекста."""

import asyncio
import logging
from typing import Optional

from llama_index.core.llms import ChatMessage, LLM, MessageRole

from src.agent.json_llm import extract_json_object
from src.agent.schemas import AnswerVerifierResult, ClaimRecord
from src.core.config import get_settings
from src.rag.context_format import format_nodes_for_context

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM = """Ты проверяешь юридический ответ ассистента на опору переданным фрагментам базы знаний.

Отвечай ТОЛЬКО одним JSON-объектом без markdown и без текста вне JSON.

ВАЖНО: корневой объект ОБЯЗАН содержать ровно эти три поля верхнего уровня: "claims", "revised_answer_body", "substantive".
Не возвращай один только объект утверждения без обёртки — всегда массив "claims" (можно с одним элементом).

Схема:
{
  "claims": [
    {
      "text": "формулировка утверждения из ответа",
      "supported": true/false,
      "evidence_chunk_indices": [1, 2]
    }
  ],
  "revised_answer_body": "итоговый текст ответа пользователю: удали предложения с неподдержанными фактами о нормах/сроках; допускается смягчить в осторожные формулировки («скорее всего», «возможно») только если смысл всё ещё следует из контекста; не добавляй новых статей и фактов",
  "substantive": true/false
}

Правила:
- supported=true только если утверждение явно вытекает из текста фрагментов с указанными номерами [i].
- Номера статей, сроки, обязанности без цитаты в фрагментах — supported=false.
- Если после правки не остаётся ответа на вопрос пользователя, substantive=false и revised_answer_body оставь пустой строкой.
- Не дублируй блок «Источники» — только тело ответа."""

VERIFIER_USER = """Фрагменты контекста (номер [i] и node_id соответствуют переданным чанкам):
{context}

Вопрос пользователя:
{question}

Черновик ответа ассистента (только тело, без списка источников):
{draft}

JSON:"""


def _coerce_evidence_indices(val: object) -> list[int]:
    if val is None:
        return []
    if isinstance(val, bool):
        return []
    if isinstance(val, int):
        return [val]
    out: list[int] = []
    if isinstance(val, list):
        for x in val:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                pass
    return out


def _normalize_claim_dict(d: dict) -> dict:
    if not isinstance(d, dict):
        return {"text": "", "supported": False, "evidence_chunk_indices": []}
    text = str(d.get("text", "")).strip()
    supported = bool(d.get("supported", False))
    ev = d.get("evidence_chunk_indices")
    if ev is None:
        ev = d.get("evidence_indices") or d.get("chunk_indices") or d.get("supporting_chunk_indices")
    return {
        "text": text,
        "supported": supported,
        "evidence_chunk_indices": _coerce_evidence_indices(ev),
    }


def _normalize_verifier_payload(data: object, draft_body: str) -> dict | None:
    """
    Приводит типичные ошибки формата JSON верификатора к схеме AnswerVerifierResult.
    Например, когда модель возвращает один объект claim вместо корневого объекта.
    """
    if data is None:
        return None
    draft = (draft_body or "").strip()
    if isinstance(data, list):
        claims_raw = [x for x in data if isinstance(x, dict)]
        if not claims_raw:
            return {"claims": [], "revised_answer_body": draft, "substantive": bool(draft)}
        return {
            "claims": [_normalize_claim_dict(c) for c in claims_raw],
            "revised_answer_body": draft,
            "substantive": bool(draft),
        }
    if not isinstance(data, dict):
        return None
    # Уже почти полная форма
    if "claims" in data and isinstance(data["claims"], list):
        out = {k: v for k, v in data.items()}
        out["claims"] = [_normalize_claim_dict(c) if isinstance(c, dict) else _normalize_claim_dict({}) for c in data["claims"]]
        rb = out.get("revised_answer_body")
        if rb is None or (isinstance(rb, str) and not rb.strip()):
            out["revised_answer_body"] = draft_body
        rb_str = str(out.get("revised_answer_body") or "").strip()
        if "substantive" not in out:
            out["substantive"] = bool(rb_str)
        return out
    # Один claim в корне (частая ошибка модели)
    if "text" in data and "revised_answer_body" not in data:
        has_claim_shape = "supported" in data or any(
            k in data for k in ("evidence_chunk_indices", "evidence_indices", "chunk_indices")
        )
        if has_claim_shape or len(data) <= 6:
            return {
                "claims": [_normalize_claim_dict(data)],
                "revised_answer_body": draft_body,
                "substantive": bool(draft),
            }
    # Только revised без claims
    if "revised_answer_body" in data and "claims" not in data:
        rb = str(data.get("revised_answer_body") or "").strip()
        sub = data.get("substantive")
        if sub is None:
            sub = bool(rb or draft)
        return {
            "claims": [],
            "revised_answer_body": rb or draft,
            "substantive": bool(sub),
        }
    return data


def _parse_verifier_response(raw: str, draft_body: str) -> AnswerVerifierResult | None:
    data = extract_json_object(raw)
    if data is None:
        logger.warning("AnswerVerifier JSON extract failed, raw prefix: %s", raw[:300])
        return None
    normalized = _normalize_verifier_payload(data, draft_body)
    if normalized is None:
        return None
    try:
        return AnswerVerifierResult.model_validate(normalized)
    except Exception as e:
        logger.warning("AnswerVerifier validate after normalize failed: %s", e)
        # Последняя попытка: claims вручную
        try:
            claims_raw = normalized.get("claims") if isinstance(normalized, dict) else None
            if isinstance(claims_raw, list):
                claims = [ClaimRecord.model_validate(_normalize_claim_dict(c)) for c in claims_raw if isinstance(c, dict)]
                rb = str(normalized.get("revised_answer_body") or draft_body).strip()
                sub = bool(normalized.get("substantive", bool(rb)))
                return AnswerVerifierResult(claims=claims, revised_answer_body=rb, substantive=sub)
        except Exception:
            pass
        return None


VERIFIER_REFUSAL = (
    "По переданным фрагментам базы знаний нельзя сформировать уверенный ответ на Ваш вопрос "
    "без риска неточностей.\n\n"
    "Попробуйте уточнить вопрос (роли, сроки, основание) или нажмите «Уточнить вопрос»."
)


class AnswerVerifier:
    """Проверка черновика: утверждения ↔ чанки, правка или отказ."""

    def __init__(self, llm: Optional[LLM] = None):
        self._llm = llm

    def set_llm(self, llm: LLM) -> None:
        self._llm = llm

    async def verify(
        self,
        question: str,
        draft_body: str,
        nodes: list,
    ) -> tuple[AnswerVerifierResult, str]:
        """
        Возвращает (результат, финальное тело ответа или текст отказа).
        При ошибке LLM/JSON — консервативный отказ (VERIFIER_REFUSAL).
        """
        q = (question or "").strip()
        body = (draft_body or "").strip()
        if not self._llm or not nodes:
            return (
                AnswerVerifierResult(
                    claims=[],
                    revised_answer_body=body,
                    substantive=bool(body),
                ),
                body,
            )

        context = format_nodes_for_context(nodes)
        if not context.strip():
            return (
                AnswerVerifierResult(claims=[], revised_answer_body=body, substantive=bool(body)),
                body,
            )

        user_msg = VERIFIER_USER.format(context=context, question=q, draft=body)
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=VERIFIER_SYSTEM),
            ChatMessage(role=MessageRole.USER, content=user_msg),
        ]
        settings = get_settings()
        raw = ""
        error: Exception | None = None
        max_attempts = settings.llm_retry_attempts
        for attempt in range(max_attempts):
            try:
                response = await asyncio.wait_for(
                    self._llm.achat(messages), timeout=settings.llm_timeout_sec
                )
                msg = getattr(response, "message", response)
                raw = str(getattr(msg, "content", response) or "").strip()
                error = None
                break
            except Exception as e:
                error = e
                logger.warning(
                    "AnswerVerifier attempt %d/%d failed (%s): %r",
                    attempt + 1,
                    max_attempts,
                    type(e).__name__,
                    e,
                )
        if error is not None:
            logger.warning(
                "AnswerVerifier LLM failed after %d attempts (%s): %r",
                max_attempts,
                type(error).__name__,
                error,
            )
            return (
                AnswerVerifierResult(
                    claims=[],
                    revised_answer_body="",
                    substantive=False,
                ),
                VERIFIER_REFUSAL,
            )

        parsed = _parse_verifier_response(raw, body)
        if parsed is None:
            return (
                AnswerVerifierResult(
                    claims=[],
                    revised_answer_body="",
                    substantive=False,
                ),
                VERIFIER_REFUSAL,
            )

        revised = (parsed.revised_answer_body or "").strip()
        if not parsed.substantive or not revised:
            return (
                AnswerVerifierResult(
                    claims=parsed.claims,
                    revised_answer_body="",
                    substantive=False,
                ),
                VERIFIER_REFUSAL,
            )

        return parsed, revised
