"""LLM-агент: JSON-парсинг и моки вызовов"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.answer_verifier import AnswerVerifier, VERIFIER_REFUSAL
from src.agent.json_llm import extract_json_object, parse_llm_json
from src.agent.legal_query_agent import LegalQueryAgent
from src.agent.schemas import LegalQueryAgentResult


def test_extract_json_from_fence():
    raw = '```json\n{"needs_rewrite": false, "preserved_facts": [], "legal_query": "тест", "rationale": "x"}\n```'
    d = extract_json_object(raw)
    assert d is not None
    assert d["needs_rewrite"] is False
    assert d["legal_query"] == "тест"


def test_parse_llm_json_legal_result():
    raw = '{"needs_rewrite": true, "preserved_facts": ["работник"], "legal_query": "Права работника при увольнении", "rationale": ""}'
    m = parse_llm_json(raw, LegalQueryAgentResult)
    assert m is not None
    assert m.needs_rewrite is True
    assert "работник" in m.preserved_facts


@pytest.mark.asyncio
async def test_legal_query_agent_fallback_on_bad_json():
    llm = MagicMock()
    llm.achat = AsyncMock(return_value=SimpleNamespace(message=SimpleNamespace(content="not json")))
    agent = LegalQueryAgent(llm)
    r = await agent.normalize("Когда зарплата?")
    assert r.legal_query == "Когда зарплата?"
    assert r.needs_rewrite is False


@pytest.mark.asyncio
async def test_legal_query_agent_success():
    payload = (
        '{"needs_rewrite": false, "preserved_facts": [], '
        '"legal_query": "ignored when false", "rationale": "ok"}'
    )
    llm = MagicMock()
    llm.achat = AsyncMock(return_value=SimpleNamespace(message=SimpleNamespace(content=payload)))
    agent = LegalQueryAgent(llm)
    r = await agent.normalize("Статья 140 ТК срок расчёта")
    assert r.needs_rewrite is False
    assert r.legal_query == "Статья 140 ТК срок расчёта"


@pytest.mark.asyncio
async def test_answer_verifier_refusal_on_parse_error():
    llm = MagicMock()
    llm.achat = AsyncMock(return_value=SimpleNamespace(message=SimpleNamespace(content="oops")))
    v = AnswerVerifier(llm)
    from llama_index.core.schema import NodeWithScore, TextNode

    n = TextNode(text="ч. 1 ст. 140 ТК", metadata={"full_citation": "ст. 140 ТК РФ"})
    n.id_ = "nid1"
    nodes = [NodeWithScore(node=n, score=0.9)]

    res, text = await v.verify("Когда зарплата?", "Вы имеете право на всё.", nodes)
    assert text == VERIFIER_REFUSAL
    assert not res.substantive


@pytest.mark.asyncio
async def test_answer_verifier_success():
    payload = (
        '{"claims": [{"text": "x", "supported": true, "evidence_chunk_indices": [1]}], '
        '"revised_answer_body": "Краткий ответ по ст. 140 ТК.", "substantive": true}'
    )
    llm = MagicMock()
    llm.achat = AsyncMock(return_value=SimpleNamespace(message=SimpleNamespace(content=payload)))
    v = AnswerVerifier(llm)
    from llama_index.core.schema import NodeWithScore, TextNode

    n = TextNode(text="ст. 140 ТК", metadata={"full_citation": "ст. 140 ТК РФ"})
    n.id_ = "a"
    nodes = [NodeWithScore(node=n, score=0.5)]

    res, text = await v.verify("Срок?", "Краткий ответ по ст. 140 ТК.", nodes)
    assert res.substantive
    assert "140" in text


@pytest.mark.asyncio
async def test_answer_verifier_wraps_single_claim_object():
    """HF иногда возвращает один claim без корневых claims/revised_answer_body/substantive."""
    payload = (
        '{"text": "Ежегодный отпуск 28 дней", "supported": true, "evidence_chunk_indices": [1, 2]}'
    )
    llm = MagicMock()
    llm.achat = AsyncMock(return_value=SimpleNamespace(message=SimpleNamespace(content=payload)))
    v = AnswerVerifier(llm)
    from llama_index.core.schema import NodeWithScore, TextNode

    n = TextNode(text="ст. 115 ТК: 28 дней", metadata={"full_citation": "ст. 115 ТК"})
    n.id_ = "n1"
    nodes = [NodeWithScore(node=n, score=0.9)]

    draft = "Для большинства работников 28 календарных дней основного отпуска (ст. 115 ТК)."
    res, text = await v.verify("Сколько дней отпуска?", draft, nodes)
    assert res.substantive
    assert text == draft
    assert len(res.claims) == 1
    assert res.claims[0].supported
    assert res.claims[0].evidence_chunk_indices == [1, 2]
