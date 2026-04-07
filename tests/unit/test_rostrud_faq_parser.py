"""Парсер FAQ Роструда"""

from pathlib import Path

from src.knowledge.chunker import chunk_rostrud_faq_file
from src.knowledge.rostrud_faq_parser import parse_rostrud_faq_markdown


def test_parse_faq_extracts_ids_and_url():
    sample = """<!-- source: x faq_id: 1001 url: https://example.com/f/1001 -->

**Вопрос:** Что такое отпуск?

**Ответ:**

Первое предложение. Второе предложение. Третье.
"""
    p = parse_rostrud_faq_markdown(sample, "/tmp/faq_001001.md")
    assert p is not None
    assert p.faq_id == "1001"
    assert "example.com" in p.source_url
    assert "отпуск" in p.question.lower()
    assert "Третье" in p.answer_body


def test_chunk_rostrud_faq_two_nodes():
    root = Path(__file__).resolve().parents[2] / "knowledge_base" / "rostrud" / "faq"
    f = root / "faq_001001.md"
    if not f.exists():
        return
    text = f.read_text(encoding="utf-8")
    nodes = chunk_rostrud_faq_file(text, str(f))
    assert len(nodes) == 2
    roles = {nodes[0].metadata.get("chunk_role"), nodes[1].metadata.get("chunk_role")}
    assert roles == {"retrieval", "context"}
    assert nodes[0].metadata.get("faq_id") == nodes[1].metadata.get("faq_id")
