"""Тесты форматирования источников в ответе"""

from src.rag.source_format import build_formatted_citations


class _FakeNode:
    def __init__(self, metadata: dict):
        self.metadata = metadata


class _FakeNWS:
    def __init__(self, metadata: dict):
        self.node = _FakeNode(metadata)


def test_faq_urls_then_grouped_tk_articles():
    nodes = [
        _FakeNWS(
            {
                "source": "rostrud_faq",
                "faq_id": "993",
                "source_url": "https://онлайнинспекция.рф/questions/viewFaq/993",
                "full_citation": "Ответ Роструда (онлайнинспекция): https://онлайнинспекция.рф/questions/viewFaq/993",
            }
        ),
        _FakeNWS(
            {
                "source": "rostrud_faq",
                "faq_id": "947",
                "source_url": "https://онлайнинспекция.рф/questions/viewFaq/947",
                "full_citation": "Ответ Роструда (онлайнинспекция): https://онлайнинспекция.рф/questions/viewFaq/947",
            }
        ),
        _FakeNWS({"source": "tkrf", "article": "353", "full_citation": "ст. 353 ТК"}),
        _FakeNWS({"source": "tkrf", "article": "127", "full_citation": "ст. 127 ТК"}),
        _FakeNWS({"source": "tkrf", "article": "292", "full_citation": "ст. 292 ТК"}),
    ]
    lines = build_formatted_citations(nodes)
    assert lines[0] == "Ответ Роструда: https://онлайнинспекция.рф/questions/viewFaq/993"
    assert lines[1] == "Ответ Роструда: https://онлайнинспекция.рф/questions/viewFaq/947"
    assert lines[2] == "Статьи 353, 127, 292 ТК РФ"


def test_dedupe_same_faq_url():
    u = "https://онлайнинспекция.рф/questions/viewFaq/993"
    nodes = [
        _FakeNWS({"source": "rostrud_faq", "faq_id": "993", "source_url": u, "full_citation": u}),
        _FakeNWS({"source": "rostrud_faq", "faq_id": "993", "source_url": u, "full_citation": u}),
    ]
    assert build_formatted_citations(nodes) == [f"Ответ Роструда: {u}"]


def test_tk_from_full_citation_without_article_field():
    nodes = [
        _FakeNWS({"source": "custom", "full_citation": "ст. 140 ТК"}),
    ]
    assert build_formatted_citations(nodes) == ["Статьи 140 ТК РФ"]


def test_rostrud_plain_goes_to_other():
    nodes = [
        _FakeNWS(
            {
                "source": "rostrud",
                "article": "0",
                "full_citation": "Ответ Роструда #1",
            }
        ),
    ]
    assert build_formatted_citations(nodes) == ["Ответ Роструда #1"]
