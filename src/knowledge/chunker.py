"""Чанкинг ТК РФ по статьям"""

import re
from pathlib import Path

from llama_index.core.schema import Document, TextNode

from src.knowledge.metadata import ChunkMetadata, extract_references
from src.knowledge.rostrud_faq_parser import (
    _first_n_sentences,
    parse_rostrud_faq_markdown,
)

# паттерн: "Статья N. Заголовок" или "Статья N.N. Заголовок"
ARTICLE_HEADER = re.compile(
    r"^Статья\s+(\d+(?:\.\d+)?)\.\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_section_chapter(text_before: str) -> tuple[str, str]:
    """Извлекает section и chapter из текста выше статьи"""
    section, chapter = "", ""
    lines = text_before.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("Глава ") or "Глава " in line:
            chapter = line
            break
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("Раздел ") or "Раздел " in line:
            section = line
            break
    return section, chapter


def chunk_tkrf(text: str, source_path: str = "tkrf") -> list[TextNode]:
    """Разбивает текст ТК РФ по статьям"""
    nodes: list[TextNode] = []
    matches = list(ARTICLE_HEADER.finditer(text))

    if not matches:
        # Fallback: весь текст как один чанк
        if text.strip():
            refs = extract_references(text)
            meta = ChunkMetadata(
                article="0",
                source=source_path,
                source_type="law",
                references=refs,
            )
            nodes.append(
                TextNode(text=text.strip(), metadata=meta.to_dict())
            )
        return nodes

    for i, m in enumerate(matches):
        article_num = m.group(1)
        title = m.group(2).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        text_before = text[:start]
        section, chapter = _extract_section_chapter(text_before)

        refs = extract_references(content)
        full_citation = f"ст. {article_num} ТК"
        meta = ChunkMetadata(
            article=article_num,
            section=section,
            chapter=chapter,
            source=source_path,
            source_type="law",
            references=refs,
            full_citation=full_citation,
        )

        node = TextNode(text=content, metadata=meta.to_dict())
        nodes.append(node)

    return nodes


def chunk_rostrud_plain(text: str, source_path: str = "rostrud") -> list[TextNode]:
    """Чанкинг для прочих разъяснений Роструда (не в формате faq_*.md)."""
    nodes: list[TextNode] = []
    blocks = re.split(r"\n{2,}", text)
    for idx, block in enumerate(blocks):
        if not block.strip():
            continue
        refs = extract_references(block)
        meta = ChunkMetadata(
            article=str(idx),
            source=source_path,
            source_type="explanation",
            references=refs,
            full_citation=f"Ответ Роструда #{idx + 1}",
        )
        nodes.append(TextNode(text=block.strip(), metadata=meta.to_dict()))
    return nodes


def chunk_rostrud_faq_file(text: str, file_path: str) -> list[TextNode]:
    """
    Чанкинг разъяснений Роструда на два узла: retrieval (вопрос + 2 первых предложения ответа)
    и context (полный ответ).
    """
    parsed = parse_rostrud_faq_markdown(text, file_path)
    if not parsed:
        return chunk_rostrud_plain(text, source_path="rostrud")

    faq_id = parsed.faq_id
    url = parsed.source_url
    full_citation = f"Ответ Роструда (онлайнинспекция): {url}"

    preview = _first_n_sentences(parsed.answer_body, 2)
    retrieval_text = f"{parsed.question}\n\n{preview}".strip()
    refs_q = extract_references(parsed.question + " " + preview)
    refs_a = extract_references(parsed.answer_body)

    meta_base = ChunkMetadata(
        article="",
        source="rostrud_faq",
        source_type="explanation",
        references=sorted(set(refs_q + refs_a)),
        full_citation=full_citation,
        faq_id=faq_id,
        source_url=url,
    )

    meta_a = ChunkMetadata(
        article=meta_base.article,
        source=meta_base.source,
        source_type=meta_base.source_type,
        references=meta_base.references,
        full_citation=meta_base.full_citation,
        faq_id=meta_base.faq_id,
        source_url=meta_base.source_url,
        chunk_role="retrieval",
    )
    meta_b = ChunkMetadata(
        article=meta_base.article,
        source=meta_base.source,
        source_type=meta_base.source_type,
        references=extract_references(parsed.answer_body),
        full_citation=meta_base.full_citation,
        faq_id=meta_base.faq_id,
        source_url=meta_base.source_url,
        chunk_role="context",
    )

    node_a = TextNode(text=retrieval_text, metadata=meta_a.to_dict())
    node_b = TextNode(text=parsed.answer_body, metadata=meta_b.to_dict())
    node_a.id_ = f"faq_{faq_id}_retrieval"
    node_b.id_ = f"faq_{faq_id}_context"
    return [node_a, node_b]


def _is_rostrud_faq_path(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    return "/faq/faq_" in p or p.endswith(".md") and "/faq/" in p and "faq_" in p


def chunk_document(doc: Document) -> list[TextNode]:
    """Чанкинг документа в зависимости от источника."""
    path = str(doc.metadata.get("file_path", ""))
    text = doc.text

    if "rostrud" in path.lower():
        if _is_rostrud_faq_path(path):
            return chunk_rostrud_faq_file(text, path)
        return chunk_rostrud_plain(text, source_path="rostrud")
    return chunk_tkrf(text, source_path="tkrf")
