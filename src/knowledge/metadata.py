"""Схема метаданных чанков"""

import re
from dataclasses import dataclass, field
from typing import Optional

# Регулярные выражения для извлечения ссылок между нормами
# статья, статьёй, статьи, ст., ст
ARTICLE_PATTERN = re.compile(
    r"(?:статья|статьёй|статьи|ст\.?)\s*(\d+(?:\s*[-–]\s*\d+)?)",
    re.IGNORECASE,
)
ARTICLES_RANGE_PATTERN = re.compile(
    r"статьи\s+(\d+)\s*[-–]\s*(\d+)",
    re.IGNORECASE,
)
PART_SECTION_PATTERN = re.compile(
    r"^(ЧАСТЬ\s+(?:ПЕРВАЯ|ВТОРАЯ|ТРЕТЬЯ|ЧЕТВЁРТАЯ|ПЯТАЯ|ШЕСТАЯ|СЕДЬМАЯ))\s*$",
    re.IGNORECASE | re.MULTILINE,
)
SECTION_PATTERN = re.compile(
    r"^Раздел\s+[IVXLCDM]+\.\s+(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
CHAPTER_PATTERN = re.compile(
    r"^Глава\s+(\d+\.?\s*.+)$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class ChunkMetadata:
    """Метаданные чанка."""

    article: str  # "114", "115" или "" для FAQ
    section: str = ""
    chapter: str = ""
    source: str = "tkrf"  # tkrf | rostrud | rostrud_faq
    source_type: str = "law"  # law | explanation
    references: list[str] = field(default_factory=list)  # упоминания других статей
    full_citation: str = ""  # "ст. 114 ТК" или ссылка на FAQ
    faq_id: str = ""
    source_url: str = ""
    chunk_role: str = ""  # "" | "retrieval" | "context"

    def to_dict(self) -> dict:
        """Для LlamaIndex TextNode.metadata."""
        d = {
            "article": self.article,
            "section": self.section,
            "chapter": self.chapter,
            "source": self.source,
            "source_type": self.source_type,
            "references": self.references,
            "full_citation": self.full_citation,
            "faq_id": self.faq_id,
            "source_url": self.source_url,
            "chunk_role": self.chunk_role,
        }
        return {k: v for k, v in d.items() if v != "" and v != []}


def extract_references(text: str) -> list[str]:
    """Извлекает номера статей из текста (ссылки между нормами)."""
    refs: set[str] = set()

    # "статьи 114-120"
    for m in ARTICLES_RANGE_PATTERN.finditer(text):
        start, end = int(m.group(1)), int(m.group(2))
        for n in range(start, min(start + 10, end + 1)):
            refs.add(str(n))

    # "статья 114", "ст. 114", "ст 114", "ст. 114-120"
    for m in ARTICLE_PATTERN.finditer(text):
        num = m.group(1).strip()
        if "-" in num or "–" in num:
            parts = re.split(r"[-–]", num)
            if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                a, b = int(parts[0].strip()), int(parts[1].strip())
                for n in range(a, min(a + 10, b + 1)):
                    refs.add(str(n))
        elif num and num.replace(".", "").isdigit():
            refs.add(num)

    return sorted(refs, key=lambda x: int(x) if x.replace(".", "").isdigit() else 0)
