"""Форматирование списка источников для ответа (ссылки на ответы Роструда, краткий список статей ТК РФ)"""

from __future__ import annotations

import re
from typing import Any

# URL в full_citation (в том числе кириллический домен)
_URL_IN_TEXT = re.compile(r"https?://[^\s]+", re.IGNORECASE)
# ст. 140 ТК, ст. 14.1 ТК
_ST_TK = re.compile(r"ст\.\s*(\d+(?:\.\d+)?)\s*ТК", re.IGNORECASE)


def _first_url(text: str) -> str:
    if not text:
        return ""
    m = _URL_IN_TEXT.search(text)
    if not m:
        return ""
    return m.group(0).rstrip(").,;]")


def _is_rostrud_faq_chunk(meta: dict[str, Any]) -> bool:
    src = (meta.get("source") or "").lower()
    if src == "rostrud_faq" or meta.get("faq_id"):
        return True
    url = (meta.get("source_url") or "").strip() or _first_url(meta.get("full_citation") or "")
    if not url:
        return False
    u = url.lower()
    return "viewfaq" in u or "онлайнинспекция" in u or "xn--" in u  # онлайнинспекция


def build_formatted_citations(nodes: list[Any]) -> list[str]:
    """
    Собирает строки источников в порядке:
    1) Уникальные ссылки Роструда (онлайнинспекция): «Ответ Роструда: <url>» в порядке первого появления
    2) Одна строка по всем статьям ТК: «Статьи 1, 2, … ТК РФ» (порядок как в контексте, первое появление)
    """
    rostrud_lines: list[str] = []
    seen_urls: set[str] = set()
    tk_nums_order: list[str] = []
    seen_tk: set[str] = set()
    other: list[str] = []
    seen_other: set[str] = set()

    for n in nodes:
        meta = dict(getattr(n.node, "metadata", None) or {})
        fc = (meta.get("full_citation") or "").strip()
        source = (meta.get("source") or "").lower()

        url = (meta.get("source_url") or "").strip()
        if not url:
            url = _first_url(fc)

        if _is_rostrud_faq_chunk(meta) and url:
            if url not in seen_urls:
                seen_urls.add(url)
                rostrud_lines.append(f"Ответ Роструда: {url}")
            continue

        article = meta.get("article")
        if source == "tkrf" and article is not None:
            a = str(article).strip()
            if a and a != "0":
                if a not in seen_tk:
                    seen_tk.add(a)
                    tk_nums_order.append(a)
                continue

        if fc:
            m = _ST_TK.search(fc)
            if m:
                a = m.group(1)
                if a not in seen_tk:
                    seen_tk.add(a)
                    tk_nums_order.append(a)
                continue

            if fc not in seen_other:
                seen_other.add(fc)
                other.append(fc)

    out: list[str] = []
    out.extend(rostrud_lines)
    if tk_nums_order:
        out.append("Статьи " + ", ".join(tk_nums_order) + " ТК РФ")
    out.extend(other)
    return out
