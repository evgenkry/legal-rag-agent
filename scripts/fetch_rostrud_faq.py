#!/usr/bin/env python3
"""
Скачивает вопросы и ответы из раздела «Часто задаваемые» (FAQ) портала
Онлайнинспекция.рф (Роструд) и сохраняет их в knowledge_base/rostrud/ как .md.
Список подгружается AJAX-ом (как в браузере на /questions/?question_type=faq).
Запуск из корня проекта:
  .venv/bin/python -m scripts.fetch_rostrud_faq --limit 100
  python -m scripts.fetch_rostrud_faq --limit 10 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

# Корень репозитория (родитель каталога scripts/)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "knowledge_base" / "rostrud" / "faq"

# Человекочитаемый URL в шапке файлов; запросы идут на punycode-хост (IDNA)
DISPLAY_ORIGIN = "https://онлайнинспекция.рф"
LIST_PATH = "/city/questionsList/"
VIEW_FAQ_PATH = "/questions/viewFaq/"

USER_AGENT = "Mozilla/5.0 (compatible; RAG-kb-fetch/1.0; +local knowledge base mirror)"

logger = logging.getLogger(__name__)


def idna_hostname(domain: str) -> str:
    """Кодирует каждую метку домена в punycode (онлайнинспекция.рф → xn--...)."""
    return ".".join(label.encode("idna").decode("ascii") for label in domain.split("."))


def build_base_url(domain: str) -> str:
    host = idna_hostname(domain.strip().lower())
    return f"https://{host}"


def fetch_faq_ids(
    client: httpx.Client,
    base: str,
    *,
    limit: int,
    delay_s: float,
) -> list[int]:
    """Собирает уникальные id вопросов со страниц списка (page=0,1,…)."""
    headers = {
        "User-Agent": USER_AGENT,
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{base}/questions/?question_type=faq",
    }
    seen: list[int] = []
    page = 0
    link_re = re.compile(r"/questions/viewFaq/(\d+)")

    while len(seen) < limit:
        r = client.get(
            f"{base}{LIST_PATH}",
            params={
                "source": "rostrud",
                "type": "ajax",
                "question_type": "faq",
                "page": page,
            },
            headers=headers,
        )
        r.raise_for_status()
        html = r.text
        if "theresnothingtoshow" in html or not link_re.search(html):
            break
        batch = [int(m) for m in link_re.findall(html)]
        for fid in batch:
            if fid not in seen:
                seen.append(fid)
            if len(seen) >= limit:
                break
        if "showmore" not in html:
            break
        page += 1
        if delay_s > 0:
            time.sleep(delay_s)

    return seen[:limit]


def parse_view_faq_html(html: str) -> tuple[str, str]:
    """Из страницы viewFaq извлекает заголовок-вопрос и текст ответа."""
    soup = BeautifulSoup(html, "html.parser")
    col = soup.select_one("div.col-lg-9.col-md-11.vi-fs-100")
    if not col:
        raise ValueError("Не найден блок с текстом вопроса (col-lg-9)")
    h1 = col.find("h1")
    if not h1:
        raise ValueError("Не найден заголовок h1 с формулировкой вопроса")
    question = h1.get_text(" ", strip=True)
    h1.decompose()
    # Убираем хвостовые ссылки «Вернуться» и пустые узлы
    for a in col.find_all("a", href=True):
        t = a.get_text(strip=True)
        if "вернуться" in t.lower():
            a.decompose()
    answer = col.get_text("\n", strip=True)
    # Хвост страницы: опрос «получили ли ответ» и формы
    cut = re.search(
        r"(?is)\n\s*Вы получили ответ на заданный вопрос\?",
        answer,
    )
    if cut:
        answer = answer[: cut.start()].rstrip()
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return question, answer


def render_markdown(faq_id: int, question: str, answer: str) -> str:
    url = f"{DISPLAY_ORIGIN}{VIEW_FAQ_PATH}{faq_id}"
    return (
        f"<!-- source: rostrud_onlineinspector faq_id: {faq_id} url: {url} -->\n\n"
        f"**Вопрос:** {question}\n\n"
        f"**Ответ:**\n\n{answer}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Скачать FAQ Онлайнинспекция.рф в markdown.")
    parser.add_argument(
        "--domain",
        default="онлайнинспекция.рф",
        help="Домен сайта (кириллический или ASCII), по умолчанию онлайнинспекция.рф",
    )
    parser.add_argument("--limit", type=int, default=100, help="Сколько вопросов собрать (по умолчанию 100)")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Каталог для .md (по умолчанию {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.4,
        help="Пауза между HTTP-запросами, сек (вежливость к серверу)",
    )
    parser.add_argument("--timeout", type=float, default=45.0, help="Таймаут запроса, сек")
    parser.add_argument("--dry-run", action="store_true", help="Только показать id, не писать файлы")
    parser.add_argument("-v", "--verbose", action="store_true", help="Подробный лог")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    base = build_base_url(args.domain)
    args.out.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=args.timeout, follow_redirects=True) as client:
        ids = fetch_faq_ids(client, base, limit=args.limit, delay_s=args.delay)
        logger.info("Найдено id для загрузки: %d", len(ids))
        if args.dry_run:
            print(*ids)
            return 0

        ok = 0
        for i, fid in enumerate(ids):
            path = args.out / f"faq_{fid:06d}.md"
            try:
                r = client.get(
                    f"{base}{VIEW_FAQ_PATH}{fid}",
                    headers={"User-Agent": USER_AGENT},
                )
                r.raise_for_status()
                q, a = parse_view_faq_html(r.text)
                path.write_text(render_markdown(fid, q, a), encoding="utf-8")
                ok += 1
                logger.info("[%d/%d] %s", i + 1, len(ids), path.name)
            except Exception as e:
                logger.warning("Пропуск id=%s: %s", fid, e)
            if args.delay > 0 and i + 1 < len(ids):
                time.sleep(args.delay)

        logger.info("Готово: записано %d файлов в %s", ok, args.out)
        if ok:
            logger.info("Переиндексация: python -m scripts.index_knowledge")

    return 0


if __name__ == "__main__":
    sys.exit(main())
