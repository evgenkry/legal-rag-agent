"""Парсинг FAQ Роструда"""

import re
from dataclasses import dataclass

FAQ_HEADER_FAQ_ID = re.compile(r"faq_id:\s*(\d+)", re.IGNORECASE)
FAQ_HEADER_URL = re.compile(r"url:\s*(\S+?)(?:\s*-->|$)", re.IGNORECASE)
FILENAME_FAQ = re.compile(r"faq_0*(\d+)\.md$", re.IGNORECASE)

QUESTION_BLOCK = re.compile(
    r"\*\*Вопрос:\*\*\s*(.+?)(?=\*\*Ответ:\*\*|\Z)",
    re.DOTALL | re.IGNORECASE,
)
ANSWER_BLOCK = re.compile(
    r"\*\*Ответ:\*\*\s*(.+?)\Z",
    re.DOTALL | re.IGNORECASE,
)

# конец предложения (рус./лат. точка/восклицание/вопрос)
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


@dataclass
class ParsedRostrudFaq:
    faq_id: str
    source_url: str
    question: str
    answer_body: str


def _first_n_sentences(text: str, n: int = 2) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return ""
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    if not parts:
        return text[:500] if len(text) > 500 else text
    return " ".join(parts[:n])


def _parse_header(text: str) -> tuple[str, str]:
    first_line = text.lstrip().split("\n", 1)[0] if text else ""
    fid_m = FAQ_HEADER_FAQ_ID.search(first_line)
    url_m = FAQ_HEADER_URL.search(first_line)
    fid = fid_m.group(1) if fid_m else ""
    url = url_m.group(1).rstrip(">") if url_m else ""
    return fid, url


def parse_faq_id_from_path(file_path: str) -> str:
    m = FILENAME_FAQ.search(file_path.replace("\\", "/"))
    return str(int(m.group(1))) if m else ""


def parse_rostrud_faq_markdown(text: str, file_path: str) -> ParsedRostrudFaq | None:
    """Извлекает faq_id, url, вопрос и полный ответ из markdown FAQ."""
    header_fid, header_url = _parse_header(text)
    path_fid = parse_faq_id_from_path(file_path)
    faq_id = header_fid or path_fid
    if not faq_id:
        return None

    source_url = header_url
    if not source_url:
        source_url = f"https://онлайнинспекция.рф/questions/viewFaq/{faq_id}"

    qm = QUESTION_BLOCK.search(text)
    am = ANSWER_BLOCK.search(text)
    if not qm or not am:
        return None

    question = re.sub(r"\s+", " ", qm.group(1).strip())
    answer_body = am.group(1).strip()
    # Убрать дублирующий префикс "Ответ:" в теле
    answer_body = re.sub(r"^Ответ:\s*", "", answer_body, flags=re.IGNORECASE)

    return ParsedRostrudFaq(
        faq_id=faq_id,
        source_url=source_url,
        question=question,
        answer_body=answer_body,
    )
