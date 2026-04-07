"""Загрузка и парсинг документов"""

import logging
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

from src.knowledge.chunker import chunk_document

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE = Path(__file__).resolve().parent.parent.parent / "knowledge_base"


def load_documents_from_dir(
    path: Path,
    source_name: str,
    glob: str = "**/*.md",
) -> list[Document]:
    """Загружает документы из директории."""
    if not path.exists():
        logger.warning("Knowledge path %s does not exist", path)
        return []

    reader = SimpleDirectoryReader(
        input_dir=str(path),
        recursive=True,
        required_exts=[".md", ".txt"],
        filename_as_id=True,
    )
    docs = reader.load_data()
    for d in docs:
        d.metadata["source_name"] = source_name
    return docs


def ingest_knowledge_base() -> list[Document]:
    """Загружает всю базу знаний: ТК РФ + ответы Роструда."""
    documents: list[Document] = []

    tkrf_path = KNOWLEDGE_BASE / "tkrf"
    rostrud_path = KNOWLEDGE_BASE / "rostrud"
    uploads_path = KNOWLEDGE_BASE / "uploads"

    docs_tkrf = load_documents_from_dir(tkrf_path, "tkrf")
    documents.extend(docs_tkrf)

    docs_rostrud = load_documents_from_dir(rostrud_path, "rostrud")
    documents.extend(docs_rostrud)

    docs_uploads = load_documents_from_dir(uploads_path, "uploads")
    documents.extend(docs_uploads)

    logger.info("Loaded %d documents (tkrf: %d, rostrud: %d, uploads: %d)",
                len(documents), len(docs_tkrf), len(docs_rostrud), len(docs_uploads))
    return documents


def ingest_custom_file(file_path: Path, source_name: str = "custom") -> list[Document]:
    """Загружает один файл (для admin API)."""
    reader = SimpleDirectoryReader(
        input_files=[str(file_path)],
        filename_as_id=True,
    )
    docs = reader.load_data()
    for d in docs:
        d.metadata["source_name"] = source_name
    return docs
