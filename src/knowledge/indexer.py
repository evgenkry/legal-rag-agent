"""Индексация в PostgreSQL (pgvector + hybrid)"""

import logging
from urllib.parse import urlparse

import psycopg2
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.postgres import PGVectorStore

from src.core.config import get_settings
from src.knowledge.chunker import chunk_document
from src.knowledge.ingester import ingest_custom_file, ingest_knowledge_base

logger = logging.getLogger(__name__)

INDEX_TABLE = "rag_chunks"


def _parse_db_url(url: str) -> dict:
    """Парсит connection string PostgreSQL."""
    parsed = urlparse(url)
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "user": parsed.username or "postgres",
        "password": parsed.password or "",
        "database": parsed.path.lstrip("/") if parsed.path else "rag_db",
    }


def get_vector_store() -> PGVectorStore:
    """Создаёт PGVectorStore с гибридным поиском."""
    settings = get_settings()
    conn_params = _parse_db_url(settings.pgvector_url)

    vector_store = PGVectorStore.from_params(
        database=conn_params["database"],
        host=conn_params["host"],
        port=conn_params["port"],
        user=conn_params["user"],
        password=conn_params["password"],
        table_name=INDEX_TABLE,
        embed_dim=settings.embed_dim,
        hybrid_search=True,
        text_search_config="russian",
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )
    return vector_store


def index_knowledge_base(embed_model, reset: bool = False) -> int:
    """Индексирует базу знаний в pgvector."""
    documents = ingest_knowledge_base()
    nodes: list[TextNode] = []
    for doc in documents:
        nodes.extend(chunk_document(doc))

    vector_store = get_vector_store()
    if reset:
        settings = get_settings()
        conn_params = _parse_db_url(settings.pgvector_url)
        with psycopg2.connect(
            host=conn_params["host"],
            port=conn_params["port"],
            user=conn_params["user"],
            password=conn_params["password"],
            dbname=conn_params["database"],
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {INDEX_TABLE} CASCADE")
            conn.commit()

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    logger.info("Indexed %d nodes", len(nodes))
    return len(nodes)


def index_single_document(
    embed_model,
    file_path: "Path",
    source_name: str = "custom",
) -> int:
    """
    Инкрементальная индексация одного файла.
    Парсит файл, формирует чанки и эмбеддинги, добавляет в vector store.
    Возвращает число добавленных чанков.
    """
    from pathlib import Path
    path = Path(file_path) if not isinstance(file_path, Path) else file_path
    documents = ingest_custom_file(path, source_name=source_name)
    nodes: list[TextNode] = []
    for doc in documents:
        nodes.extend(chunk_document(doc))

    if not nodes:
        logger.warning("No chunks from %s", path)
        return 0

    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    index.insert_nodes(nodes)
    logger.info("Indexed %d chunks from %s", len(nodes), path)
    return len(nodes)
