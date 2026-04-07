"""Admin API — загрузка документов."""

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.api.auth import verify_admin_api_key
from src.api.dependencies import get_embed_model
from src.core.config import get_settings
from src.knowledge.indexer import index_single_document

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(verify_admin_api_key)],
)

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge_base" / "uploads"
INGEST_TIMEOUT = 120

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    source_name: str = "custom",
) -> dict:
    """
    Сохраняет документ и запускает инкрементальную индексацию.
    Возвращает: {ingest_id, status, source, filename, chunks_added, error?}
    """
    if not file.filename or not file.filename.endswith((".md", ".txt")):
        raise HTTPException(400, "Only .md and .txt files are supported")

    ingest_id = str(uuid.uuid4())
    target_dir = UPLOAD_DIR / source_name
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / file.filename

    try:
        content = await file.read()
        path.write_bytes(content)
        logger.info("Saved upload: %s (ingest_id=%s)", path, ingest_id)
    except Exception as e:
        logger.exception("Ingest save failed: %s", e)
        raise HTTPException(500, str(e))

    embed_model = get_embed_model()
    try:
        chunks_added = await asyncio.wait_for(
            asyncio.to_thread(
                index_single_document,
                embed_model,
                path,
                source_name,
            ),
            timeout=INGEST_TIMEOUT,
        )
        return {
            "ingest_id": ingest_id,
            "status": "indexed",
            "source": source_name,
            "filename": file.filename,
            "chunks_added": chunks_added,
        }
    except asyncio.TimeoutError:
        err_msg = f"Indexing timed out after {INGEST_TIMEOUT}s"
        logger.warning("%s (ingest_id=%s)", err_msg, ingest_id)
        return {
            "ingest_id": ingest_id,
            "status": "failed_indexing",
            "source": source_name,
            "filename": file.filename,
            "chunks_added": 0,
            "error": err_msg,
        }
    except Exception as e:
        err_msg = str(e)
        logger.exception("Indexing failed: %s (ingest_id=%s)", e, ingest_id)
        return {
            "ingest_id": ingest_id,
            "status": "failed_indexing",
            "source": source_name,
            "filename": file.filename,
            "chunks_added": 0,
            "error": err_msg,
        }
