#!/usr/bin/env python3
"""Индексация базы знаний в pgvector"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import get_settings
from src.core.logging_config import setup_logging
from src.knowledge.indexer import index_knowledge_base

setup_logging("INFO")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Drop and recreate index")
    args = parser.parse_args()

    settings = get_settings()
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    embed_model = HuggingFaceEmbedding(
        model_name=settings.embedding_model,
        token=settings.huggingface_hub_token,
    )

    count = index_knowledge_base(embed_model, reset=args.reset)
    logger.info("Indexed %d chunks", count)


if __name__ == "__main__":
    main()
