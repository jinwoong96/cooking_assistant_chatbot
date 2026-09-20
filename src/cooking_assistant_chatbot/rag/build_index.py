from __future__ import annotations

import logging

from ..config import settings
from .embeddings import BGEEmbeddingFunction
from .indexer import build_index_from_sqlite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run() -> int:
    logger.info(
        "Loading embedding model %s (first run downloads it, ~2GB)...",
        settings.embedding_model_name,
    )
    embedding_function = BGEEmbeddingFunction(settings.embedding_model_name)
    count = build_index_from_sqlite(
        settings.db_path, settings.chroma_db_path, embedding_function
    )
    logger.info("Indexed %d recipes into %s", count, settings.chroma_db_path)
    return count


if __name__ == "__main__":
    run()
