from __future__ import annotations

import logging

from ..config import settings
from .db import get_connection
from .ingredient_index import build_ingredient_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run() -> int:
    conn = get_connection(settings.db_path)
    try:
        count = build_ingredient_index(conn)
    finally:
        conn.close()
    logger.info("재료-레시피 색인 %d건을 %s에 적재", count, settings.db_path)
    return count


if __name__ == "__main__":
    run()
