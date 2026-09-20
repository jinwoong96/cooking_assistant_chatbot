from __future__ import annotations

import logging

from ..config import settings
from .db import get_connection, upsert_recipes
from .food_safety_client import FoodSafetyClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(page_size: int = 100) -> int:
    """Fetch every recipe from the COOKRCP01 API and upsert it into SQLite.

    Returns the number of recipes ingested.
    """
    if settings.food_safety_api_key == "sample":
        logger.warning(
            "Using the 'sample' API key: this only returns a couple of demo "
            "rows, not the full dataset. Set FOOD_SAFETY_API_KEY in .env with "
            "a personal key from https://various.foodsafetykorea.go.kr for a "
            "real ingestion run."
        )

    client = FoodSafetyClient(api_key=settings.food_safety_api_key)
    conn = get_connection(settings.db_path)
    count = 0
    try:
        batch = []
        for recipe in client.fetch_all(page_size=page_size):
            batch.append(recipe)
            if len(batch) >= page_size:
                upsert_recipes(conn, batch)
                count += len(batch)
                logger.info("Ingested %d recipes so far", count)
                batch = []
        if batch:
            upsert_recipes(conn, batch)
            count += len(batch)
    finally:
        client.close()
        conn.close()

    logger.info("Done. Ingested %d recipes into %s", count, settings.db_path)
    return count


if __name__ == "__main__":
    run()
