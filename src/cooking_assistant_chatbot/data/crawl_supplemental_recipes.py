from __future__ import annotations

import logging

from ..config import settings
from .db import get_connection, upsert_recipes
from .tenthousand_recipe_crawler import TenThousandRecipeCrawler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 자취생 일상 메뉴 위주 — 식약처 COOKRCP01(저염식/건강식 위주)에 부족한
# 분식/배달음식 스타일을 보완하기 위해 고른 키워드. 필요하면 늘리거나 바꿔도 됨.
DEFAULT_KEYWORDS = [
    "떡볶이",
    "라면",
    "카레",
    "파스타",
    "볶음밥",
    "김밥",
    "오므라이스",
    "돈까스",
    "짜장밥",
    "짬뽕",
    "토스트",
    "계란찜",
    "우동",
    "덮밥",
    "마라탕",
]


def run(keywords: list[str] | None = None, per_keyword: int = 10) -> int:
    """Crawl a small, fixed set of everyday-dish keywords from 10000recipe.com
    and upsert them into the same recipes table as the public-data recipes.

    See CLAUDE.md / tenthousand_recipe_crawler.py for the legal/scope notes
    this follows (personal, non-commercial, small-scale, not redistributed).
    """
    keywords = keywords or DEFAULT_KEYWORDS
    crawler = TenThousandRecipeCrawler()
    conn = get_connection(settings.db_path)
    total = 0
    try:
        for keyword in keywords:
            ids = crawler.search_recipe_ids(keyword, limit=per_keyword)
            recipes = []
            for recipe_id in ids:
                recipe = crawler.fetch_recipe(recipe_id)
                if recipe and recipe.name and recipe.ingredients_raw and recipe.steps:
                    recipes.append(recipe)
            if recipes:
                upsert_recipes(conn, recipes)
                total += len(recipes)
            logger.info("%s: %d/%d개 수집", keyword, len(recipes), len(ids))
    finally:
        crawler.close()
        conn.close()

    logger.info("완료. 총 %d개 레시피를 %s에 추가/갱신", total, settings.db_path)
    return total


if __name__ == "__main__":
    run()
