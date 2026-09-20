from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from .enuri_client import ShoppingItem

DEFAULT_TTL = timedelta(hours=24)


def get_cached_price(
    conn: sqlite3.Connection, ingredient_name: str, ttl: timedelta = DEFAULT_TTL
) -> ShoppingItem | None:
    row = conn.execute(
        "SELECT title, price, brand, link, fetched_at FROM price_cache WHERE ingredient_name = ?",
        (ingredient_name,),
    ).fetchone()
    if row is None:
        return None

    title, price, brand, link, fetched_at = row
    if datetime.now(timezone.utc) - datetime.fromisoformat(fetched_at) > ttl:
        return None
    return ShoppingItem(title=title, price=price, brand=brand or "", link=link or "")


def set_cached_price(conn: sqlite3.Connection, ingredient_name: str, item: ShoppingItem) -> None:
    conn.execute(
        """
        INSERT INTO price_cache (ingredient_name, title, price, brand, link, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(ingredient_name) DO UPDATE SET
            title=excluded.title,
            price=excluded.price,
            brand=excluded.brand,
            link=excluded.link,
            fetched_at=excluded.fetched_at
        """,
        (
            ingredient_name,
            item.title,
            item.price,
            item.brand,
            item.link,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
