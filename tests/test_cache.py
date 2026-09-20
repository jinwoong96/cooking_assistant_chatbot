from datetime import timedelta

from cooking_assistant_chatbot.data.db import get_connection
from cooking_assistant_chatbot.pricing.cache import get_cached_price, set_cached_price
from cooking_assistant_chatbot.pricing.enuri_client import ShoppingItem


def test_set_and_get_cached_price_roundtrip(tmp_path):
    conn = get_connection(str(tmp_path / "test.db"))
    item = ShoppingItem(title="다진마늘 500g", price=3500, brand="청정원", link="https://example.com")

    set_cached_price(conn, "다진마늘", item)

    assert get_cached_price(conn, "다진마늘") == item


def test_get_cached_price_returns_none_when_missing(tmp_path):
    conn = get_connection(str(tmp_path / "test.db"))

    assert get_cached_price(conn, "존재하지않음") is None


def test_get_cached_price_returns_none_when_expired(tmp_path):
    conn = get_connection(str(tmp_path / "test.db"))
    set_cached_price(conn, "다진마늘", ShoppingItem(title="다진마늘", price=3500))

    assert get_cached_price(conn, "다진마늘", ttl=timedelta(seconds=-1)) is None


def test_set_cached_price_overwrites_existing_entry(tmp_path):
    conn = get_connection(str(tmp_path / "test.db"))
    set_cached_price(conn, "다진마늘", ShoppingItem(title="old", price=1000))

    set_cached_price(conn, "다진마늘", ShoppingItem(title="new", price=2000))

    cached = get_cached_price(conn, "다진마늘")
    assert cached.title == "new"
    assert cached.price == 2000
