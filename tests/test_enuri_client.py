import json

import httpx
import pytest
import respx

from cooking_assistant_chatbot.pricing.enuri_client import BASE_URL, EnuriClient


def _ld_json_response(items: list[dict]) -> httpx.Response:
    body = {
        "itemListElement": [
            {"item": item, "@type": "ListItem", "position": i + 1}
            for i, item in enumerate(items)
        ]
    }
    html = (
        "<html><body><script type=\"application/ld+json\">"
        + json.dumps(body)
        + "</script></body></html>"
    )
    return httpx.Response(200, text=html)


def _product(name: str, low_price: int, brand: str = "", url: str = "") -> dict:
    return {
        "name": name,
        "offers": {"lowPrice": low_price, "highPrice": low_price + 1000, "offerCount": 1},
        "brand": {"name": brand},
        "url": url,
    }


@respx.mock
def test_search_keeps_enuri_relevance_order():
    respx.get(BASE_URL).mock(
        return_value=_ld_json_response(
            [_product("비싼 마늘", 30000), _product("싼 마늘", 5000), _product("중간 마늘", 15000)]
        )
    )
    client = EnuriClient(min_request_interval=0)

    items = client.search("다진마늘")

    assert [i.title for i in items] == ["비싼 마늘", "싼 마늘", "중간 마늘"]
    assert [i.price for i in items] == [30000, 5000, 15000]


@respx.mock
def test_search_returns_empty_list_when_no_ld_json_block():
    respx.get(BASE_URL).mock(
        return_value=httpx.Response(200, text="<html><body>no data</body></html>")
    )
    client = EnuriClient(min_request_interval=0)

    assert client.search("존재하지않는상품명xyz") == []


@respx.mock
def test_search_skips_items_without_low_price():
    respx.get(BASE_URL).mock(
        return_value=_ld_json_response(
            [
                {"name": "가격정보없음", "offers": {}, "brand": {}, "url": ""},
                _product("가격있음", 1000),
            ]
        )
    )
    client = EnuriClient(min_request_interval=0)

    items = client.search("대파")

    assert len(items) == 1
    assert items[0].title == "가격있음"


def test_throttle_waits_at_least_min_interval(monkeypatch):
    times = iter([100.0, 100.2, 100.2])
    sleeps = []

    monkeypatch.setattr(
        "cooking_assistant_chatbot.pricing.enuri_client.time.monotonic", lambda: next(times)
    )
    monkeypatch.setattr(
        "cooking_assistant_chatbot.pricing.enuri_client.time.sleep", sleeps.append
    )

    client = EnuriClient(min_request_interval=1.0)
    client._throttle()
    client._throttle()

    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(0.8)

