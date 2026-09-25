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
def test_search_cheapest_sorts_results_by_ascending_price():
    respx.get(BASE_URL).mock(
        return_value=_ld_json_response(
            [_product("비싼 마늘", 30000), _product("싼 마늘", 5000), _product("중간 마늘", 15000)]
        )
    )
    client = EnuriClient(min_request_interval=0)

    items = client.search_cheapest("다진마늘")

    assert [i.title for i in items] == ["싼 마늘", "중간 마늘", "비싼 마늘"]
    assert [i.price for i in items] == [5000, 15000, 30000]


@respx.mock
def test_search_cheapest_respects_limit():
    respx.get(BASE_URL).mock(
        return_value=_ld_json_response([_product(f"상품{i}", i * 1000) for i in range(10)])
    )
    client = EnuriClient(min_request_interval=0)

    items = client.search_cheapest("대파", limit=3)

    assert len(items) == 3


@respx.mock
def test_search_cheapest_returns_empty_list_when_no_ld_json_block():
    respx.get(BASE_URL).mock(
        return_value=httpx.Response(200, text="<html><body>no data</body></html>")
    )
    client = EnuriClient(min_request_interval=0)

    assert client.search_cheapest("존재하지않는상품명xyz") == []


@respx.mock
def test_search_cheapest_skips_items_without_low_price():
    respx.get(BASE_URL).mock(
        return_value=_ld_json_response(
            [
                {"name": "가격정보없음", "offers": {}, "brand": {}, "url": ""},
                _product("가격있음", 1000),
            ]
        )
    )
    client = EnuriClient(min_request_interval=0)

    items = client.search_cheapest("대파")

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


def _item(title: str, price: int):
    from cooking_assistant_chatbot.pricing.enuri_client import ShoppingItem

    return ShoppingItem(title=title, price=price)


def test_pick_relevant_cheapest_ignores_cheap_results_outside_the_top_pool():
    from cooking_assistant_chatbot.pricing.enuri_client import pick_relevant_cheapest

    items = [_item(f"햇반 백미 210g {i}", 10000 + i) for i in range(5)]
    items.append(_item("치즈크림 라떼 파우더", 500))  # cheap but far down the list

    picked = pick_relevant_cheapest(items, "밥")

    assert picked[0].title == "햇반 백미 210g 0"


def test_pick_relevant_cheapest_prefers_titles_naming_the_query():
    from cooking_assistant_chatbot.pricing.enuri_client import pick_relevant_cheapest

    items = [_item("파프리카 피망 5kg", 44890), _item("장난감 계산대", 30000), _item("피망 2kg", 18890)]

    assert [i.title for i in pick_relevant_cheapest(items, "피망")] == ["피망 2kg", "파프리카 피망 5kg"]


def test_pick_relevant_cheapest_falls_back_to_pool_for_spelling_variants():
    from cooking_assistant_chatbot.pricing.enuri_client import pick_relevant_cheapest

    items = [_item("유정란 60구", 27960), _item("계란 특란 30구", 18660)]

    assert pick_relevant_cheapest(items, "달걀")[0].title == "계란 특란 30구"
