import json

import httpx
import pytest
import respx

from cooking_assistant_chatbot.data.tenthousand_recipe_crawler import (
    BASE_URL,
    TenThousandRecipeCrawler,
)

_LIST_HTML = """
<html><body>
<a href="/recipe/111">떡볶이 1</a>
<a href="/recipe/222">떡볶이 2</a>
<a href="/recipe/111">떡볶이 1 (중복 링크)</a>
<a href="/recipe/333">떡볶이 3</a>
</body></html>
"""


def _recipe_json_ld(**overrides) -> str:
    data = {
        "@context": "http://schema.org/",
        "@type": "Recipe",
        "name": "백종원 떡볶이 만들기",
        "image": ["https://example.com/a.jpg", "https://example.com/b.jpg"],
        "recipeIngredient": ["떡 2컵", "물 2컵", "고추장 1스푼"],
        "recipeInstructions": [
            {"@type": "HowToStep", "text": "떡을 넣는다", "image": "https://example.com/s1.jpg"},
            {"@type": "HowToStep", "text": "끓인다"},
        ],
    }
    data.update(overrides)
    return f'<html><body><script type="application/ld+json">{json.dumps(data)}</script></body></html>'


@respx.mock
def test_search_recipe_ids_dedupes_and_preserves_order():
    respx.get(f"{BASE_URL}/recipe/list.html").mock(
        return_value=httpx.Response(200, text=_LIST_HTML)
    )
    crawler = TenThousandRecipeCrawler(min_request_interval=0)

    ids = crawler.search_recipe_ids("떡볶이")

    assert ids == ["111", "222", "333"]


@respx.mock
def test_search_recipe_ids_respects_limit():
    respx.get(f"{BASE_URL}/recipe/list.html").mock(
        return_value=httpx.Response(200, text=_LIST_HTML)
    )
    crawler = TenThousandRecipeCrawler(min_request_interval=0)

    ids = crawler.search_recipe_ids("떡볶이", limit=2)

    assert ids == ["111", "222"]


@respx.mock
def test_fetch_recipe_parses_schema_org_recipe():
    respx.get(f"{BASE_URL}/recipe/6829308").mock(
        return_value=httpx.Response(200, text=_recipe_json_ld())
    )
    crawler = TenThousandRecipeCrawler(min_request_interval=0)

    recipe = crawler.fetch_recipe("6829308")

    assert recipe.rcp_seq == "10000recipe_6829308"
    assert recipe.name == "백종원 떡볶이 만들기"
    assert recipe.ingredients_raw == "떡 2컵\n물 2컵\n고추장 1스푼"
    assert recipe.steps == ["떡을 넣는다", "끓인다"]
    assert recipe.step_image_urls == ["https://example.com/s1.jpg", ""]
    assert recipe.main_image_url == "https://example.com/a.jpg"


@respx.mock
def test_fetch_recipe_returns_none_when_no_json_ld_block():
    respx.get(f"{BASE_URL}/recipe/999").mock(
        return_value=httpx.Response(200, text="<html><body>no data</body></html>")
    )
    crawler = TenThousandRecipeCrawler(min_request_interval=0)

    assert crawler.fetch_recipe("999") is None


@respx.mock
def test_fetch_recipe_returns_none_for_non_recipe_json_ld():
    html = _recipe_json_ld(**{"@type": "BreadcrumbList"})
    respx.get(f"{BASE_URL}/recipe/999").mock(return_value=httpx.Response(200, text=html))
    crawler = TenThousandRecipeCrawler(min_request_interval=0)

    assert crawler.fetch_recipe("999") is None


def test_throttle_waits_at_least_min_interval(monkeypatch):
    times = iter([100.0, 100.3, 100.3])
    sleeps = []
    monkeypatch.setattr(
        "cooking_assistant_chatbot.data.tenthousand_recipe_crawler.time.monotonic",
        lambda: next(times),
    )
    monkeypatch.setattr(
        "cooking_assistant_chatbot.data.tenthousand_recipe_crawler.time.sleep", sleeps.append
    )
    crawler = TenThousandRecipeCrawler(min_request_interval=1.0)

    crawler._throttle()
    crawler._throttle()

    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(0.7)
