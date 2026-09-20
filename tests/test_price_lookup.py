from cooking_assistant_chatbot.data.db import get_connection
from cooking_assistant_chatbot.data.models import Recipe
from cooking_assistant_chatbot.pricing.enuri_client import ShoppingItem
from cooking_assistant_chatbot.pricing.price_lookup import estimate_recipe_price


class _StubClient:
    def __init__(self, prices: dict[str, int]):
        self._prices = prices
        self.call_count = 0

    def search_cheapest(self, query: str, limit: int = 1) -> list[ShoppingItem]:
        self.call_count += 1
        if query not in self._prices:
            return []
        return [
            ShoppingItem(title=f"{query} 상품", price=self._prices[query], brand="테스트몰", link="")
        ]


def _conn(tmp_path):
    return get_connection(str(tmp_path / "test.db"))


def test_estimate_recipe_price_sums_cheapest_item_per_ingredient(tmp_path):
    recipe = Recipe(
        rcp_seq="1",
        name="김치찌개",
        ingredients_raw="김치 200g, 돼지고기 150g, 두부 100g",
    )
    client = _StubClient({"김치": 3000, "돼지고기": 5000, "두부": 1500})

    estimate = estimate_recipe_price(recipe, client, _conn(tmp_path))

    assert estimate.total_price == 9500
    assert estimate.unresolved_ingredients == []


def test_estimate_recipe_price_tracks_unresolved_ingredients(tmp_path):
    recipe = Recipe(
        rcp_seq="1", name="김치찌개", ingredients_raw="김치 200g, 신비한재료 10g"
    )
    client = _StubClient({"김치": 3000})

    estimate = estimate_recipe_price(recipe, client, _conn(tmp_path))

    assert estimate.total_price == 3000
    assert estimate.unresolved_ingredients == ["신비한재료"]


def test_estimate_recipe_price_handles_no_ingredients(tmp_path):
    recipe = Recipe(rcp_seq="1", name="물 한 잔", ingredients_raw="")
    client = _StubClient({})

    estimate = estimate_recipe_price(recipe, client, _conn(tmp_path))

    assert estimate.ingredient_prices == []
    assert estimate.total_price == 0


def test_estimate_recipe_price_uses_cache_on_second_call(tmp_path):
    recipe = Recipe(rcp_seq="1", name="김치찌개", ingredients_raw="김치 200g")
    client = _StubClient({"김치": 3000})
    conn = _conn(tmp_path)

    estimate_recipe_price(recipe, client, conn)
    estimate_recipe_price(recipe, client, conn)

    assert client.call_count == 1
