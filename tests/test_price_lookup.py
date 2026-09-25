from cooking_assistant_chatbot.data.db import get_connection
from cooking_assistant_chatbot.data.models import Recipe
from cooking_assistant_chatbot.pricing.enuri_client import ShoppingItem
from cooking_assistant_chatbot.pricing.price_lookup import estimate_recipe_price


class _StubClient:
    def __init__(self, prices: dict[str, int]):
        self._prices = prices
        self.call_count = 0

    def search(self, query: str) -> list[ShoppingItem]:
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


class _StubClientWithSizedProducts:
    """Stub whose product titles carry a real package size, for portioned-cost tests."""

    def __init__(self, data: dict[str, tuple[int, str]]):
        self._data = data

    def search(self, query: str) -> list[ShoppingItem]:
        if query not in self._data:
            return []
        price, title = self._data[query]
        return [ShoppingItem(title=title, price=price, brand="테스트몰", link="")]


def test_estimate_recipe_price_computes_portioned_cost_when_units_match(tmp_path):
    recipe = Recipe(rcp_seq="1", name="테스트", ingredients_raw="다진마늘(10g)")
    client = _StubClientWithSizedProducts({"다진마늘": (2000, "청정원 다진마늘 500g")})

    estimate = estimate_recipe_price(recipe, client, _conn(tmp_path))

    assert estimate.ingredient_prices[0].portioned_cost == 40  # 2000 * (10/500)
    assert estimate.total_portioned_cost == 40
    assert estimate.ingredients_missing_portioned_cost == []


def test_estimate_recipe_price_leaves_portioned_cost_none_when_product_has_no_size(tmp_path):
    recipe = Recipe(rcp_seq="1", name="테스트", ingredients_raw="청양고추(3g)")
    client = _StubClientWithSizedProducts(
        {"청양고추": (980, "이마트 소소한 하루 청양고추")}
    )

    estimate = estimate_recipe_price(recipe, client, _conn(tmp_path))

    assert estimate.ingredient_prices[0].portioned_cost is None
    assert estimate.ingredients_missing_portioned_cost == ["청양고추"]


def test_estimate_recipe_price_leaves_portioned_cost_none_when_units_mismatch(tmp_path):
    recipe = Recipe(rcp_seq="1", name="테스트", ingredients_raw="참기름(20g)")
    client = _StubClientWithSizedProducts({"참기름": (4800, "오뚜기 옛날 참기름 320ml")})

    estimate = estimate_recipe_price(recipe, client, _conn(tmp_path))

    assert estimate.ingredient_prices[0].portioned_cost is None


class _RecordingClient:
    """Stub that records every query it's asked to search, and returns a
    canned result only for queries in `known`."""

    def __init__(self, known: dict[str, int]):
        self._known = known
        self.queries: list[str] = []

    def search(self, query: str) -> list[ShoppingItem]:
        self.queries.append(query)
        if query not in self._known:
            return []
        return [ShoppingItem(title=f"{query} 상품", price=self._known[query])]


def test_estimate_recipe_price_corrects_ambiguous_name_before_searching(tmp_path):
    recipe = Recipe(rcp_seq="1", name="테스트", ingredients_raw="얇게 썬 돼지고기(100g)")
    client = _RecordingClient({"돼지고기": 5000})

    def correct_name(name, context):
        assert name == "얇게 썬 돼지고기"
        return "돼지고기"

    estimate = estimate_recipe_price(
        recipe, client, _conn(tmp_path), correct_name=correct_name
    )

    assert client.queries == ["돼지고기"]  # corrected before ever searching the raw name
    assert estimate.ingredient_prices[0].cheapest_item.price == 5000


def test_estimate_recipe_price_corrects_as_fallback_after_empty_search(tmp_path):
    recipe = Recipe(rcp_seq="1", name="테스트", ingredients_raw="이상한재료(10g)")
    client = _RecordingClient({"양파": 1000})

    estimate = estimate_recipe_price(
        recipe, client, _conn(tmp_path), correct_name=lambda name, ctx: "양파"
    )

    assert client.queries == ["이상한재료", "양파"]
    assert estimate.ingredient_prices[0].cheapest_item.price == 1000


def test_estimate_recipe_price_skips_ingredient_when_correction_says_not_real(tmp_path):
    recipe = Recipe(rcp_seq="1", name="테스트", ingredients_raw="간 맞출 때")
    client = _RecordingClient({})

    estimate = estimate_recipe_price(
        recipe, client, _conn(tmp_path), correct_name=lambda name, ctx: None
    )

    assert client.queries == []  # never searched at all
    assert estimate.ingredient_prices[0].cheapest_item is None


def test_estimate_recipe_price_without_correct_name_behaves_as_before(tmp_path):
    recipe = Recipe(rcp_seq="1", name="테스트", ingredients_raw="얇게 썬 돼지고기(100g)")
    client = _RecordingClient({})

    estimate = estimate_recipe_price(recipe, client, _conn(tmp_path))

    assert client.queries == ["얇게 썬 돼지고기"]
    assert estimate.ingredient_prices[0].cheapest_item is None


def test_estimate_recipe_price_reports_progress_per_ingredient(tmp_path):
    recipe = Recipe(rcp_seq="1", name="김치찌개", ingredients_raw="김치 200g, 두부 100g")
    client = _StubClient({"김치": 3000, "두부": 1500})
    calls = []

    estimate_recipe_price(
        recipe, client, _conn(tmp_path), on_progress=lambda *args: calls.append(args)
    )

    assert calls == [(1, 2, "김치"), (2, 2, "두부")]


def test_cache_is_kept_per_price_basis(tmp_path):
    class _Sized:
        def __init__(self):
            self.calls = 0

        def search(self, query: str) -> list[ShoppingItem]:
            self.calls += 1
            return [ShoppingItem(title="김치 10kg", price=9000), ShoppingItem(title="김치 500g", price=3000)]

    recipe = Recipe(rcp_seq="1", name="김치찌개", ingredients_raw="김치 200g")
    client = _Sized()
    conn = _conn(tmp_path)

    small = estimate_recipe_price(recipe, client, conn, basis="min_spend")
    value = estimate_recipe_price(recipe, client, conn, basis="unit_price")
    small_again = estimate_recipe_price(recipe, client, conn, basis="min_spend")

    assert small.ingredient_prices[0].cheapest_item.title == "김치 500g"
    assert value.ingredient_prices[0].cheapest_item.title == "김치 10kg"
    assert small_again.ingredient_prices[0].cheapest_item.title == "김치 500g"
    assert client.calls == 2
