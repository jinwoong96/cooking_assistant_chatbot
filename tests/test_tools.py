from cooking_assistant_chatbot.agent import tools
from cooking_assistant_chatbot.data.db import get_connection, get_recipe_by_name, upsert_recipes
from cooking_assistant_chatbot.data.models import Recipe
from cooking_assistant_chatbot.pricing.enuri_client import ShoppingItem
from cooking_assistant_chatbot.pricing.price_lookup import IngredientPrice, RecipePriceEstimate


class _StubSearcher:
    def __init__(self, recipes: list[Recipe]):
        self._recipes = recipes

    def search(self, query: str, top_k: int = 1) -> list[Recipe]:
        return self._recipes[:top_k]


def _ctx(tmp_path, db_recipes, semantic=None):
    conn = get_connection(str(tmp_path / "test.db"))
    upsert_recipes(conn, db_recipes)
    return tools.ToolContext(
        searcher=_StubSearcher(semantic or []), price_client=None, conn=conn, report=lambda _s: None
    )


def test_get_recipe_by_name_ignores_spacing(tmp_path):
    conn = get_connection(str(tmp_path / "test.db"))
    upsert_recipes(conn, [Recipe(rcp_seq="1", name="새우 두부 계란찜")])

    assert get_recipe_by_name(conn, "새우두부계란찜").rcp_seq == "1"
    assert get_recipe_by_name(conn, "없는메뉴") is None


def test_get_recipe_prefers_exact_name_over_semantic_search(tmp_path):
    exact = Recipe(rcp_seq="1", name="김치찌개", ingredients_raw="김치 200g", steps=["끓인다"])
    other = Recipe(rcp_seq="2", name="참치김치찌개")
    ctx = _ctx(tmp_path, [exact, other], semantic=[other])

    result = tools.run_tool("get_recipe", '{"recipe_name": "김치찌개"}', ctx)

    assert result.startswith("레시피: 김치찌개")
    assert "끓인다" in result


def test_semantic_fallback_says_it_is_a_near_match(tmp_path):
    near = Recipe(rcp_seq="2", name="참치김치찌개")
    ctx = _ctx(tmp_path, [near], semantic=[near])

    result = tools.run_tool("get_recipe", '{"recipe_name": "김치찌개"}', ctx)

    assert "정확히 같은 이름은 없어서" in result
    assert "참치김치찌개" in result


def test_recipe_not_found(tmp_path):
    ctx = _ctx(tmp_path, [])

    result = tools.run_tool("get_nutrition", '{"recipe_name": "없는메뉴"}', ctx)

    assert "찾지 못함" in result


def test_recipe_detail_marks_missing_servings():
    detail = tools.format_recipe_detail("라면", Recipe(rcp_seq="1", name="라면"))

    assert "인분 수: 정보 없음" in detail


def test_nutrition_lists_available_values():
    recipe = Recipe(
        rcp_seq="1", name="두부조림", energy_kcal="210", sodium_mg="480", weight_info="250"
    )

    result = tools.format_nutrition("두부조림", recipe)

    assert "열량: 210kcal" in result
    assert "나트륨: 480mg" in result
    assert "1인분 중량: 250g" in result
    assert "단백질" not in result


def test_nutrition_says_when_recipe_has_none():
    result = tools.format_nutrition("떡볶이", Recipe(rcp_seq="10000recipe_1", name="떡볶이"))

    assert "영양성분 정보가 없음" in result


def test_cost_states_how_many_ingredients_the_portioned_total_covers():
    recipe = Recipe(rcp_seq="1", name="오므라이스")
    estimate = RecipePriceEstimate(
        recipe_name="오므라이스",
        ingredient_prices=[
            IngredientPrice(
                ingredient_name="밥",
                quantity_text="200g",
                cheapest_item=ShoppingItem(title="햇반 210g", price=1500),
                portioned_cost=1428,
            ),
            IngredientPrice(
                ingredient_name="달걀",
                quantity_text="2개",
                cheapest_item=ShoppingItem(title="달걀 30구", price=7000),
            ),
            IngredientPrice(ingredient_name="케첩", quantity_text="약간"),
        ],
    )

    result = tools.format_cost("오므라이스", recipe, estimate)

    assert "총액: 8,500원 (2개 재료 기준)" in result
    assert "원가 합계: 1,428원 (전체 3개 재료 중 1개만 계산됨" in result
    assert "가격을 못 찾은 재료: 케첩" in result


def test_unknown_tool_and_bad_arguments_do_not_raise(tmp_path):
    ctx = _ctx(tmp_path, [])

    assert "알 수 없는 도구" in tools.run_tool("nope", "{}", ctx)
    assert "찾지 못함" in tools.run_tool("get_recipe", "not json", ctx)
