from cooking_assistant_chatbot.data.db import get_connection, upsert_recipes
from cooking_assistant_chatbot.data.ingredient_index import (
    build_ingredient_index,
    find_recipes_by_ingredients,
)
from cooking_assistant_chatbot.data.models import Recipe


def _recipe(rcp_seq: str, name: str, ingredients_raw: str) -> Recipe:
    return Recipe(rcp_seq=rcp_seq, name=name, ingredients_raw=ingredients_raw)


def _seed(tmp_path):
    conn = get_connection(str(tmp_path / "test.db"))
    upsert_recipes(
        conn,
        [
            _recipe("1", "두부계란찜", "두부 100g, 계란 2개, 대파 10g"),
            _recipe("2", "두부조림", "두부 200g, 간장 10g, 대파 10g"),
            _recipe("3", "계란찜", "계란 3개, 물 100g"),
            _recipe("4", "두부계란볶음", "두부 100g, 계란 2개"),
        ],
    )
    build_ingredient_index(conn)
    return conn


def test_build_ingredient_index_returns_total_entry_count(tmp_path):
    conn = _seed(tmp_path)

    count = conn.execute("SELECT COUNT(*) FROM recipe_ingredients").fetchone()[0]

    assert count == 3 + 3 + 2 + 2


def test_find_recipes_by_ingredients_requires_all_terms(tmp_path):
    conn = _seed(tmp_path)

    results = find_recipes_by_ingredients(conn, ["두부", "계란"])

    names = {r.name for r in results}
    assert names == {"두부계란찜", "두부계란볶음"}


def test_find_recipes_by_ingredients_ranks_fewer_ingredients_first(tmp_path):
    conn = _seed(tmp_path)

    results = find_recipes_by_ingredients(conn, ["두부", "계란"])

    assert results[0].name == "두부계란볶음"  # 2 ingredients vs. 3


def test_find_recipes_by_ingredients_matches_substring(tmp_path):
    conn = _seed(tmp_path)
    upsert_recipes(
        conn, [_recipe("5", "다진계란국", "다진 계란 50g, 대파 5g")]
    )
    build_ingredient_index(conn)

    results = find_recipes_by_ingredients(conn, ["계란"])

    assert "다진계란국" in {r.name for r in results}


def test_find_recipes_by_ingredients_returns_empty_when_no_recipe_has_all_terms(tmp_path):
    conn = _seed(tmp_path)

    assert find_recipes_by_ingredients(conn, ["두부", "존재하지않는재료"]) == []


def test_find_recipes_by_ingredients_returns_empty_for_no_terms(tmp_path):
    conn = _seed(tmp_path)

    assert find_recipes_by_ingredients(conn, []) == []


def test_build_ingredient_index_is_idempotent_on_rerun(tmp_path):
    conn = _seed(tmp_path)

    count_again = build_ingredient_index(conn)

    total = conn.execute("SELECT COUNT(*) FROM recipe_ingredients").fetchone()[0]
    assert count_again == total  # no duplicate accumulation from re-running
