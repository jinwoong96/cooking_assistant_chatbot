from cooking_assistant_chatbot.data import crawl_supplemental_recipes as module
from cooking_assistant_chatbot.data.db import get_connection
from cooking_assistant_chatbot.data.models import Recipe


def test_run_upserts_valid_recipes_and_skips_incomplete_ones(tmp_path, monkeypatch):
    good = Recipe(
        rcp_seq="10000recipe_1",
        name="떡볶이",
        ingredients_raw="떡 2컵",
        steps=["끓인다"],
    )
    no_steps = Recipe(rcp_seq="10000recipe_2", name="빈 레시피", ingredients_raw="떡 2컵", steps=[])

    class Crawler:
        def __init__(self):
            self.fetched = []

        def search_recipe_ids(self, keyword, limit=10):
            return ["1", "2"]

        def fetch_recipe(self, recipe_id):
            self.fetched.append(recipe_id)
            return {"1": good, "2": no_steps}.get(recipe_id)

        def close(self):
            pass

    monkeypatch.setattr(module, "TenThousandRecipeCrawler", Crawler)
    monkeypatch.setattr(module.settings, "db_path", str(tmp_path / "test.db"))

    count = module.run(keywords=["떡볶이"], per_keyword=10)

    assert count == 1
    conn = get_connection(str(tmp_path / "test.db"))
    row = conn.execute(
        "SELECT name FROM recipes WHERE rcp_seq = ?", ("10000recipe_1",)
    ).fetchone()
    assert row[0] == "떡볶이"
    assert conn.execute(
        "SELECT * FROM recipes WHERE rcp_seq = ?", ("10000recipe_2",)
    ).fetchone() is None


def test_run_handles_fetch_returning_none(tmp_path, monkeypatch):
    class Crawler:
        def search_recipe_ids(self, keyword, limit=10):
            return ["404"]

        def fetch_recipe(self, recipe_id):
            return None

        def close(self):
            pass

    monkeypatch.setattr(module, "TenThousandRecipeCrawler", Crawler)
    monkeypatch.setattr(module.settings, "db_path", str(tmp_path / "test.db"))

    count = module.run(keywords=["떡볶이"], per_keyword=10)

    assert count == 0
