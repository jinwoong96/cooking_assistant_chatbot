import json

from cooking_assistant_chatbot.data.db import get_connection, upsert_recipes
from cooking_assistant_chatbot.data.models import Recipe


def _recipe(rcp_seq: str, name: str) -> Recipe:
    return Recipe(rcp_seq=rcp_seq, name=name, steps=["1단계"], step_image_urls=[""])


def test_upsert_inserts_new_recipe(tmp_path):
    conn = get_connection(str(tmp_path / "test.db"))

    upsert_recipes(conn, [_recipe("1", "김치찌개")])

    row = conn.execute(
        "SELECT name, steps_json FROM recipes WHERE rcp_seq = ?", ("1",)
    ).fetchone()
    assert row[0] == "김치찌개"
    assert json.loads(row[1]) == ["1단계"]


def test_upsert_updates_existing_recipe_on_conflict(tmp_path):
    conn = get_connection(str(tmp_path / "test.db"))
    upsert_recipes(conn, [_recipe("1", "김치찌개")])

    upsert_recipes(conn, [_recipe("1", "묵은지 김치찌개")])

    count = conn.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]
    name = conn.execute(
        "SELECT name FROM recipes WHERE rcp_seq = ?", ("1",)
    ).fetchone()[0]
    assert count == 1
    assert name == "묵은지 김치찌개"


def test_get_connection_creates_parent_directory(tmp_path):
    db_path = tmp_path / "nested" / "app.db"

    conn = get_connection(str(db_path))

    assert db_path.exists()
    conn.close()
