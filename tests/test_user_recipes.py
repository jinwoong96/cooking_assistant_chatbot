import pytest

from cooking_assistant_chatbot.data import user_recipes
from cooking_assistant_chatbot.data.db import get_all_recipes, get_connection
from cooking_assistant_chatbot.data.user_recipes import PermissionDenied, RecipeDraft


def _conn(tmp_path):
    return get_connection(str(tmp_path / "test.db"))


def _draft(**overrides):
    values = dict(
        name="간장계란밥",
        servings=1,
        ingredients=["밥 1공기", "계란 2개", ""],
        steps=["1. 계란을 부친다.", "2) 밥에 올려 비빈다."],
        nutrition={"energy_kcal": "550"},
    )
    values.update(overrides)
    return RecipeDraft(**values)


def test_save_cleans_and_stores_a_post(tmp_path):
    conn = _conn(tmp_path)

    saved = user_recipes.save(conn, _draft(), "진웅")

    assert saved.author == "진웅"
    assert saved.draft.ingredients == ["밥 1공기", "계란 2개"]
    assert saved.draft.steps == ["계란을 부친다.", "밥에 올려 비빈다."]  # typed numbers dropped
    assert user_recipes.get(conn, saved.id).draft.nutrition == {"energy_kcal": "550"}


def test_posts_stay_out_of_the_searchable_recipes_table(tmp_path):
    conn = _conn(tmp_path)

    user_recipes.save(conn, _draft(), "진웅")

    assert get_all_recipes(conn) == []


def test_validation_messages(tmp_path):
    conn = _conn(tmp_path)

    with pytest.raises(ValueError) as error:
        user_recipes.save(conn, RecipeDraft(nutrition={"fat_g": "많이"}), "")

    message = str(error.value)
    for expected in ["내 이름", "메뉴 이름", "재료", "조리 순서", "숫자로"]:
        assert expected in message


def test_only_the_author_can_edit_or_delete(tmp_path):
    conn = _conn(tmp_path)
    saved = user_recipes.save(conn, _draft(), "진웅")

    with pytest.raises(PermissionDenied):
        user_recipes.save(conn, _draft(name="남의 수정"), "다른사람", saved.id)
    with pytest.raises(PermissionDenied):
        user_recipes.delete(conn, saved.id, "다른사람")

    edited = user_recipes.save(conn, _draft(name="간장계란밥 v2"), " 진웅 ", saved.id)
    assert edited.draft.name == "간장계란밥 v2"
    user_recipes.delete(conn, saved.id, "진웅")
    assert user_recipes.get(conn, saved.id) is None


def test_search_by_title_or_author_newest_first(tmp_path):
    conn = _conn(tmp_path)
    first = user_recipes.save(conn, _draft(name="김치 볶음밥"), "진웅")
    second = user_recipes.save(conn, _draft(name="두부조림"), "룸메")

    assert [r.id for r in user_recipes.search(conn)] == [second.id, first.id]
    assert [r.id for r in user_recipes.search(conn, "김치볶음")] == [first.id]
    assert [r.id for r in user_recipes.search(conn, "룸메")] == [second.id]


def test_render_shows_author_steps_and_optional_nutrition(tmp_path):
    conn = _conn(tmp_path)
    saved = user_recipes.save(conn, _draft(), "진웅")

    text = user_recipes.render(saved)

    assert "## 간장계란밥" in text and "✍️ 진웅" in text and "**1인분**" in text
    assert "1. 계란을 부친다." in text and "- 밥 1공기" in text
    assert "열량 550kcal" in text
    no_nutrition = user_recipes.save(conn, _draft(nutrition={}), "진웅")
    assert "영양성분" not in user_recipes.render(no_nutrition)
