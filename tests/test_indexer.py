import chromadb

from cooking_assistant_chatbot.rag.indexer import (
    COLLECTION_NAME,
    build_index,
    recipe_document_text,
)
from cooking_assistant_chatbot.data.models import Recipe
from tests.fakes import FakeEmbeddingFunction


def _recipe(rcp_seq: str, name: str, ingredients: str = "", category: str = "") -> Recipe:
    return Recipe(
        rcp_seq=rcp_seq, name=name, ingredients_raw=ingredients, category=category
    )


def test_recipe_document_text_includes_populated_fields_only():
    text = recipe_document_text(_recipe("1", "김치찌개", ingredients="김치, 돼지고기"))

    assert "김치찌개" in text
    assert "재료: 김치, 돼지고기" in text
    assert "카테고리:" not in text


def test_build_index_returns_count_and_upserts_all_recipes(tmp_path):
    recipes = [_recipe(str(i), f"레시피{i}") for i in range(5)]

    count = build_index(recipes, str(tmp_path), FakeEmbeddingFunction())

    assert count == 5
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_collection(
        COLLECTION_NAME, embedding_function=FakeEmbeddingFunction()
    )
    assert collection.count() == 5


def test_build_index_rebuilds_from_scratch_on_second_call(tmp_path):
    build_index([_recipe("1", "김치찌개")], str(tmp_path), FakeEmbeddingFunction())

    count = build_index([_recipe("2", "된장찌개")], str(tmp_path), FakeEmbeddingFunction())

    assert count == 1
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_collection(
        COLLECTION_NAME, embedding_function=FakeEmbeddingFunction()
    )
    assert collection.count() == 1
