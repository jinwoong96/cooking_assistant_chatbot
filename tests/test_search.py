from cooking_assistant_chatbot.data.db import get_connection, upsert_recipes
from cooking_assistant_chatbot.data.models import Recipe
from cooking_assistant_chatbot.rag.indexer import build_index
from cooking_assistant_chatbot.rag.search import RecipeSearcher
from tests.fakes import FakeEmbeddingFunction


def _recipe(rcp_seq: str, name: str, ingredients: str = "") -> Recipe:
    return Recipe(
        rcp_seq=rcp_seq,
        name=name,
        ingredients_raw=ingredients,
        steps=["1. 끓인다."],
        step_image_urls=[""],
    )


def _setup(tmp_path):
    recipes = [
        _recipe("1", "김치찌개", ingredients="김치, 돼지고기, 두부"),
        _recipe("2", "된장찌개", ingredients="된장, 두부, 애호박"),
        _recipe("3", "토마토 파스타", ingredients="스파게티면, 토마토소스"),
    ]
    db_path = str(tmp_path / "test.db")
    conn = get_connection(db_path)
    upsert_recipes(conn, recipes)
    conn.close()

    chroma_path = str(tmp_path / "chroma")
    build_index(recipes, chroma_path, FakeEmbeddingFunction())
    return db_path, chroma_path


def test_search_returns_full_recipe_from_sqlite(tmp_path):
    db_path, chroma_path = _setup(tmp_path)
    searcher = RecipeSearcher(chroma_path, db_path, FakeEmbeddingFunction())

    results = searcher.search("김치찌개", top_k=1)

    assert len(results) == 1
    assert results[0].name == "김치찌개"
    assert results[0].steps == ["1. 끓인다."]


def test_search_ranks_closer_match_first(tmp_path):
    db_path, chroma_path = _setup(tmp_path)
    searcher = RecipeSearcher(chroma_path, db_path, FakeEmbeddingFunction())

    results = searcher.search("김치찌개 만드는 법", top_k=3)

    names = [r.name for r in results]
    assert names[0] == "김치찌개"
    assert "토마토 파스타" in names  # still returned, just not first


def test_search_returns_empty_list_when_collection_is_empty(tmp_path):
    db_path = str(tmp_path / "test.db")
    get_connection(db_path).close()
    chroma_path = str(tmp_path / "chroma")
    build_index([], chroma_path, FakeEmbeddingFunction())

    searcher = RecipeSearcher(chroma_path, db_path, FakeEmbeddingFunction())
    assert searcher.search("아무거나") == []
