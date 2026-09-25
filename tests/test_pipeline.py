from cooking_assistant_chatbot.agent import pipeline
from cooking_assistant_chatbot.agent.router import RouteResult
from cooking_assistant_chatbot.data.db import get_connection
from cooking_assistant_chatbot.data.models import Recipe
from cooking_assistant_chatbot.pricing.enuri_client import ShoppingItem
from tests.fakes import fake_text_response, fake_tool_call_response


class _StubSearcher:
    def __init__(self, recipes: list[Recipe]):
        self._recipes = recipes

    def search(self, query: str, top_k: int = 1) -> list[Recipe]:
        return self._recipes[:top_k]


class _StubPriceClient:
    def search_cheapest(self, query: str, limit: int = 1) -> list[ShoppingItem]:
        return [ShoppingItem(title=f"{query} 상품", price=1000)]


def _recipe() -> Recipe:
    return Recipe(
        rcp_seq="1", name="김치찌개", ingredients_raw="김치 200g", steps=["1. 끓인다"]
    )


def test_handle_message_routes_to_recipe_flow_and_composes_reply(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline, "route", lambda msg: RouteResult(intent="recipe_price", menu_name="김치찌개")
    )
    monkeypatch.setattr(
        pipeline, "chat", lambda **kwargs: fake_text_response("맛있는 김치찌개예요! 총 1000원 정도 들어요.")
    )
    conn = get_connection(str(tmp_path / "test.db"))

    reply = pipeline.handle_message(
        "김치찌개 해먹고 싶어", _StubSearcher([_recipe()]), _StubPriceClient(), conn
    )

    assert reply.text == "맛있는 김치찌개예요! 총 1000원 정도 들어요."


def test_handle_message_reports_when_recipe_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "route",
        lambda msg: RouteResult(intent="recipe_price", menu_name="존재하지않는메뉴"),
    )
    conn = get_connection(str(tmp_path / "test.db"))

    reply = pipeline.handle_message(
        "존재하지않는메뉴 해줘", _StubSearcher([]), _StubPriceClient(), conn
    )

    assert "존재하지않는메뉴" in reply.text
    assert "찾지 못했" in reply.text


def test_handle_message_routes_to_general_chat(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "route", lambda msg: RouteResult(intent="general_chat"))
    monkeypatch.setattr(pipeline, "chat", lambda **kwargs: fake_text_response("안녕하세요!"))
    conn = get_connection(str(tmp_path / "test.db"))

    reply = pipeline.handle_message("안녕", _StubSearcher([]), _StubPriceClient(), conn)

    assert reply.text == "안녕하세요!"


def test_general_chat_includes_prior_history_as_messages(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "route", lambda msg: RouteResult(intent="general_chat"))
    captured = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return fake_text_response("이어서 답할게요")

    monkeypatch.setattr(pipeline, "chat", fake_chat)
    conn = get_connection(str(tmp_path / "test.db"))
    history = [
        {"role": "user", "content": "두부 요리 뭐 있어?"},
        {"role": "assistant", "content": "두부조림 어때요?"},
    ]

    reply = pipeline.handle_message(
        "그거 말고 다른 건?", _StubSearcher([]), _StubPriceClient(), conn, history=history
    )

    assert reply.text == "이어서 답할게요"
    assert history[0] in captured["messages"]
    assert history[1] in captured["messages"]


def test_general_chat_uses_style_search_tool_and_grounds_final_reply(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "route", lambda msg: RouteResult(intent="general_chat"))
    recipe = _recipe()
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return fake_tool_call_response("search_recipes_by_style", {"query": "매콤한 국물요리"})
        return fake_text_response(f"{recipe.name} 어때요?")

    monkeypatch.setattr(pipeline, "chat", fake_chat)
    conn = get_connection(str(tmp_path / "test.db"))

    reply = pipeline.handle_message(
        "매콤한 국물요리 추천해줘",
        _StubSearcher([recipe]),
        _StubPriceClient(),
        conn,
    )

    assert reply.text == f"{recipe.name} 어때요?"
    assert len(calls) == 2
    tool_messages = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert recipe.name in tool_messages[0]["content"]


def test_general_chat_uses_ingredients_search_tool_for_exact_match(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "route", lambda msg: RouteResult(intent="general_chat"))
    conn = get_connection(str(tmp_path / "test.db"))
    from cooking_assistant_chatbot.data.db import upsert_recipes
    from cooking_assistant_chatbot.data.ingredient_index import build_ingredient_index

    exact_match = Recipe(rcp_seq="10", name="두부계란찜", ingredients_raw="두부 100g, 계란 2개")
    upsert_recipes(conn, [exact_match])
    build_ingredient_index(conn)

    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return fake_tool_call_response(
                "search_recipes_by_ingredients", {"ingredients": ["두부", "계란"]}
            )
        return fake_text_response("두부계란찜 어때요?")

    monkeypatch.setattr(pipeline, "chat", fake_chat)

    reply = pipeline.handle_message(
        "냉장고에 두부랑 계란 있는데 뭐 해먹지?",
        _StubSearcher([]),  # semantic search deliberately returns nothing
        _StubPriceClient(),
        conn,
    )

    assert reply.text == "두부계란찜 어때요?"
    tool_messages = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
    assert "두부계란찜" in tool_messages[0]["content"]


def test_recipe_reply_speech_is_template_summary_not_llm_text(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline, "route", lambda msg: RouteResult(intent="recipe_price", menu_name="김치찌개")
    )
    monkeypatch.setattr(pipeline, "chat", lambda **kwargs: fake_text_response("**긴 마크다운 답변**"))
    conn = get_connection(str(tmp_path / "test.db"))

    reply = pipeline.handle_message(
        "김치찌개 해먹고 싶어", _StubSearcher([_recipe()]), _StubPriceClient(), conn
    )

    assert reply.speech == (
        "김치찌개 레시피예요. 재료를 전부 새로 사면 약 1,000원 정도예요. "
        "조리 순서는 1단계이고 화면에 정리해뒀어요."
    )


def test_general_chat_speech_strips_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "route", lambda msg: RouteResult(intent="general_chat"))
    monkeypatch.setattr(
        pipeline, "chat", lambda **kwargs: fake_text_response("### 팁\n- **칼**은 날카롭게 😊")
    )
    conn = get_connection(str(tmp_path / "test.db"))

    reply = pipeline.handle_message("요리 팁", _StubSearcher([]), _StubPriceClient(), conn)

    assert reply.speech == "팁\n칼은 날카롭게"


def _estimate(total_price: int, portioned: int | None):
    from cooking_assistant_chatbot.pricing.price_lookup import (
        IngredientPrice,
        RecipePriceEstimate,
    )

    return RecipePriceEstimate(
        recipe_name="x",
        ingredient_prices=[
            IngredientPrice(
                ingredient_name="재료",
                quantity_text="",
                cheapest_item=ShoppingItem(title="재료", price=total_price),
                portioned_cost=portioned,
            )
        ],
    )


def test_recipe_speech_summary_includes_servings_and_portioned_cost():
    recipe = Recipe(
        rcp_seq="1",
        name="들깨삼겹살",
        ingredients_raw="[ 2인분 ] 삼겹살(200g)",
        steps=["a", "b", "c"],
    )

    speech = pipeline.recipe_speech_summary(recipe, _estimate(51927, 5074))

    assert speech == (
        "들깨삼겹살 레시피예요. 2인분 기준이에요. "
        "재료를 전부 새로 사면 약 51,927원이고, 이번에 쓰는 양만 치면 약 5,074원 정도예요. "
        "조리 순서는 3단계이고 화면에 정리해뒀어요."
    )


def test_recipe_speech_summary_cleans_crawled_title_tags():
    recipe = Recipe(rcp_seq="1", name="[자취요리] 초간단 떡볶이 황금 레시피", steps=["a"])

    speech = pipeline.recipe_speech_summary(recipe, _estimate(0, None))

    assert speech.startswith("초간단 떡볶이 황금 레시피예요.")
    assert "원" not in speech  # no price sentence when nothing was priced


def test_recipe_path_reports_each_stage_including_ingredient_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline, "route", lambda msg: RouteResult(intent="recipe_price", menu_name="김치찌개")
    )
    monkeypatch.setattr(pipeline, "chat", lambda **kwargs: fake_text_response("답변"))
    conn = get_connection(str(tmp_path / "test.db"))
    statuses = []

    pipeline.handle_message(
        "김치찌개 해먹고 싶어",
        _StubSearcher([_recipe()]),
        _StubPriceClient(),
        conn,
        on_progress=statuses.append,
    )

    assert statuses == [
        "요청 이해하는 중",
        "'김치찌개' 레시피 찾는 중",
        "재료 가격 조회 중 (1/1 · 김치)",
        "답변 작성 중",
    ]


def test_general_chat_reports_db_search_when_a_tool_is_called(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "route", lambda msg: RouteResult(intent="general_chat"))
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return fake_tool_call_response("search_recipes_by_style", {"query": "국물"})
        return fake_text_response("추천")

    monkeypatch.setattr(pipeline, "chat", fake_chat)
    conn = get_connection(str(tmp_path / "test.db"))
    statuses = []

    pipeline.handle_message(
        "국물요리 추천", _StubSearcher([_recipe()]), _StubPriceClient(), conn,
        on_progress=statuses.append,
    )

    assert statuses == [
        "요청 이해하는 중",
        "답변 생각하는 중",
        "레시피 DB에서 찾는 중",
        "찾은 레시피로 답변 작성 중",
    ]
