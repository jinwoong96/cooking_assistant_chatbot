from cooking_assistant_chatbot.agent import pipeline
from cooking_assistant_chatbot.agent.router import RouteResult
from cooking_assistant_chatbot.data.db import get_connection
from cooking_assistant_chatbot.data.models import Recipe
from cooking_assistant_chatbot.pricing.enuri_client import ShoppingItem
from tests.fakes import fake_text_response


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

    assert reply == "맛있는 김치찌개예요! 총 1000원 정도 들어요."


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

    assert "존재하지않는메뉴" in reply
    assert "찾지 못했" in reply


def test_handle_message_routes_to_general_chat(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "route", lambda msg: RouteResult(intent="general_chat"))
    monkeypatch.setattr(pipeline, "chat", lambda **kwargs: fake_text_response("안녕하세요!"))
    conn = get_connection(str(tmp_path / "test.db"))

    reply = pipeline.handle_message("안녕", _StubSearcher([]), _StubPriceClient(), conn)

    assert reply == "안녕하세요!"
