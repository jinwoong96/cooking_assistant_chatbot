from cooking_assistant_chatbot.agent import router
from tests.fakes import fake_text_response, fake_tool_call_response


def test_route_returns_recipe_price_intent_with_menu_name(monkeypatch):
    monkeypatch.setattr(
        router,
        "chat",
        lambda **kwargs: fake_tool_call_response(
            "handle_recipe_request", {"menu_name": "김치찌개"}
        ),
    )

    result = router.route("김치찌개 해먹고 싶어")

    assert result.intent == "recipe_price"
    assert result.menu_name == "김치찌개"


def test_route_returns_general_chat_intent(monkeypatch):
    monkeypatch.setattr(
        router, "chat", lambda **kwargs: fake_tool_call_response("general_chat", {})
    )

    result = router.route("오늘 기분이 어때?")

    assert result.intent == "general_chat"
    assert result.menu_name is None


def test_route_defaults_to_general_chat_when_model_returns_no_tool_call(monkeypatch):
    monkeypatch.setattr(router, "chat", lambda **kwargs: fake_text_response("그냥 텍스트 응답"))

    result = router.route("아무 말")

    assert result.intent == "general_chat"


def test_route_falls_back_to_raw_message_when_menu_name_missing(monkeypatch):
    monkeypatch.setattr(
        router,
        "chat",
        lambda **kwargs: fake_tool_call_response("handle_recipe_request", {}),
    )

    result = router.route("된장찌개 해먹고 싶어")

    assert result.intent == "recipe_price"
    assert result.menu_name == "된장찌개 해먹고 싶어"
