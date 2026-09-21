from cooking_assistant_chatbot.agent import ingredient_correction
from tests.fakes import fake_text_response, fake_tool_call_response


def test_correct_ingredient_name_returns_cleaned_name(monkeypatch):
    monkeypatch.setattr(
        ingredient_correction,
        "chat",
        lambda **kwargs: fake_tool_call_response(
            "resolve_ingredient_name",
            {"is_purchasable_ingredient": True, "cleaned_name": "돼지고기"},
        ),
    )

    result = ingredient_correction.correct_ingredient_name(
        "얇게 썬 돼지고기", "얇게 썬 돼지고기, 양파(50g)"
    )

    assert result == "돼지고기"


def test_correct_ingredient_name_returns_none_when_not_a_real_ingredient(monkeypatch):
    monkeypatch.setattr(
        ingredient_correction,
        "chat",
        lambda **kwargs: fake_tool_call_response(
            "resolve_ingredient_name", {"is_purchasable_ingredient": False}
        ),
    )

    result = ingredient_correction.correct_ingredient_name("간 맞출 때", "소금, 간 맞출 때")

    assert result is None


def test_correct_ingredient_name_falls_back_to_original_when_no_tool_call(monkeypatch):
    monkeypatch.setattr(
        ingredient_correction, "chat", lambda **kwargs: fake_text_response("그냥 텍스트")
    )

    result = ingredient_correction.correct_ingredient_name("애매한 재료", "애매한 재료")

    assert result == "애매한 재료"


def test_correct_ingredient_name_falls_back_when_cleaned_name_missing(monkeypatch):
    monkeypatch.setattr(
        ingredient_correction,
        "chat",
        lambda **kwargs: fake_tool_call_response(
            "resolve_ingredient_name", {"is_purchasable_ingredient": True}
        ),
    )

    result = ingredient_correction.correct_ingredient_name("애매한 재료", "애매한 재료")

    assert result == "애매한 재료"
