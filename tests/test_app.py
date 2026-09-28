from cooking_assistant_chatbot import app
from cooking_assistant_chatbot.cooking import session as cooking
from cooking_assistant_chatbot.data.models import Recipe


def test_resolve_auth_returns_none_when_no_password_configured(monkeypatch):
    monkeypatch.setattr(app.settings, "app_password", "")

    assert app.resolve_auth() is None


def test_resolve_auth_returns_username_password_tuple_when_configured(monkeypatch):
    monkeypatch.setattr(app.settings, "app_username", "me")
    monkeypatch.setattr(app.settings, "app_password", "hunter2")

    assert app.resolve_auth() == ("me", "hunter2")


def test_plain_history_flattens_gradio_messages_for_the_llm():
    history = [
        {"role": "user", "content": "두부 요리 뭐 있어?"},
        {"role": "assistant", "content": [{"type": "text", "text": "두부조림 어때요?"}]},
        {"role": "assistant", "content": None},
    ]

    assert app._plain_history(history) == [
        {"role": "user", "content": "두부 요리 뭐 있어?"},
        {"role": "assistant", "content": "두부조림 어때요?"},
    ]


def test_progress_message_shows_status_and_whole_seconds():
    assert app._progress_message("재료 가격 조회 중 (3/18 · 두부)", 42.7) == {
        "role": "assistant",
        "content": "⏳ 재료 가격 조회 중 (3/18 · 두부) · 42초",
    }
    assert app._progress_message("", 0)["content"] == "⏳ 처리 중 · 0초"


def test_is_cooking_start_request():
    assert app.is_cooking_start_request("요리 시작")
    assert app.is_cooking_start_request("이제 요리 시작하자")
    assert app.is_cooking_start_request("요리 모드 켜줘")
    assert not app.is_cooking_start_request("된장찌개 레시피 알려줘")


def test_typed_cooking_commands_go_to_cooking_mode_only_while_it_is_on():
    session, _ = cooking.start(Recipe(rcp_seq="1", name="라면", steps=["물을 끓인다"]))

    assert app.is_typed_cooking_command("타이머 3분", session)
    assert app.is_typed_cooking_command("다음", session)
    assert not app.is_typed_cooking_command("대신 뭐 넣어도 돼?", session)
    assert not app.is_typed_cooking_command("타이머 3분", None)
