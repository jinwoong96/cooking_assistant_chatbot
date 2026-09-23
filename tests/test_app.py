from cooking_assistant_chatbot import app


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
