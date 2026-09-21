from cooking_assistant_chatbot import app


def test_resolve_auth_returns_none_when_no_password_configured(monkeypatch):
    monkeypatch.setattr(app.settings, "app_password", "")

    assert app.resolve_auth() is None


def test_resolve_auth_returns_username_password_tuple_when_configured(monkeypatch):
    monkeypatch.setattr(app.settings, "app_username", "me")
    monkeypatch.setattr(app.settings, "app_password", "hunter2")

    assert app.resolve_auth() == ("me", "hunter2")
