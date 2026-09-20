from cooking_assistant_chatbot.llm import client


def test_chat_defaults_to_configured_ollama_model_and_api_base(monkeypatch):
    monkeypatch.setattr(client.settings, "llm_model", "ollama_chat/qwen3:14b")
    monkeypatch.setattr(client.settings, "ollama_base_url", "http://localhost:11434")
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(client.litellm, "completion", fake_completion)

    result = client.chat(messages=[{"role": "user", "content": "hi"}])

    assert result == "ok"
    assert captured["model"] == "ollama_chat/qwen3:14b"
    assert captured["api_base"] == "http://localhost:11434"
    assert captured["messages"] == [{"role": "user", "content": "hi"}]


def test_chat_omits_api_base_for_non_ollama_model(monkeypatch):
    captured = {}
    monkeypatch.setattr(client.litellm, "completion", lambda **kwargs: captured.update(kwargs))

    client.chat(messages=[], model="claude-3-5-sonnet-latest")

    assert captured["model"] == "claude-3-5-sonnet-latest"
    assert captured["api_base"] is None


def test_chat_passes_tools_through(monkeypatch):
    captured = {}
    monkeypatch.setattr(client.litellm, "completion", lambda **kwargs: captured.update(kwargs))
    tools = [{"type": "function", "function": {"name": "search_recipe"}}]

    client.chat(messages=[], tools=tools)

    assert captured["tools"] == tools
