from __future__ import annotations

from typing import Any

import litellm

from ..config import settings


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str | None = None,
    **kwargs: Any,
):
    """Thin wrapper around litellm.completion.

    Defaults to the local Ollama model configured in settings. Swapping to a
    cloud provider later is just passing a different `model` (e.g.
    "claude-3-5-sonnet-latest") — LiteLLM normalizes the request/response
    shape either way, so nothing else here needs to change. `api_base` is
    only set for Ollama models; cloud providers use their own default
    endpoints via API key.
    """
    target_model = model or settings.llm_model
    api_base = settings.ollama_base_url if target_model.startswith("ollama") else None

    return litellm.completion(
        model=target_model,
        messages=messages,
        tools=tools,
        api_base=api_base,
        **kwargs,
    )
