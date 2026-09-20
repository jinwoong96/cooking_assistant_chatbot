from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from ..llm.client import chat

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "handle_recipe_request",
            "description": (
                "사용자가 특정 메뉴의 레시피나 재료 가격을 알고 싶어할 때 호출한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "menu_name": {
                        "type": "string",
                        "description": "사용자가 요청한 정확한 메뉴/요리 이름 (예: '김치찌개')",
                    }
                },
                "required": ["menu_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "general_chat",
            "description": (
                "특정 메뉴의 레시피/가격 요청이 아닌, 요리에 대한 일반적인 대화나 잡담일 때 호출한다."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_SYSTEM_PROMPT = (
    "너는 요리 챗봇의 라우터다. 사용자 메시지를 보고 반드시 둘 중 하나의 도구를 호출해라: "
    "특정 메뉴의 레시피나 가격을 알고 싶어하면 handle_recipe_request, "
    "그 외 일반적인 요리 대화면 general_chat."
)


@dataclass
class RouteResult:
    intent: Literal["recipe_price", "general_chat"]
    menu_name: str | None = None


def route(user_message: str) -> RouteResult:
    """Classify a user message and, for recipe requests, extract the menu name.

    A single LLM call with two tool choices does both jobs at once (intent +
    extraction), rather than a separate classification step followed by a
    separate extraction step.
    """
    response = chat(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        tools=_TOOLS,
    )
    message = response.choices[0].message
    tool_calls = message.tool_calls or []
    if not tool_calls:
        return RouteResult(intent="general_chat")

    call = tool_calls[0]
    if call.function.name == "handle_recipe_request":
        args = json.loads(call.function.arguments)
        menu_name = args.get("menu_name") or user_message
        return RouteResult(intent="recipe_price", menu_name=menu_name)

    return RouteResult(intent="general_chat")
