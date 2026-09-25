from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable

from ..llm.client import chat
from ..pricing.enuri_client import EnuriClient
from ..rag.search import RecipeSearcher
from ..voice.tts import to_speech_text
from .tools import TOOLS, ToolContext, run_tool

__all__ = ["handle_message", "Reply"]

MAX_TOOL_ROUNDS = 4
"""How many rounds of tool calls one user message may trigger before the
model is made to answer with what it has. Enough for "추천해주고 재료비도"
(search -> cost) with slack; a cap because a small local model can loop."""

_TEMPERATURE = 0.3
"""Same setting that fixed qwen3:14b ignoring provided data and free-
associating a generic reply in the old compose step (see CLAUDE.md)."""

_SYSTEM_PROMPT = (
    "너는 자취생을 위한 친근한 요리 챗봇이다. 자연스러운 한국어로 대답해라.\n"
    "사용자가 실제로 물어본 것에만 맞춰 답해라:\n"
    "- 추천·탐색 질문이면 search_recipes (가진 재료 목록이 있으면 search_recipes_by_ingredients)\n"
    "- 만드는 법·재료를 물으면 get_recipe\n"
    "- 칼로리·영양을 물으면 get_nutrition\n"
    "- 가격·비용·재료비를 물으면 estimate_ingredient_cost (느리니 물어봤을 때만)\n"
    "질문이 여러 가지를 함께 물으면 필요한 도구를 차례로 여러 번 불러도 된다. "
    "'그거', '아까 그 메뉴'처럼 이전 대화를 가리키면 이전 대화에 나온 레시피 이름을 그대로 써라. "
    "재료 대체나 조리 팁처럼 일반 상식으로 충분한 질문은 도구 없이 바로 답해도 된다.\n"
    "도구 결과를 받았다면 그 데이터만 근거로 답하고, 결과에 없는 수치(가격, 칼로리, 인분 수)를 "
    "지어내지 마라. 도구 결과에 적힌 주의사항(예: 일부 재료만 계산됨)은 답변에 반영해라. "
    "검색 결과에 레시피가 여러 개 있어도, 답변에서 소개하는 레시피의 재료·조리법·가격은 그 "
    "레시피 자신의 결과에서만 가져와라. 다른 레시피의 특징(예: 다른 레시피 이름에 있는 "
    "'전자레인지')을 섞지 말고, 조리법 특징을 말하려면 get_recipe로 확인한 내용만 말해라. "
    "사용자에게 되묻지 말고 완결된 답을 해라."
)


@dataclass
class Reply:
    text: str
    """Markdown shown in the chat window."""
    speech: str
    """The same reply as plain text for TTS (markdown/emoji stripped)."""


def _assistant_tool_message(message) -> dict:
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in message.tool_calls
        ],
    }


def handle_message(
    user_message: str,
    searcher: RecipeSearcher,
    price_client: EnuriClient,
    conn: sqlite3.Connection,
    history: list[dict] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> Reply:
    """Answer one user message with a tool-calling loop.

    The model decides which tools (see `agent.tools.TOOLS`) the question
    actually needs — recipe detail, nutrition, cost, search — instead of the
    old fixed router + search -> price -> compose pipeline that ran every
    step for every recipe question. `history` is Gradio's OpenAI-style
    message list (final replies only; tool results aren't kept), which is
    enough for follow-ups since replies name the recipe they're about.

    `on_progress` receives short Korean status lines ("재료 가격 조회 중
    (3/18 · 두부)") for the UI to show while a reply is being built — a cost
    lookup can take 100s+.
    """
    report = on_progress or (lambda _status: None)
    ctx = ToolContext(searcher=searcher, price_client=price_client, conn=conn, report=report)

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_message})

    report("요청 이해하는 중")
    for round_index in range(MAX_TOOL_ROUNDS + 1):
        # The last round offers no tools, forcing an answer from what's
        # been gathered so far.
        tools = TOOLS if round_index < MAX_TOOL_ROUNDS else None
        response = chat(messages=messages, tools=tools, temperature=_TEMPERATURE)
        message = response.choices[0].message
        if not message.tool_calls or tools is None:
            text = message.content or "죄송해요, 답변을 만들지 못했어요. 다시 한 번 물어봐주실래요?"
            return Reply(text=text, speech=to_speech_text(text))

        messages.append(_assistant_tool_message(message))
        for call in message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": run_tool(call.function.name, call.function.arguments, ctx),
                }
            )
        report("답변 작성 중")

    raise AssertionError("unreachable: the last round always returns")
