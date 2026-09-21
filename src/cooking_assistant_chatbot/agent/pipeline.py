from __future__ import annotations

import json
import sqlite3

from ..data.ingredient_index import find_recipes_by_ingredients
from ..data.models import Recipe
from ..llm.client import chat
from ..pricing.enuri_client import EnuriClient
from ..pricing.price_lookup import RecipePriceEstimate, estimate_recipe_price
from ..rag.search import RecipeSearcher
from .ingredient_correction import correct_ingredient_name
from .router import RouteResult, route

__all__ = ["handle_message", "RouteResult"]

_GENERAL_CHAT_SYSTEM_PROMPT = (
    "너는 친근한 요리 챗봇이다. 자연스러운 한국어로 대답해라. 두 가지 검색 도구가 있다: "
    "search_recipes_by_ingredients는 사용자가 가진 재료를 '전부' 포함하는 레시피를 정확히 "
    "찾아준다 (예: '냉장고에 두부랑 계란 있는데 뭐 해먹지'처럼 구체적인 재료 목록이 있을 때). "
    "search_recipes_by_style은 재료가 아니라 분위기·맛·메뉴 종류로 의미 기반 검색한다 "
    "(예: '매콤한 국물요리 추천해줘', '이거랑 비슷한 메뉴 있어?'). 상황에 맞는 도구를 골라 "
    "쓰고, 재료 대체처럼 일반 상식으로 충분한 질문은 도구 없이 바로 답해도 된다. 이전 대화 "
    "맥락을 참고해서 자연스럽게 이어서 대답해라."
)

_GENERAL_CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_recipes_by_ingredients",
            "description": (
                "사용자가 가진 재료를 전부 포함하는 레시피를 정확히 검색한다 "
                "(재료명 기반 정확 매칭). '냉장고에 있는 재료로 뭐 해먹지' 같은 질문에 사용."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ingredients": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "가진 재료 목록 (예: ['두부', '계란'])",
                    }
                },
                "required": ["ingredients"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_recipes_by_style",
            "description": (
                "재료가 아니라 맛·분위기·메뉴 종류 등으로 레시피를 의미 기반 검색한다. "
                "'비슷한 메뉴 추천해줘', '매콤한 국물요리 뭐 있어' 같은 질문에 사용."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색어 (예: '매콤한 국물요리', '든든한 자취요리')",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

_COMPOSE_RECIPE_SYSTEM_PROMPT = (
    "너는 요리 정보를 요약해서 알려주는 챗봇이다. 이번 턴은 대화의 시작이 아니라, "
    "사용자가 이미 요청한 레시피에 대한 정보 응답이다. 반드시 아래 사용자 메시지에 주어진 "
    "데이터만 사용해서 답변하고, 데이터에 없는 내용을 지어내지 마라. 몇 인분인지가 주어지지 "
    "않았다면 인분 수를 절대 추측하거나 언급하지 마라. 사용자에게 재료를 준비했는지 되묻거나 "
    "대화를 여는 질문을 하지 말고, 바로 메뉴명·(주어졌다면) 인분 수·재료 전체 구매 시 총 "
    "비용·레시피 사용량 기준 예상 원가·조리 순서 요약을 담은 완결된 답변을 작성해라."
)


def _compose_recipe_reply(recipe: Recipe, estimate: RecipePriceEstimate) -> str:
    def _price_line(p) -> str:
        if not p.cheapest_item:
            return "가격 정보 없음"
        line = f"{p.cheapest_item.price}원 구매 ({p.cheapest_item.title})"
        if p.portioned_cost is not None:
            line += f" / 이 레시피에 쓰는 양({p.quantity_text or '수량 미상'})만큼은 약 {p.portioned_cost}원"
        return line

    ingredient_lines = "\n".join(
        f"- {p.ingredient_name}: {_price_line(p)}" for p in estimate.ingredient_prices
    )
    steps = "\n".join(recipe.steps)
    missing_note = (
        f" (다음 재료는 상품 용량 표기가 없거나 단위가 달라 소분원가 계산 불가: "
        f"{', '.join(estimate.ingredients_missing_portioned_cost)})"
        if estimate.ingredients_missing_portioned_cost
        else ""
    )
    servings_line = (
        f"인분 수: {recipe.servings}인분\n" if recipe.servings is not None else ""
    )
    prompt = (
        "아래 레시피와 예상 재료비 정보를 참고해서, 사용자에게 친근한 한국어로 답변을 "
        "작성해줘. 메뉴명, (주어졌다면) 인분 수, 재료를 전부 새로 구매할 때의 총 비용, "
        "레시피에 필요한 양만큼만 썼을 때의 예상 원가(소분원가) 합계, 간단한 조리 순서 "
        "요약을 포함해줘.\n\n"
        f"메뉴: {recipe.name}\n"
        f"{servings_line}"
        f"재료별 가격:\n{ingredient_lines}\n"
        f"전체 재료 새로 구매 시 총 비용: {estimate.total_price}원\n"
        f"레시피 사용량 기준 예상 원가 합계: {estimate.total_portioned_cost}원{missing_note}\n"
        f"조리 순서:\n{steps}"
    )
    response = chat(
        messages=[
            {"role": "system", "content": _COMPOSE_RECIPE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


def _format_recipes_for_llm(recipes: list[Recipe]) -> str:
    if not recipes:
        return "검색 결과 없음."
    lines = []
    for r in recipes:
        ingredients_preview = ", ".join(
            line.strip() for line in r.ingredients_raw.split("\n")[:6] if line.strip()
        )
        lines.append(f"- {r.name} (재료: {ingredients_preview})")
    return "\n".join(lines)


def _run_general_chat_tool(
    call, searcher: RecipeSearcher, conn: sqlite3.Connection
) -> list[Recipe]:
    args = json.loads(call.function.arguments)
    if call.function.name == "search_recipes_by_ingredients":
        return find_recipes_by_ingredients(conn, args.get("ingredients") or [], limit=3)
    return searcher.search(args.get("query", ""), top_k=3)


def _general_chat_reply(
    user_message: str,
    history: list[dict],
    searcher: RecipeSearcher,
    conn: sqlite3.Connection,
) -> str:
    """LLM passthrough for anything that isn't a specific recipe/price
    request, optionally grounded in the recipe DB via search tools (exact
    ingredient match, or semantic style/similarity search — see
    _GENERAL_CHAT_TOOLS). `history` is Gradio's OpenAI-style message list,
    passed straight through as prior turns so follow-up questions ("그거
    말고 다른 건?") have context.
    """
    messages = [{"role": "system", "content": _GENERAL_CHAT_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = chat(messages=messages, tools=_GENERAL_CHAT_TOOLS)
    message = response.choices[0].message
    tool_calls = message.tool_calls or []
    if not tool_calls:
        return message.content

    call = tool_calls[0]
    results = _run_general_chat_tool(call, searcher, conn)

    messages.append(
        {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": _format_recipes_for_llm(results),
        }
    )

    follow_up = chat(messages=messages)
    return follow_up.choices[0].message.content


def handle_message(
    user_message: str,
    searcher: RecipeSearcher,
    price_client: EnuriClient,
    conn: sqlite3.Connection,
    history: list[dict] | None = None,
) -> str:
    """Route a user message and produce a final reply.

    Fixed pipeline for the recipe/price path (search -> price -> compose),
    per the earlier design decision to keep MVP orchestration deterministic
    rather than a fully autonomous agent loop. `history` (Gradio's
    OpenAI-style message list) is only threaded into the general-chat path;
    the recipe/price path stays single-shot since each request names its own
    dish.
    """
    result = route(user_message)

    if result.intent == "general_chat":
        return _general_chat_reply(user_message, history or [], searcher, conn)

    recipes = searcher.search(result.menu_name, top_k=1)
    if not recipes:
        return f"'{result.menu_name}' 레시피를 찾지 못했어요. 다른 메뉴로 물어봐주실래요?"

    recipe = recipes[0]
    estimate = estimate_recipe_price(
        recipe, price_client, conn, correct_name=correct_ingredient_name
    )
    return _compose_recipe_reply(recipe, estimate)
