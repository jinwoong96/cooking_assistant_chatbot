from __future__ import annotations

import sqlite3

from ..data.models import Recipe
from ..llm.client import chat
from ..pricing.enuri_client import EnuriClient
from ..pricing.price_lookup import RecipePriceEstimate, estimate_recipe_price
from ..rag.search import RecipeSearcher
from .router import RouteResult, route

__all__ = ["handle_message", "RouteResult"]

_GENERAL_CHAT_SYSTEM_PROMPT = "너는 친근한 요리 챗봇이다. 자연스러운 한국어로 대답해라."

_COMPOSE_RECIPE_SYSTEM_PROMPT = (
    "너는 요리 정보를 요약해서 알려주는 챗봇이다. 이번 턴은 대화의 시작이 아니라, "
    "사용자가 이미 요청한 레시피에 대한 정보 응답이다. 반드시 아래 사용자 메시지에 주어진 "
    "데이터만 사용해서 답변하고, 데이터에 없는 내용을 지어내지 마라. 사용자에게 재료를 "
    "준비했는지 되묻거나 대화를 여는 질문을 하지 말고, 바로 메뉴명·예상 총 비용·조리 순서 "
    "요약을 담은 완결된 답변을 작성해라."
)


def _compose_recipe_reply(recipe: Recipe, estimate: RecipePriceEstimate) -> str:
    ingredient_lines = "\n".join(
        f"- {p.ingredient_name}: "
        + (
            f"{p.cheapest_item.price}원 ({p.cheapest_item.title})"
            if p.cheapest_item
            else "가격 정보 없음"
        )
        for p in estimate.ingredient_prices
    )
    steps = "\n".join(recipe.steps)
    prompt = (
        "아래 레시피와 예상 재료비 정보를 참고해서, 사용자에게 친근한 한국어로 답변을 "
        "작성해줘. 메뉴명, 대략적인 총 예상 비용, 간단한 조리 순서 요약을 포함해줘.\n\n"
        f"메뉴: {recipe.name}\n"
        f"재료별 최저가:\n{ingredient_lines}\n"
        f"예상 총 비용: {estimate.total_price}원\n"
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


def _general_chat_reply(user_message: str) -> str:
    response = chat(
        messages=[
            {"role": "system", "content": _GENERAL_CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )
    return response.choices[0].message.content


def handle_message(
    user_message: str,
    searcher: RecipeSearcher,
    price_client: EnuriClient,
    conn: sqlite3.Connection,
) -> str:
    """Route a user message and produce a final reply.

    Fixed pipeline for the recipe/price path (search -> price -> compose),
    per the earlier design decision to keep MVP orchestration deterministic
    rather than a fully autonomous agent loop.
    """
    result = route(user_message)

    if result.intent == "general_chat":
        return _general_chat_reply(user_message)

    recipes = searcher.search(result.menu_name, top_k=1)
    if not recipes:
        return f"'{result.menu_name}' 레시피를 찾지 못했어요. 다른 메뉴로 물어봐주실래요?"

    recipe = recipes[0]
    estimate = estimate_recipe_price(recipe, price_client, conn)
    return _compose_recipe_reply(recipe, estimate)
