from __future__ import annotations

import sqlite3

from ..data.models import Recipe
from ..llm.client import chat
from ..pricing.enuri_client import EnuriClient
from ..pricing.price_lookup import RecipePriceEstimate, estimate_recipe_price
from ..rag.search import RecipeSearcher
from .ingredient_correction import correct_ingredient_name
from .router import RouteResult, route

__all__ = ["handle_message", "RouteResult"]

_GENERAL_CHAT_SYSTEM_PROMPT = "너는 친근한 요리 챗봇이다. 자연스러운 한국어로 대답해라."

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
    estimate = estimate_recipe_price(
        recipe, price_client, conn, correct_name=correct_ingredient_name
    )
    return _compose_recipe_reply(recipe, estimate)
