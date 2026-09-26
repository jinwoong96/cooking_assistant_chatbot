"""Tools the chat agent can call, one per thing a user might want to know.

Split out of the old fixed recipe pipeline (search -> price every
ingredient -> compose) so a question only pays for what it asks: "칼로리
알려줘" no longer waits ~100s on price lookups it doesn't need.

Every tool that works on one recipe takes a `recipe_name` and resolves it
itself (exact name first, then semantic search), rather than passing opaque
ids around — the model can then reuse a name from an earlier turn for
follow-ups ("그거 재료비는?") without any extra session state.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable

from ..data.db import get_recipe_by_name
from ..data.ingredient_index import find_recipes_by_ingredients
from ..data.models import Recipe
from ..pricing.enuri_client import EnuriClient
from ..pricing.price_lookup import RecipePriceEstimate, estimate_recipe_price
from ..pricing.selection import (
    DEFAULT_PRICE_BASIS,
    PRICE_BASES,
    PRICE_BASIS_LABELS,
    PriceBasis,
)
from ..rag.search import RecipeSearcher
from .ingredient_correction import correct_ingredient_name


@dataclass
class ToolContext:
    searcher: RecipeSearcher
    price_client: EnuriClient
    conn: sqlite3.Connection
    report: Callable[[str], None]
    price_basis: PriceBasis = DEFAULT_PRICE_BASIS
    """The UI's default; a request can override it per call."""
    shown_recipe_seq: str | None = None
    """The last recipe whose ingredients this turn showed the user (via
    get_recipe or estimate_ingredient_cost) — what cooking mode starts from."""


def _recipe_name_param(description: str) -> dict:
    return {
        "type": "object",
        "properties": {
            "recipe_name": {
                "type": "string",
                "description": description,
            }
        },
        "required": ["recipe_name"],
    }


# No concrete dish as an example here: with "(예: '김치찌개')" the model
# filled in 김치찌개 for "그거 칼로리는?" in a brand-new chat.
_RECIPE_NAME_DESC = (
    "사용자가 말한 메뉴/레시피 이름 그대로. 이전 대화에서 나온 레시피면 그 이름을 그대로 쓴다. "
    "사용자가 메뉴를 말하지 않았고 이전 대화에도 없으면 이 도구를 부르지 않는다."
)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_recipes",
            "description": (
                "메뉴 이름이나 맛·분위기·종류로 레시피 후보 목록을 의미 기반 검색한다. "
                "'매콤한 국물요리 추천해줘', '간단한 자취 요리 뭐 있어?', '이거랑 비슷한 메뉴' 같은 "
                "추천/탐색 질문에 사용. 빠르다."
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
    {
        "type": "function",
        "function": {
            "name": "search_recipes_by_ingredients",
            "description": (
                "사용자가 가진 재료를 '전부' 포함하는 레시피를 정확히 검색한다. "
                "'냉장고에 두부랑 계란 있는데 뭐 해먹지' 같은 질문에 사용. 빠르다."
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
            "name": "get_recipe",
            "description": (
                "특정 메뉴의 레시피 상세(재료와 분량, 인분 수, 조리 순서)를 가져온다. "
                "'김치찌개 레시피 알려줘', '어떻게 만들어?' 같은 질문에 사용. 빠르다."
            ),
            "parameters": _recipe_name_param(_RECIPE_NAME_DESC),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_nutrition",
            "description": (
                "특정 메뉴의 영양성분(열량, 탄수화물, 단백질, 지방, 나트륨)을 가져온다. "
                "'칼로리 얼마야?', '다이어트에 괜찮아?' 같은 질문에 사용. 빠르다."
            ),
            "parameters": _recipe_name_param(_RECIPE_NAME_DESC),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_ingredient_cost",
            "description": (
                "특정 메뉴의 재료를 인터넷 최저가로 조회해 재료비를 계산한다 (재료를 전부 새로 살 때 "
                "금액과, 레시피에 쓰는 양만큼의 원가). 재료 하나당 1초 이상 걸려 느리므로, 사용자가 "
                "가격·비용·재료비를 물었을 때만 사용한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_name": {"type": "string", "description": _RECIPE_NAME_DESC},
                    "price_basis": {
                        "type": "string",
                        "enum": list(PRICE_BASES),
                        "description": (
                            "상품 고르는 기준. 사용자가 기준을 말했을 때만 넣는다: "
                            "'소량만', '작은 거', '돈 적게', '1인분만 살래' → min_spend, "
                            "'가성비', '단위가격', '대용량이 싸면' → unit_price, "
                            "'제일 잘 맞는 상품' → relevance. 말하지 않았으면 생략."
                        ),
                    },
                },
                "required": ["recipe_name"],
            },
        },
    },
]


RECIPE_NAME_TOOLS = frozenset({"get_recipe", "get_nutrition", "estimate_ingredient_cost"})
"""Tools that take a `recipe_name` — the pipeline checks the name was
actually mentioned before running them (`pipeline.is_grounded`)."""


def _resolve_recipe(name: str, ctx: ToolContext) -> Recipe | None:
    exact = get_recipe_by_name(ctx.conn, name)
    if exact is not None:
        return exact
    found = ctx.searcher.search(name, top_k=1)
    return found[0] if found else None


def _found_header(requested: str, recipe: Recipe) -> str:
    if "".join(requested.split()) == "".join(recipe.name.split()):
        return f"레시피: {recipe.name}"
    # Semantic fallback can land on a near match; say so, so the model
    # doesn't present e.g. "돼지고기김치찌개" as exactly what was asked for.
    return f"'{requested}'와 정확히 같은 이름은 없어서 가장 비슷한 레시피를 찾음: {recipe.name}"


def _not_found(requested: str) -> str:
    return f"'{requested}' 레시피를 DB에서 찾지 못함."


def _format_recipe_list(recipes: list[Recipe]) -> str:
    if not recipes:
        return "검색 결과 없음."
    lines = []
    for r in recipes:
        ingredients_preview = ", ".join(
            line.strip() for line in r.ingredients_raw.split("\n")[:6] if line.strip()
        )
        lines.append(f"- {r.name} (재료: {ingredients_preview})")
    return "\n".join(lines)


def format_recipe_detail(requested: str, recipe: Recipe) -> str:
    lines = [_found_header(requested, recipe)]
    if recipe.servings is not None:
        lines.append(f"인분 수: {recipe.servings}인분")
    else:
        lines.append("인분 수: 정보 없음 (추측해서 말하지 말 것)")
    lines.append(f"재료:\n{recipe.ingredients_raw or '정보 없음'}")
    if recipe.steps:
        lines.append("조리 순서:\n" + "\n".join(recipe.steps))
    if recipe.na_tip:
        lines.append(f"저감 조리 팁: {recipe.na_tip}")
    return "\n".join(lines)


_NUTRITION_FIELDS = [
    ("열량", "energy_kcal", "kcal"),
    ("탄수화물", "carbohydrate_g", "g"),
    ("단백질", "protein_g", "g"),
    ("지방", "fat_g", "g"),
    ("나트륨", "sodium_mg", "mg"),
]


def format_nutrition(requested: str, recipe: Recipe) -> str:
    header = _found_header(requested, recipe)
    values = [
        (label, getattr(recipe, field).strip(), unit) for label, field, unit in _NUTRITION_FIELDS
    ]
    if not any(value for _, value, _ in values):
        # Only the 식약처 COOKRCP01 recipes carry nutrition data; the
        # 만개의레시피 supplement doesn't.
        return f"{header}\n이 레시피는 영양성분 정보가 없음 (추정치를 지어내지 말 것)."
    lines = [header, "영양성분 (식약처 레시피 DB 제공 값):"]
    lines += [f"- {label}: {value}{unit}" for label, value, unit in values if value]
    if recipe.weight_info.strip():
        lines.append(f"- 1인분 중량: {recipe.weight_info.strip()}g")
    return "\n".join(lines)


_BASIS_EXPLANATIONS: dict[PriceBasis, str] = {
    "relevance": "검색 상위 상품 중 최저가 (대용량 포장이 많음)",
    "min_spend": "레시피에 필요한 양 이상인 상품 중 지금 내는 돈이 가장 적은 것",
    "unit_price": "g/ml당 가격이 가장 싼 상품 (가성비)",
}


def format_cost(
    requested: str,
    recipe: Recipe,
    estimate: RecipePriceEstimate,
    basis: PriceBasis = DEFAULT_PRICE_BASIS,
) -> str:
    lines = [_found_header(requested, recipe)]
    lines.append(f"가격 기준: {PRICE_BASIS_LABELS[basis]} — {_BASIS_EXPLANATIONS[basis]}")
    if recipe.servings is not None:
        lines.append(f"인분 수: {recipe.servings}인분")
    lines.append("재료별 최저가:")
    for p in estimate.ingredient_prices:
        if not p.cheapest_item:
            lines.append(f"- {p.ingredient_name}: 가격 정보 없음")
            continue
        line = f"- {p.ingredient_name}: {p.cheapest_item.price:,}원 ({p.cheapest_item.title})"
        if p.portioned_cost is not None:
            line += f" / 레시피 사용량({p.quantity_text})만큼은 약 {p.portioned_cost:,}원"
        lines.append(line)

    priced = [p for p in estimate.ingredient_prices if p.cheapest_item]
    portioned = [p for p in priced if p.portioned_cost is not None]
    lines.append(f"재료를 전부 새로 살 때 총액: {estimate.total_price:,}원 ({len(priced)}개 재료 기준)")
    # A portioned total covering 3 of 18 ingredients reads as "this dish costs
    # 697원" unless the coverage is stated right next to it.
    lines.append(
        f"레시피 사용량 기준 원가 합계: {estimate.total_portioned_cost:,}원 "
        f"(전체 {len(estimate.ingredient_prices)}개 재료 중 {len(portioned)}개만 계산됨 — "
        "나머지는 상품 용량 표기가 없거나 단위가 달라 계산 불가. 이 합계를 요리 전체 원가처럼 말하지 말 것)"
    )
    if estimate.unresolved_ingredients:
        lines.append(f"가격을 못 찾은 재료: {', '.join(estimate.unresolved_ingredients)}")
    return "\n".join(lines)


def run_tool(name: str, arguments: str, ctx: ToolContext) -> str:
    """Execute one tool call and return its result as text for the LLM."""
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        args = {}

    if name == "search_recipes":
        query = args.get("query", "")
        ctx.report(f"'{query}' 레시피 검색 중")
        return _format_recipe_list(ctx.searcher.search(query, top_k=5))

    if name == "search_recipes_by_ingredients":
        ingredients = args.get("ingredients") or []
        ctx.report(f"{', '.join(ingredients)} 들어간 레시피 찾는 중")
        return _format_recipe_list(find_recipes_by_ingredients(ctx.conn, ingredients, limit=5))

    if name in RECIPE_NAME_TOOLS:
        requested = args.get("recipe_name", "")
        ctx.report(f"'{requested}' 레시피 찾는 중")
        recipe = _resolve_recipe(requested, ctx)
        if recipe is None:
            return _not_found(requested)
        if name in ("get_recipe", "estimate_ingredient_cost"):
            ctx.shown_recipe_seq = recipe.rcp_seq
        if name == "get_recipe":
            return format_recipe_detail(requested, recipe)
        if name == "get_nutrition":
            return format_nutrition(requested, recipe)
        basis = args.get("price_basis")
        if basis not in PRICE_BASES:
            basis = ctx.price_basis
        estimate = estimate_recipe_price(
            recipe,
            ctx.price_client,
            ctx.conn,
            correct_name=correct_ingredient_name,
            on_progress=lambda done, total, ingredient: ctx.report(
                f"재료 가격 조회 중 ({done}/{total} · {ingredient})"
            ),
            basis=basis,
        )
        return format_cost(requested, recipe, estimate, basis)

    return f"알 수 없는 도구: {name}"
