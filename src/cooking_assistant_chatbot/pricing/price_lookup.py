from __future__ import annotations

import sqlite3
from typing import Callable

from pydantic import BaseModel

from ..data.models import Recipe
from .cache import get_cached_price, set_cached_price
from .enuri_client import EnuriClient, ShoppingItem
from .ingredient_parser import parse_ingredients
from .unit_parser import parse_quantity

CorrectName = Callable[[str, str], "str | None"]

_AMBIGUOUS_WORD_COUNT = 3
"""A parsed ingredient name with this many words or more is treated as
ambiguous (likely still has leftover descriptive text like "얇게 썬", "고기
삶는 재료") and is sent through `correct_name` before searching, rather than
only as a fallback after a failed search — plain search failures aren't the
only symptom; a bad name can also return a confidently wrong product."""


class IngredientPrice(BaseModel):
    ingredient_name: str
    quantity_text: str
    cheapest_item: ShoppingItem | None = None
    portioned_cost: int | None = None
    """Estimated cost of just the amount the recipe needs (not the whole
    purchased package). None when either the recipe amount or the product's
    package size couldn't be parsed as a weight/volume, or they're different
    kinds of quantity (e.g. recipe needs grams but the product is sold by
    count) — see `unit_parser.parse_quantity`."""


class RecipePriceEstimate(BaseModel):
    recipe_name: str
    ingredient_prices: list[IngredientPrice]

    @property
    def total_price(self) -> int:
        """Total cost of buying a full purchasable package of every ingredient."""
        return sum(
            p.cheapest_item.price for p in self.ingredient_prices if p.cheapest_item
        )

    @property
    def total_portioned_cost(self) -> int:
        """Total cost of just the amounts the recipe actually needs, for
        ingredients where that could be computed."""
        return sum(
            p.portioned_cost for p in self.ingredient_prices if p.portioned_cost is not None
        )

    @property
    def unresolved_ingredients(self) -> list[str]:
        return [p.ingredient_name for p in self.ingredient_prices if p.cheapest_item is None]

    @property
    def ingredients_missing_portioned_cost(self) -> list[str]:
        """Ingredients with a price but no computable portioned cost."""
        return [
            p.ingredient_name
            for p in self.ingredient_prices
            if p.cheapest_item is not None and p.portioned_cost is None
        ]


def _compute_portioned_cost(recipe_quantity_text: str, item: ShoppingItem) -> int | None:
    recipe_qty = parse_quantity(recipe_quantity_text)
    package_qty = parse_quantity(item.title)
    if recipe_qty is None or package_qty is None:
        return None
    if recipe_qty.unit != package_qty.unit or package_qty.value == 0:
        return None
    return round(item.price * (recipe_qty.value / package_qty.value))


def _looks_ambiguous(name: str) -> bool:
    return len(name.split()) >= _AMBIGUOUS_WORD_COUNT


def _search_with_correction(
    client: EnuriClient,
    name: str,
    ingredients_context: str,
    correct_name: CorrectName | None,
) -> list[ShoppingItem]:
    """Search for `name`, optionally cleaning it up via `correct_name` first
    (when it looks ambiguous) or as a fallback (when the plain search finds
    nothing). Returns [] if `correct_name` decides this isn't a real
    purchasable ingredient at all."""
    already_corrected = False
    search_name = name

    if correct_name is not None and _looks_ambiguous(name):
        corrected = correct_name(name, ingredients_context)
        if corrected is None:
            return []
        search_name = corrected
        already_corrected = True

    items = client.search_cheapest(search_name, limit=1)

    if not items and not already_corrected and correct_name is not None:
        corrected = correct_name(name, ingredients_context)
        if corrected is None:
            return []
        items = client.search_cheapest(corrected, limit=1)

    return items


def estimate_recipe_price(
    recipe: Recipe,
    client: EnuriClient,
    conn: sqlite3.Connection,
    correct_name: CorrectName | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> RecipePriceEstimate:
    """Parse a recipe's ingredients and look up the cheapest matching product for each.

    Prices are cached in SQLite (see `pricing.cache`) so repeated lookups of
    the same ingredient don't re-scrape enuri.com — both for latency and to
    keep request volume low (see CLAUDE.md's crawling notes).

    `correct_name`, if given, is an LLM-backed callback (see
    `agent.ingredient_correction.correct_ingredient_name`) used to clean up
    ambiguous ingredient names before/after searching. Kept as an injected
    callback rather than importing the LLM client directly so this module
    stays pure/LLM-free and easy to test.

    `on_progress(done_count, total, ingredient_name)` is called before each
    ingredient is looked up (1-based), so the UI can show progress during
    what's usually the slowest part of a reply (enuri's 1 req/sec throttle).

    This is the plain function that a future tool-calling layer would expose
    to the LLM agent; wiring that up is a separate, later feature.
    """
    parsed = parse_ingredients(recipe.ingredients_raw, recipe_name=recipe.name)

    ingredient_prices = []
    for index, ingredient in enumerate(parsed, start=1):
        if on_progress is not None:
            on_progress(index, len(parsed), ingredient.name)
        cheapest = get_cached_price(conn, ingredient.name)
        if cheapest is None:
            items = _search_with_correction(
                client, ingredient.name, recipe.ingredients_raw, correct_name
            )
            cheapest = items[0] if items else None
            if cheapest is not None:
                set_cached_price(conn, ingredient.name, cheapest)

        portioned_cost = (
            _compute_portioned_cost(ingredient.quantity_text, cheapest) if cheapest else None
        )

        ingredient_prices.append(
            IngredientPrice(
                ingredient_name=ingredient.name,
                quantity_text=ingredient.quantity_text,
                cheapest_item=cheapest,
                portioned_cost=portioned_cost,
            )
        )

    return RecipePriceEstimate(recipe_name=recipe.name, ingredient_prices=ingredient_prices)
