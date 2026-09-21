from __future__ import annotations

import sqlite3

from pydantic import BaseModel

from ..data.models import Recipe
from .cache import get_cached_price, set_cached_price
from .enuri_client import EnuriClient, ShoppingItem
from .ingredient_parser import parse_ingredients
from .unit_parser import parse_quantity


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


def estimate_recipe_price(
    recipe: Recipe, client: EnuriClient, conn: sqlite3.Connection
) -> RecipePriceEstimate:
    """Parse a recipe's ingredients and look up the cheapest matching product for each.

    Prices are cached in SQLite (see `pricing.cache`) so repeated lookups of
    the same ingredient don't re-scrape enuri.com — both for latency and to
    keep request volume low (see CLAUDE.md's crawling notes).

    This is the plain function that a future tool-calling layer would expose
    to the LLM agent; wiring that up is a separate, later feature.
    """
    parsed = parse_ingredients(recipe.ingredients_raw, recipe_name=recipe.name)

    ingredient_prices = []
    for ingredient in parsed:
        cheapest = get_cached_price(conn, ingredient.name)
        if cheapest is None:
            items = client.search_cheapest(ingredient.name, limit=1)
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
