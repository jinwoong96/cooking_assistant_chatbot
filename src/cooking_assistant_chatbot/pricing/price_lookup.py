from __future__ import annotations

import sqlite3

from pydantic import BaseModel

from ..data.models import Recipe
from .cache import get_cached_price, set_cached_price
from .enuri_client import EnuriClient, ShoppingItem
from .ingredient_parser import parse_ingredients


class IngredientPrice(BaseModel):
    ingredient_name: str
    quantity_text: str
    cheapest_item: ShoppingItem | None = None


class RecipePriceEstimate(BaseModel):
    recipe_name: str
    ingredient_prices: list[IngredientPrice]

    @property
    def total_price(self) -> int:
        return sum(
            p.cheapest_item.price for p in self.ingredient_prices if p.cheapest_item
        )

    @property
    def unresolved_ingredients(self) -> list[str]:
        return [p.ingredient_name for p in self.ingredient_prices if p.cheapest_item is None]


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

        ingredient_prices.append(
            IngredientPrice(
                ingredient_name=ingredient.name,
                quantity_text=ingredient.quantity_text,
                cheapest_item=cheapest,
            )
        )

    return RecipePriceEstimate(recipe_name=recipe.name, ingredient_prices=ingredient_prices)
