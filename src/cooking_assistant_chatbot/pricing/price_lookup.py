from __future__ import annotations

from pydantic import BaseModel

from ..data.models import Recipe
from .eleven_st_client import ElevenStClient, ShoppingItem
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


def estimate_recipe_price(recipe: Recipe, client: ElevenStClient) -> RecipePriceEstimate:
    """Parse a recipe's ingredients and look up the cheapest matching product for each.

    This is the plain function that a future tool-calling layer would expose
    to the LLM agent; wiring that up is a separate, later feature.
    """
    parsed = parse_ingredients(recipe.ingredients_raw, recipe_name=recipe.name)

    ingredient_prices = []
    for ingredient in parsed:
        items = client.search_cheapest(ingredient.name, page_size=1)
        cheapest = items[0] if items else None
        ingredient_prices.append(
            IngredientPrice(
                ingredient_name=ingredient.name,
                quantity_text=ingredient.quantity_text,
                cheapest_item=cheapest,
            )
        )

    return RecipePriceEstimate(recipe_name=recipe.name, ingredient_prices=ingredient_prices)
