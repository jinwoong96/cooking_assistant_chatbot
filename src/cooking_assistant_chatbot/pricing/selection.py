"""Which search result to price an ingredient at, per a user-selectable basis.

enuri returns ~40 results in its own relevance order. How to pick one:
- "relevance" (default): cheapest of the top few relevance results. The top
  results are the right product, but produce there is mostly 5-10kg sacks.
- "min_spend": the least money out of pocket for a package that still
  covers what the recipe needs — for someone cooking for one.
- "unit_price": the lowest price per g/ml, i.e. best value.

"min_spend" replaced a first try at "smallest package": enuri's price for a
small item is often a multi-pack's ("스팸 120g" at 19,350원 while "스팸
300g" was 3,430원), so smallest-first came out *more* expensive than
relevance on a real recipe (오므라이스 105,446원 vs 92,340원). Cheapest-
that-covers-the-need skips those bundle prices on its own.

"min_spend" and "unit_price" look past the top few, so they only consider
results that are the ingredient itself (`_is_the_ingredient`) — deeper in
the list, unrelated listings are common (a search for "피망" also returns
toy cash registers, "감자" returns 감자칩).
"""

from __future__ import annotations

import re
from typing import Literal

from .enuri_client import ShoppingItem
from .unit_parser import parse_package_quantity, parse_quantity

PriceBasis = Literal["relevance", "min_spend", "unit_price"]
PRICE_BASES: tuple[PriceBasis, ...] = ("relevance", "min_spend", "unit_price")
DEFAULT_PRICE_BASIS: PriceBasis = "relevance"

PRICE_BASIS_LABELS: dict[PriceBasis, str] = {
    "relevance": "관련도 우선",
    "min_spend": "최소 지출",
    "unit_price": "단위가격 우선",
}

RELEVANCE_POOL = 5
"""Only this many top relevance results are candidates under "relevance".
Taking the cheapest of all ~40 used to pick unrelated cheap listings from
far down the page: 29 of 76 cached matches didn't even name the ingredient
("밥" -> a latte powder, "고춧가루" -> apple vinegar, "청고추" -> packing
string), while enuri's top results were right (햇반, real 고춧가루)."""


def _names(item: ShoppingItem, query: str) -> bool:
    key = "".join(query.split())
    return bool(key) and key in "".join(item.title.split())


_TOKEN_SPLIT_RE = re.compile(r"[\s/()\[\],+·&]+")
_TRAILING_SIZE_RE = re.compile(r"[\d.]+\s*(kg|g|ml|l|개|구|입|봉|팩)?$", re.IGNORECASE)


def _is_the_ingredient(item: ShoppingItem, query: str) -> bool:
    """Stricter than `_names`: some title word must *end* with the query.
    Korean compounds put the head noun last, so this keeps 수미감자, 진간장,
    흑미밥 but drops 감자칩, 감자전분, 대파분태 — products made from the
    ingredient rather than the ingredient itself."""
    key = "".join(query.split())
    if not key:
        return False
    if " " in query.strip():
        return _names(item, query)
    for token in _TOKEN_SPLIT_RE.split(item.title):
        token = _TRAILING_SIZE_RE.sub("", token)
        if token.endswith(key):
            return True
    return False


def _cheapest(items: list[ShoppingItem]) -> ShoppingItem | None:
    return min(items, key=lambda i: i.price, default=None)


def _pick_relevance(items: list[ShoppingItem], query: str) -> ShoppingItem | None:
    """Top `RELEVANCE_POOL`, preferring titles that name the query (spaces
    ignored), cheapest first. Falls back to the whole pool when none name
    it, since spelling variants are common (달걀 -> "계란 30구", 케첩 ->
    "케찹")."""
    top = items[:RELEVANCE_POOL]
    return _cheapest([i for i in top if _names(i, query)] or top)


def _sized(items: list[ShoppingItem]) -> list[tuple[ShoppingItem, float, str]]:
    sized = []
    for item in items:
        qty = parse_package_quantity(item.title)
        if qty is not None and qty.value > 0:
            sized.append((item, qty.value, qty.unit))
    return sized


def _pick_min_spend(
    candidates: list[ShoppingItem], needed_text: str
) -> ShoppingItem | None:
    needed = parse_quantity(needed_text)
    if needed is not None:
        covering = [
            s for s in _sized(candidates) if s[2] == needed.unit and s[1] >= needed.value
        ]
        # Only packages that cover the need — not a 6ml sample sachet of
        # 간장 for a recipe that needs 30ml. Ties go to the smaller package.
        if covering:
            return min(covering, key=lambda s: (s[0].price, s[1]))[0]
    return _cheapest(candidates)


def _pick_unit_price(candidates: list[ShoppingItem]) -> ShoppingItem | None:
    sized = _sized(candidates)
    if not sized:
        return None
    return min(sized, key=lambda s: s[0].price / s[1])[0]


def choose_item(
    items: list[ShoppingItem],
    query: str,
    needed_text: str = "",
    basis: PriceBasis = DEFAULT_PRICE_BASIS,
) -> ShoppingItem | None:
    """Pick the result to price `query` at. `items` must be in enuri's
    relevance order; `needed_text` is the recipe's amount ("100g"), used by
    "min_spend". Falls back to the "relevance" pick whenever the chosen basis
    has nothing to work with (no title names the ingredient, or none
    states a package size — e.g. eggs are sold by count, "30구")."""
    if basis != "relevance":
        candidates = [i for i in items if _is_the_ingredient(i, query)]
        picked = (
            _pick_min_spend(candidates, needed_text)
            if basis == "min_spend"
            else _pick_unit_price(candidates)
        )
        if picked is not None:
            return picked
    return _pick_relevance(items, query)
