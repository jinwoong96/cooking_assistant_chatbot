from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

# Negative lookahead only excludes a following LATIN letter (blocks partial
# matches like the "g" in "50mg" or "gram", or "l" in a longer English word).
# A following Korean character is fine and common ("500g짜리", "1kg에") since
# Korean particles attach directly to numbers/units with no space.
_WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(kg|g)(?![a-zA-Z])", re.IGNORECASE)
_VOLUME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ml|l)(?![a-zA-Z])", re.IGNORECASE)


class Quantity(BaseModel):
    value: float
    unit: Literal["g", "ml"]


def parse_quantity(text: str) -> Quantity | None:
    """Best-effort extraction of a weight or volume amount from free text.

    Returns the amount normalized to grams (for kg/g) or milliliters (for
    L/ml) — the first such match in the text. Returns None when no clear
    weight/volume amount is found, e.g. the text uses a count ("1개", "5구")
    or a vague amount ("약간", "적당량"). Count-based and vague quantities
    are intentionally not supported: converting "1개" of an ingredient into
    grams, or comparing it against a differently-counted product, isn't
    something we can do reliably without more data than we have.
    """
    match = _WEIGHT_RE.search(text)
    if match:
        value = float(match.group(1))
        if match.group(2).lower() == "kg":
            value *= 1000
        return Quantity(value=value, unit="g")

    match = _VOLUME_RE.search(text)
    if match:
        value = float(match.group(1))
        if match.group(2).lower() == "l":
            value *= 1000
        return Quantity(value=value, unit="ml")

    return None


def parse_package_quantity(title: str) -> Quantity | None:
    """A product title's package size: the *largest* weight (else volume)
    in it, not the first. Titles often mention a smaller number before the
    package size — "수미감자 소 (조림용 40g 미만) 10kg" is a 10kg sack,
    "올리브오일 140g (10g x 14포)" is 140g."""
    weights = [
        float(m.group(1)) * (1000 if m.group(2).lower() == "kg" else 1)
        for m in _WEIGHT_RE.finditer(title)
    ]
    if weights:
        return Quantity(value=max(weights), unit="g")
    volumes = [
        float(m.group(1)) * (1000 if m.group(2).lower() == "l" else 1)
        for m in _VOLUME_RE.finditer(title)
    ]
    if volumes:
        return Quantity(value=max(volumes), unit="ml")
    return None
