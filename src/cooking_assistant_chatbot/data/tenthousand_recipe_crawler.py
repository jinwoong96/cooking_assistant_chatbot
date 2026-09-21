from __future__ import annotations

import json
import re
import time

import httpx

from .models import Recipe

BASE_URL = "https://www.10000recipe.com"
_JSON_LD_RE = re.compile(r'<script type="application/ld\+json">(.*?)</script>', re.S)
_RECIPE_LINK_RE = re.compile(r'href="/recipe/(\d+)"')
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


class TenThousandRecipeCrawler:
    """Small-scale supplemental crawler for 10000recipe.com (만개의레시피).

    Used only to fill gaps in the 식약처 COOKRCP01 dataset, which skews
    toward health/저염식 recipes rather than everyday/자취생 dishes — see
    CLAUDE.md's crawling notes for the legal review this follows: personal,
    non-commercial, small-scale only, never redistributed, robots.txt/ToS
    checked first. robots.txt here only blocks /admin/, /app/, /static/ and
    doesn't specify a crawl-delay, but this still self-throttles (like
    `pricing.enuri_client.EnuriClient`) to stay a good citizen rather than
    hitting the site as fast as possible just because nothing stops it.

    Reads the page's `<script type="application/ld+json">` schema.org
    Recipe block (same technique as the enuri price scraper) rather than the
    rendered HTML — stable, and the ingredients here are already a clean
    array (e.g. "떡 2컵") instead of COOKRCP01's messy free-text blob.
    """

    def __init__(self, http_client: httpx.Client | None = None, min_request_interval: float = 1.0):
        self._http = http_client or httpx.Client(
            timeout=10.0, headers={"User-Agent": _DEFAULT_USER_AGENT}, follow_redirects=True
        )
        self._min_request_interval = min_request_interval
        self._last_request_at: float | None = None

    def _throttle(self) -> None:
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            remaining = self._min_request_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at = time.monotonic()

    def search_recipe_ids(self, keyword: str, limit: int = 10) -> list[str]:
        """Return up to `limit` recipe ids for `keyword`, sorted by 만개의레시피's
        own "추천순"(recommended) order — a reasonable quality signal for a
        supplemental dataset without needing our own ranking."""
        self._throttle()
        response = self._http.get(
            f"{BASE_URL}/recipe/list.html", params={"q": keyword, "order": "reco"}
        )
        response.raise_for_status()
        seen: dict[str, None] = {}
        for recipe_id in _RECIPE_LINK_RE.findall(response.text):
            seen.setdefault(recipe_id, None)
        return list(seen)[:limit]

    def fetch_recipe(self, recipe_id: str) -> Recipe | None:
        """Fetch and parse one recipe. Returns None if the page has no
        parseable schema.org Recipe block (e.g. removed/private post)."""
        self._throttle()
        response = self._http.get(f"{BASE_URL}/recipe/{recipe_id}")
        response.raise_for_status()

        match = _JSON_LD_RE.search(response.text)
        if not match:
            return None
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        if data.get("@type") != "Recipe":
            return None

        return self._to_recipe(recipe_id, data)

    @staticmethod
    def _to_recipe(recipe_id: str, data: dict) -> Recipe:
        ingredients = [i for i in (data.get("recipeIngredient") or []) if i and i.strip()]
        step_entries = [s for s in (data.get("recipeInstructions") or []) if s.get("text")]
        images = data.get("image")
        main_image = images[0] if isinstance(images, list) and images else (images or "")

        return Recipe(
            rcp_seq=f"10000recipe_{recipe_id}",
            name=(data.get("name") or "").strip(),
            ingredients_raw="\n".join(ingredients),
            steps=[s["text"].strip() for s in step_entries],
            step_image_urls=[s.get("image", "") for s in step_entries],
            main_image_url=main_image,
        )

    def close(self) -> None:
        self._http.close()
