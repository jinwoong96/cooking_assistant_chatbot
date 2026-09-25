from __future__ import annotations

import json
import re
import time

import httpx
from pydantic import BaseModel

BASE_URL = "https://price.enuri.com/search"
_LD_JSON_RE = re.compile(r'<script type="application/ld\+json">(.*?)</script>', re.S)
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


class EnuriScrapeError(RuntimeError):
    """Raised when the expected JSON-LD product list can't be found/parsed."""


class ShoppingItem(BaseModel):
    title: str
    price: int
    brand: str = ""
    link: str = ""


class EnuriClient:
    """Scrapes 에누리 (enuri.com) price-comparison search results.

    Replaces the dead/unreachable Naver Shopping / 쿠팡파트너스 / 11번가 API
    options (all shut down, sales-gated, or business-registration-gated — see
    CLAUDE.md for the full trail). enuri.com has no official API, so this
    scrapes its search results page instead of calling one.

    Rather than parsing the page's React/Next.js markup (CSS-module class
    names that change on every frontend redeploy), this reads the
    `<script type="application/ld+json">` schema.org ItemList block the page
    already includes for SEO. Each item's `offers.lowPrice` is enuri's own
    aggregated lowest price across every listed seller for that product —
    exactly the "cheapest price for this ingredient" we want, and a more
    stable target than the rendered HTML.

    robots.txt for price.enuri.com specifies `Crawl-delay: 1`; this client
    enforces at least that much time between requests itself so callers don't
    have to remember to.
    """

    def __init__(
        self,
        http_client: httpx.Client | None = None,
        min_request_interval: float = 1.0,
    ):
        self._http = http_client or httpx.Client(
            timeout=10.0, headers={"User-Agent": _DEFAULT_USER_AGENT}
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

    def search(self, query: str) -> list[ShoppingItem]:
        """All priced results, in enuri's own relevance order. Which one to
        price an ingredient at is `pricing.selection.choose_item`'s job."""
        self._throttle()
        response = self._http.get(BASE_URL, params={"keyword": query})
        response.raise_for_status()

        match = _LD_JSON_RE.search(response.text)
        if not match:
            return []

        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError as e:
            raise EnuriScrapeError("Could not parse product JSON-LD block") from e

        items = []
        for entry in data.get("itemListElement", []):
            item = entry.get("item", {})
            low_price = (item.get("offers") or {}).get("lowPrice")
            if low_price is None:
                continue
            items.append(
                ShoppingItem(
                    title=item.get("name", ""),
                    price=int(low_price),
                    brand=(item.get("brand") or {}).get("name", ""),
                    link=item.get("url", ""),
                )
            )

        return items

    def close(self) -> None:
        self._http.close()
