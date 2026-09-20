from __future__ import annotations

import httpx

from .models import Recipe

BASE_URL = "http://openapi.foodsafetykorea.go.kr/api"
SERVICE_ID = "COOKRCP01"


class FoodSafetyApiError(RuntimeError):
    """Raised when the 식약처 Open API returns an error payload."""


class FoodSafetyClient:
    """Client for the 식약처 조리식품의 레시피 DB (COOKRCP01) Open API.

    The "sample" API key always returns the same couple of demo rows
    regardless of the requested index range, so it's only useful to verify
    request/parsing logic works, not to fetch the full dataset. A personal
    key (see .env.example) is required for real ingestion.
    """

    def __init__(self, api_key: str, http_client: httpx.Client | None = None):
        self._api_key = api_key
        self._http = http_client or httpx.Client(timeout=10.0)

    def fetch_page(self, start_idx: int, end_idx: int) -> tuple[list[Recipe], int]:
        """Fetch rows in the 1-based, inclusive range [start_idx, end_idx].

        Returns (recipes, total_count).
        """
        url = f"{BASE_URL}/{self._api_key}/{SERVICE_ID}/json/{start_idx}/{end_idx}"
        response = self._http.get(url)
        response.raise_for_status()
        payload = response.json()

        body = payload.get(SERVICE_ID)
        if body is None:
            raise FoodSafetyApiError(f"Unexpected response shape: {payload!r}")

        rows = body.get("row")
        if rows is None:
            result = body.get("RESULT", {})
            raise FoodSafetyApiError(
                f"API error {result.get('CODE')}: {result.get('MSG')}"
            )

        total_count = int(body.get("total_count", len(rows)))
        return [Recipe.from_api_row(row) for row in rows], total_count

    def fetch_all(self, page_size: int = 100):
        """Yield every recipe in the dataset, paging through the API."""
        start_idx = 1
        while True:
            end_idx = start_idx + page_size - 1
            recipes, total_count = self.fetch_page(start_idx, end_idx)
            if not recipes:
                return
            yield from recipes
            if end_idx >= total_count:
                return
            start_idx = end_idx + 1

    def close(self) -> None:
        self._http.close()
