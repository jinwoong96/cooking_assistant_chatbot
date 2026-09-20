from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx
from pydantic import BaseModel

BASE_URL = "http://openapi.11st.co.kr/openapi/OpenApiService.tmall"
# The API returns EUC-KR/CP949-encoded XML regardless of Accept-Charset.
_RESPONSE_ENCODING = "cp949"
_SORT_PRICE_ASC = "G"


class ElevenStApiError(RuntimeError):
    """Raised when the 11번가 Open API returns something other than a Products list."""


class ShoppingItem(BaseModel):
    title: str
    price: int
    seller: str = ""
    link: str = ""


class ElevenStClient:
    """Client for the 11번가 Open API product search (ProductSearch).

    Replaces the Naver Shopping search API, which was officially shut down
    2026-07-31 with no replacement. Chosen because, unlike 쿠팡파트너스, it
    doesn't require first generating 150,000원 of affiliate sales before the
    API is activated.

    NOTE: the exact error-response XML schema hasn't been verified against a
    live key yet (no official schema found during research). If the real API
    returns something other than a bare <Products> root on error, this will
    surface as ElevenStApiError with the raw response body rather than a
    parsed error code/message — adjust once tested with a real key.
    """

    def __init__(self, api_key: str, http_client: httpx.Client | None = None):
        self._api_key = api_key
        self._http = http_client or httpx.Client(timeout=10.0)

    def search_cheapest(self, query: str, page_size: int = 5) -> list[ShoppingItem]:
        response = self._http.get(
            BASE_URL,
            params={
                "key": self._api_key,
                "apiCode": "ProductSearch",
                "keyword": query,
                "sortCd": _SORT_PRICE_ASC,
                "pageNum": 1,
                "pageSize": page_size,
            },
        )
        response.raise_for_status()
        xml_text = response.content.decode(_RESPONSE_ENCODING)

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            raise ElevenStApiError(f"Could not parse response as XML: {xml_text[:300]}") from e

        if root.tag != "Products":
            raise ElevenStApiError(f"Unexpected response (root=<{root.tag}>): {xml_text[:300]}")

        items = []
        for product in root.findall("Product"):
            sale_price = product.findtext("SalePrice")
            list_price = product.findtext("ProductPrice")
            price_text = sale_price if sale_price and sale_price != "0" else list_price
            if not price_text:
                continue
            items.append(
                ShoppingItem(
                    title=(product.findtext("ProductName") or "").strip(),
                    price=int(price_text),
                    seller=(product.findtext("Seller") or "").strip(),
                    link=(product.findtext("DetailPageUrl") or "").strip(),
                )
            )
        return items

    def close(self) -> None:
        self._http.close()
