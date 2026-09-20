import httpx
import pytest
import respx

from cooking_assistant_chatbot.pricing.eleven_st_client import (
    BASE_URL,
    ElevenStApiError,
    ElevenStClient,
)

_SAMPLE_XML = """<?xml version="1.0" encoding="euc-kr"?>
<Products>
<TotalCount>2</TotalCount>
<Product>
<ProductCode>123</ProductCode>
<ProductName>청정원 다진마늘 500g</ProductName>
<ProductPrice>4000</ProductPrice>
<SalePrice>3500</SalePrice>
<Seller>11번가</Seller>
<DetailPageUrl>https://example.com/a</DetailPageUrl>
</Product>
<Product>
<ProductCode>124</ProductCode>
<ProductName>국산 다진마늘 1kg</ProductName>
<ProductPrice>7000</ProductPrice>
<SalePrice>0</SalePrice>
<Seller>다른몰</Seller>
<DetailPageUrl>https://example.com/b</DetailPageUrl>
</Product>
</Products>"""


def _mock_response(xml: str, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, content=xml.encode("cp949"))


@respx.mock
def test_search_cheapest_parses_products_and_prefers_sale_price():
    respx.get(BASE_URL).mock(return_value=_mock_response(_SAMPLE_XML))
    client = ElevenStClient(api_key="testkey")

    items = client.search_cheapest("다진마늘")

    assert len(items) == 2
    assert items[0].title == "청정원 다진마늘 500g"
    assert items[0].price == 3500  # SalePrice preferred over ProductPrice
    assert items[0].seller == "11번가"
    assert items[1].price == 7000  # SalePrice was 0, falls back to ProductPrice


@respx.mock
def test_search_cheapest_sends_price_ascending_sort_and_keyword():
    route = respx.get(BASE_URL).mock(
        return_value=_mock_response("<Products><TotalCount>0</TotalCount></Products>")
    )
    client = ElevenStClient(api_key="testkey")

    client.search_cheapest("대파", page_size=3)

    request = route.calls.last.request
    assert request.url.params["keyword"] == "대파"
    assert request.url.params["apiCode"] == "ProductSearch"
    assert request.url.params["sortCd"] == "G"
    assert request.url.params["pageSize"] == "3"
    assert request.url.params["key"] == "testkey"


@respx.mock
def test_search_cheapest_returns_empty_list_when_no_products():
    respx.get(BASE_URL).mock(
        return_value=_mock_response("<Products><TotalCount>0</TotalCount></Products>")
    )
    client = ElevenStClient(api_key="testkey")

    assert client.search_cheapest("존재하지않는상품명xyz") == []


@respx.mock
def test_search_cheapest_raises_on_unexpected_root_element():
    respx.get(BASE_URL).mock(
        return_value=_mock_response("<Error><Code>900</Code></Error>")
    )
    client = ElevenStClient(api_key="badkey")

    with pytest.raises(ElevenStApiError, match="Error"):
        client.search_cheapest("대파")
