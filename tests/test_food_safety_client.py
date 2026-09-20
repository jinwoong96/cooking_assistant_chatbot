import httpx
import pytest
import respx

from cooking_assistant_chatbot.data.food_safety_client import (
    BASE_URL,
    SERVICE_ID,
    FoodSafetyApiError,
    FoodSafetyClient,
)


def _page_payload(rows: list[dict], total_count: int) -> dict:
    return {SERVICE_ID: {"total_count": str(total_count), "row": rows}}


def _row(seq: str, name: str) -> dict:
    return {"RCP_SEQ": seq, "RCP_NM": name}


@respx.mock
def test_fetch_page_parses_rows_and_total_count():
    respx.get(f"{BASE_URL}/testkey/{SERVICE_ID}/json/1/2").mock(
        return_value=httpx.Response(
            200, json=_page_payload([_row("1", "A"), _row("2", "B")], total_count=2)
        )
    )
    client = FoodSafetyClient(api_key="testkey")

    recipes, total_count = client.fetch_page(1, 2)

    assert total_count == 2
    assert [r.name for r in recipes] == ["A", "B"]


@respx.mock
def test_fetch_page_raises_on_error_payload():
    respx.get(f"{BASE_URL}/badkey/{SERVICE_ID}/json/1/2").mock(
        return_value=httpx.Response(
            200,
            json={
                SERVICE_ID: {
                    "RESULT": {"CODE": "INFO-200", "MSG": "인증키가 유효하지 않습니다."}
                }
            },
        )
    )
    client = FoodSafetyClient(api_key="badkey")

    with pytest.raises(FoodSafetyApiError, match="INFO-200"):
        client.fetch_page(1, 2)


@respx.mock
def test_fetch_all_pages_through_until_total_count_reached():
    respx.get(f"{BASE_URL}/testkey/{SERVICE_ID}/json/1/2").mock(
        return_value=httpx.Response(
            200, json=_page_payload([_row("1", "A"), _row("2", "B")], total_count=3)
        )
    )
    respx.get(f"{BASE_URL}/testkey/{SERVICE_ID}/json/3/4").mock(
        return_value=httpx.Response(
            200, json=_page_payload([_row("3", "C")], total_count=3)
        )
    )
    client = FoodSafetyClient(api_key="testkey")

    names = [r.name for r in client.fetch_all(page_size=2)]

    assert names == ["A", "B", "C"]
