from __future__ import annotations

import json

from ..llm.client import chat

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "resolve_ingredient_name",
            "description": "레시피 재료 원문을 실제 쇼핑몰 검색에 적합한 형태로 정리한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "is_purchasable_ingredient": {
                        "type": "boolean",
                        "description": (
                            "이 텍스트가 실제로 마트에서 구매 가능한 단일 재료를 가리키는지. "
                            "'간 맞출 때'처럼 조리 지시사항이거나 재료가 아니면 false."
                        ),
                    },
                    "cleaned_name": {
                        "type": "string",
                        "description": (
                            "is_purchasable_ingredient가 true일 때, 쇼핑몰 검색에 쓸 간결한 "
                            "재료명 (조리법 설명·브랜드·수량은 제외, 예: '돼지고기', '다진 양파')."
                        ),
                    },
                },
                "required": ["is_purchasable_ingredient"],
            },
        },
    }
]

_SYSTEM_PROMPT = (
    "너는 레시피 재료 텍스트에서 실제 쇼핑몰 검색에 쓸 간결한 재료명을 뽑아내는 도우미다. "
    "주어진 재료 표기는 규칙 기반 파싱만으로는 애매하게 남은 것이다 (조리법 설명이 섞였거나, "
    "재료가 아닌 조리 지시사항일 수 있다). 손질/조리 방식을 설명하는 수식어(예: '완자', '얇게 "
    "썬', '채 썬', '송송 썬', '고기 삶는 재료', '밥 밑간', '무침양념')는 실제 상품명에 없는 "
    "경우가 많으니 반드시 제거하고 핵심 재료명만 남겨라. 예시:\n"
    "- '얇게 썬 쇠고기' -> '쇠고기'\n"
    "- '고기 삶는 재료 양파' -> '양파'\n"
    "- '밥 밑간 참기름' -> '참기름'\n"
    "- '완자 다진 돼지고기' -> '다진 돼지고기'\n"
    "- '간 맞출 때'처럼 재료가 아니라 조리 지시사항이면 is_purchasable_ingredient를 false로.\n"
    "전체 재료 목록 원문을 맥락으로 참고해서 판단하고, 반드시 resolve_ingredient_name 도구를 "
    "호출해라."
)


def correct_ingredient_name(raw_name: str, ingredients_context: str) -> str | None:
    """Ask the LLM to clean up an ambiguous ingredient name for shopping search.

    Returns a cleaned search term, or None if the model decides this text
    isn't actually a purchasable ingredient (e.g. "간 맞출 때" — a cooking
    instruction the rule-based parser mistakenly captured as an ingredient).
    Falls back to the original name if the model doesn't call the tool or
    doesn't give a usable cleaned name.

    Empirically (temperature=0, against 9 real ambiguous names from the DB)
    qwen3:14b cleans about half of them correctly and leaves the rest
    unchanged rather than making them worse — it doesn't invent a wrong
    correction. That's an acceptable "best effort" ceiling for a local 14B
    model on this task, matching this project's general rule-based-first
    philosophy: this only ever helps or is a no-op, never regresses the
    plain-search fallback that ran before this existed.
    """
    response = chat(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"정리가 필요한 재료 표기: {raw_name}\n"
                    f"전체 재료 목록 원문:\n{ingredients_context}"
                ),
            },
        ],
        tools=_TOOLS,
        temperature=0.0,
    )
    message = response.choices[0].message
    tool_calls = message.tool_calls or []
    if not tool_calls:
        return raw_name

    args = json.loads(tool_calls[0].function.arguments)
    if not args.get("is_purchasable_ingredient", True):
        return None

    cleaned = args.get("cleaned_name")
    return cleaned.strip() if cleaned and cleaned.strip() else raw_name
