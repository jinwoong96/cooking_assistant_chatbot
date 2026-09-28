from cooking_assistant_chatbot.agent import pipeline
from cooking_assistant_chatbot.data.db import get_connection, upsert_recipes
from cooking_assistant_chatbot.data.models import Recipe
from cooking_assistant_chatbot.pricing.enuri_client import ShoppingItem
from tests.fakes import fake_text_response, fake_tool_call_response


class _StubSearcher:
    def __init__(self, recipes: list[Recipe]):
        self._recipes = recipes

    def search(self, query: str, top_k: int = 1) -> list[Recipe]:
        return self._recipes[:top_k]


class _StubPriceClient:
    def __init__(self):
        self.queries = []

    def search(self, query: str) -> list[ShoppingItem]:
        self.queries.append(query)
        return [ShoppingItem(title=f"{query} 상품", price=1000)]


def _recipe() -> Recipe:
    return Recipe(
        rcp_seq="1",
        name="김치찌개",
        ingredients_raw="김치 200g",
        steps=["1. 끓인다"],
        energy_kcal="350",
        protein_g="20",
    )


def _scripted_chat(responses: list):
    """Fake `chat` that returns `responses` in order and records each call."""
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return responses[len(calls) - 1]

    return fake_chat, calls


def _tool_results(call_kwargs: dict) -> list[str]:
    return [m["content"] for m in call_kwargs["messages"] if m.get("role") == "tool"]


def _run(monkeypatch, tmp_path, responses, message="질문", recipes=None, **kwargs):
    fake_chat, calls = _scripted_chat(responses)
    monkeypatch.setattr(pipeline, "chat", fake_chat)
    conn = get_connection(str(tmp_path / "test.db"))
    recipes = [_recipe()] if recipes is None else recipes
    upsert_recipes(conn, recipes)
    price_client = _StubPriceClient()
    reply = pipeline.handle_message(
        message, _StubSearcher(recipes), price_client, conn, **kwargs
    )
    return reply, calls, price_client


def test_plain_answer_without_tools(tmp_path, monkeypatch):
    reply, calls, _ = _run(monkeypatch, tmp_path, [fake_text_response("안녕하세요!")], "안녕")

    assert reply.text == "안녕하세요!"
    assert len(calls) == 1


def test_nutrition_question_does_not_look_up_prices(tmp_path, monkeypatch):
    reply, calls, price_client = _run(
        monkeypatch,
        tmp_path,
        [
            fake_tool_call_response("get_nutrition", {"recipe_name": "김치찌개"}),
            fake_text_response("김치찌개는 350kcal예요."),
        ],
        "김치찌개 칼로리 얼마야?",
    )

    assert reply.text == "김치찌개는 350kcal예요."
    assert "350kcal" in _tool_results(calls[1])[0]
    assert price_client.queries == []


def test_cost_tool_prices_ingredients_and_grounds_reply(tmp_path, monkeypatch):
    reply, calls, price_client = _run(
        monkeypatch,
        tmp_path,
        [
            fake_tool_call_response("estimate_ingredient_cost", {"recipe_name": "김치찌개"}),
            fake_text_response("재료비는 1,000원이에요."),
        ],
        "김치찌개 재료비 얼마야?",
    )

    assert price_client.queries == ["김치"]
    assert "총액: 1,000원" in _tool_results(calls[1])[0]


def test_chains_multiple_tool_rounds(tmp_path, monkeypatch):
    reply, calls, _ = _run(
        monkeypatch,
        tmp_path,
        [
            fake_tool_call_response("search_recipes", {"query": "국물요리"}),
            fake_tool_call_response("estimate_ingredient_cost", {"recipe_name": "김치찌개"}),
            fake_text_response("김치찌개 추천해요. 재료비 1,000원."),
        ],
    )

    assert reply.text == "김치찌개 추천해요. 재료비 1,000원."
    results = _tool_results(calls[2])
    assert len(results) == 2
    assert "김치찌개" in results[0]
    assert "총액" in results[1]


def test_forces_an_answer_after_max_tool_rounds(tmp_path, monkeypatch):
    looping = fake_tool_call_response("search_recipes", {"query": "국물"})
    responses = [looping] * pipeline.MAX_TOOL_ROUNDS + [fake_text_response("정리한 답변")]

    reply, calls, _ = _run(monkeypatch, tmp_path, responses)

    assert reply.text == "정리한 답변"
    assert len(calls) == pipeline.MAX_TOOL_ROUNDS + 1
    assert calls[-1]["tools"] is None
    assert all(c["tools"] for c in calls[:-1])


def test_history_is_passed_as_prior_turns(tmp_path, monkeypatch):
    history = [
        {"role": "user", "content": "두부 요리 뭐 있어?"},
        {"role": "assistant", "content": "두부조림 어때요?"},
    ]

    _, calls, _ = _run(
        monkeypatch, tmp_path, [fake_text_response("이어서")], "그거 말고?", history=history
    )

    assert calls[0]["messages"][1:3] == history


def test_ingredients_search_tool_uses_exact_index(tmp_path, monkeypatch):
    from cooking_assistant_chatbot.data.ingredient_index import build_ingredient_index

    fake_chat, calls = _scripted_chat(
        [
            fake_tool_call_response("search_recipes_by_ingredients", {"ingredients": ["두부", "계란"]}),
            fake_text_response("두부계란찜 어때요?"),
        ]
    )
    monkeypatch.setattr(pipeline, "chat", fake_chat)
    conn = get_connection(str(tmp_path / "test.db"))
    upsert_recipes(conn, [Recipe(rcp_seq="10", name="두부계란찜", ingredients_raw="두부 100g, 계란 2개")])
    build_ingredient_index(conn)

    pipeline.handle_message("두부랑 계란 있어", _StubSearcher([]), _StubPriceClient(), conn)

    assert "두부계란찜" in _tool_results(calls[1])[0]


def test_speech_is_the_reply_with_markdown_stripped(tmp_path, monkeypatch):
    reply, _, _ = _run(
        monkeypatch, tmp_path, [fake_text_response("### 팁\n- **칼**은 날카롭게 😊")]
    )

    assert reply.speech == "팁\n칼은 날카롭게"


def test_reports_progress_through_tool_calls(tmp_path, monkeypatch):
    statuses = []

    _run(
        monkeypatch,
        tmp_path,
        [
            fake_tool_call_response("estimate_ingredient_cost", {"recipe_name": "김치찌개"}),
            fake_text_response("답변"),
        ],
        "김치찌개 재료비",
        on_progress=statuses.append,
    )

    assert statuses == [
        "요청 이해하는 중",
        "'김치찌개' 레시피 찾는 중",
        "재료 가격 조회 중 (1/1 · 김치)",
        "답변 작성 중",
    ]


class _SizedPriceClient:
    def search(self, query: str) -> list[ShoppingItem]:
        return [
            ShoppingItem(title=f"{query} 10kg", price=9000),
            ShoppingItem(title=f"{query} 500g", price=3000),
        ]


def test_ui_price_basis_is_used_unless_the_model_overrides_it(tmp_path, monkeypatch):
    def run(tool_args, **kwargs):
        fake_chat, calls = _scripted_chat(
            [fake_tool_call_response("estimate_ingredient_cost", tool_args), fake_text_response("답")]
        )
        monkeypatch.setattr(pipeline, "chat", fake_chat)
        conn = get_connection(str(tmp_path / f"{len(kwargs)}{len(tool_args)}.db"))
        upsert_recipes(conn, [_recipe()])
        pipeline.handle_message("김치찌개 재료비", _StubSearcher([_recipe()]), _SizedPriceClient(), conn, **kwargs)
        return _tool_results(calls[1])[0]

    ui_default = run({"recipe_name": "김치찌개"}, price_basis="min_spend")
    overridden = run({"recipe_name": "김치찌개", "price_basis": "unit_price"}, price_basis="min_spend")

    assert "가격 기준: 최소 지출" in ui_default
    assert "김치 500g" in ui_default
    assert "가격 기준: 단위가격 우선" in overridden
    assert "김치 10kg" in overridden


def test_recipe_tool_is_not_run_for_a_dish_nobody_mentioned(tmp_path, monkeypatch):
    reply, calls, price_client = _run(
        monkeypatch,
        tmp_path,
        [
            fake_tool_call_response("estimate_ingredient_cost", {"recipe_name": "김치찌개"}),
            fake_text_response("어떤 메뉴 말씀이세요?"),
        ],
        "그거 재료비는?",
    )

    assert price_client.queries == []
    assert "추측하지 말고" in _tool_results(calls[1])[0]


def test_recipe_named_earlier_in_history_is_grounded(tmp_path, monkeypatch):
    history = [
        {"role": "user", "content": "된장 두부찌개 알려줘"},
        {"role": "assistant", "content": "된장 두부찌개는 이렇게 만들어요."},
    ]

    _, calls, _ = _run(
        monkeypatch,
        tmp_path,
        [
            fake_tool_call_response("get_nutrition", {"recipe_name": "김치찌개"}),
            fake_text_response("답"),
        ],
        "그거 칼로리는?",
        history=history,
    )

    # Shares only "찌개" with the history: not the dish that was discussed.
    assert "추측하지 말고" in _tool_results(calls[1])[0]


def test_is_grounded_tolerates_spacing_and_small_wording_differences():
    assert pipeline.is_grounded("된장찌개", "된장 두부찌개 레시피예요")
    assert pipeline.is_grounded("쉬운 계란찜", "- 맛있는 계란찜 만드는법 쉬운 계란찜 레시피")
    assert not pipeline.is_grounded("김치찌개", "그거 칼로리는?")
    assert not pipeline.is_grounded("", "아무 말")


def test_reply_remembers_the_recipe_whose_details_were_shown(tmp_path, monkeypatch):
    reply, _, _ = _run(
        monkeypatch,
        tmp_path,
        [fake_tool_call_response("get_recipe", {"recipe_name": "김치찌개"}), fake_text_response("레시피")],
        "김치찌개 레시피",
    )
    nutrition_only, _, _ = _run(
        monkeypatch,
        tmp_path,
        [fake_tool_call_response("get_nutrition", {"recipe_name": "김치찌개"}), fake_text_response("칼로리")],
        "김치찌개 칼로리",
    )

    assert reply.recipe_seq == "1"
    assert nutrition_only.recipe_seq is None


def test_chat_can_draft_a_recipe_for_the_registration_form(tmp_path, monkeypatch):
    reply, calls, _ = _run(
        monkeypatch,
        tmp_path,
        [
            fake_tool_call_response(
                "prepare_recipe_registration",
                {
                    "name": "우리집 김치볶음밥",
                    "servings": 2,
                    "ingredients": ["밥 2공기", "김치 1컵"],
                    "steps": ["1. 김치를 볶는다", "밥을 넣고 볶는다"],
                    "energy_kcal": 600,
                },
            ),
            fake_text_response("레시피 등록 탭에서 확인 후 저장해주세요."),
        ],
        "내 김치볶음밥 레시피 등록할래. 밥 2공기, 김치 1컵...",
    )

    draft = reply.recipe_draft
    assert draft.name == "우리집 김치볶음밥" and draft.servings == 2
    assert draft.steps == ["김치를 볶는다", "밥을 넣고 볶는다"]
    assert draft.nutrition == {"energy_kcal": "600"}
    assert "저장 안 됨" in _tool_results(calls[1])[0]


def _recipe_without_nutrition() -> Recipe:
    return Recipe(rcp_seq="10000recipe_1", name="김치볶음밥", ingredients_raw="김치 1컵", steps=["볶는다"])


def test_made_up_calories_for_a_recipe_without_nutrition_are_rewritten(tmp_path, monkeypatch):
    reply, calls, _ = _run(
        monkeypatch,
        tmp_path,
        [
            fake_tool_call_response("get_nutrition", {"recipe_name": "김치볶음밥"}),
            fake_text_response("정보는 없지만 대략 300~400kcal 정도예요."),
            fake_text_response("김치볶음밥 레시피는 영양성분 정보가 없어요."),
        ],
        "김치볶음밥 칼로리는?",
        recipes=[_recipe_without_nutrition()],
    )

    assert reply.text == "김치볶음밥 레시피는 영양성분 정보가 없어요."
    assert calls[2]["messages"][-1]["role"] == "user"
    assert "영양성분 정보가 없다" in calls[2]["messages"][-1]["content"]


def test_falls_back_to_a_fixed_reply_if_the_rewrite_still_guesses(tmp_path, monkeypatch):
    reply, _, _ = _run(
        monkeypatch,
        tmp_path,
        [
            fake_tool_call_response("get_nutrition", {"recipe_name": "김치볶음밥"}),
            fake_text_response("대략 450kcal예요."),
            fake_text_response("그래도 450 kcal 정도예요."),
        ],
        "김치볶음밥 칼로리는?",
        recipes=[_recipe_without_nutrition()],
    )

    assert "kcal" not in reply.text
    assert "영양성분 정보가 없어서" in reply.text


def test_reply_without_numbers_for_missing_nutrition_is_kept(tmp_path, monkeypatch):
    reply, calls, _ = _run(
        monkeypatch,
        tmp_path,
        [
            fake_tool_call_response("get_nutrition", {"recipe_name": "김치볶음밥"}),
            fake_text_response("이 레시피는 영양성분 정보가 없어요."),
        ],
        "김치볶음밥 칼로리는?",
        recipes=[_recipe_without_nutrition()],
    )

    assert reply.text == "이 레시피는 영양성분 정보가 없어요."
    assert len(calls) == 2
