from cooking_assistant_chatbot.pricing.enuri_client import ShoppingItem
from cooking_assistant_chatbot.pricing.selection import choose_item


def _item(title: str, price: int) -> ShoppingItem:
    return ShoppingItem(title=title, price=price)


def test_relevance_ignores_cheap_results_outside_the_top_pool():
    items = [_item(f"햇반 백미 210g {i}", 10000 + i) for i in range(5)]
    items.append(_item("치즈크림 라떼 파우더", 500))  # cheap but far down the list

    assert choose_item(items, "밥").title == "햇반 백미 210g 0"


def test_relevance_prefers_titles_naming_the_query():
    items = [_item("파프리카 피망 5kg", 44890), _item("장난감 계산대", 30000), _item("피망 2kg", 18890)]

    assert choose_item(items, "피망").title == "피망 2kg"


def test_relevance_falls_back_to_pool_for_spelling_variants():
    items = [_item("유정란 60구", 27960), _item("계란 특란 30구", 18660)]

    assert choose_item(items, "달걀").title == "계란 특란 30구"


def test_min_spend_picks_cheapest_package_covering_the_recipe_need():
    items = [
        _item("스팸 클래식 200g", 15870),
        _item("스팸 라이트 120g", 19350),  # multi-pack price on a small item
        _item("스팸 클래식 300g", 3430),
        _item("스팸 미니 80g", 1000),  # cheapest, but doesn't cover 100g
    ]

    assert choose_item(items, "스팸", "100g", basis="min_spend").title == "스팸 클래식 300g"


def test_min_spend_skips_sample_sachets_below_the_need():
    items = [_item("진간장 1.7L", 3710), _item("진간장 6ml", 1140), _item("진간장 500ml", 8870)]

    assert choose_item(items, "간장", "30ml", basis="min_spend").title == "진간장 1.7L"


def test_min_spend_takes_cheapest_when_need_is_unknown():
    items = [_item("감자 10kg", 5860), _item("감자 5kg", 6460), _item("감자 1kg", 3000)]

    assert choose_item(items, "감자", "1개", basis="min_spend").title == "감자 1kg"


def test_min_spend_looks_past_the_top_pool_but_only_at_the_ingredient_itself():
    items = [_item(f"감자 {i + 5}kg", 6000) for i in range(5)]
    items += [_item("감자칩 50g", 1000), _item("장난감 계산대", 500), _item("햇 수미감자 3kg", 4000)]

    assert choose_item(items, "감자", basis="min_spend").title == "햇 수미감자 3kg"


def test_ingredient_match_accepts_compounds_ending_in_the_name():
    items = [_item("대파분태 100g", 900), _item("흙대파(봉) 850g", 2980), _item("깐대파 200g", 2590)]

    assert choose_item(items, "대파", basis="min_spend").title == "깐대파 200g"


def test_package_size_is_the_largest_amount_in_the_title():
    items = [_item("수미감자 소 (조림용 40g 미만) 10kg", 6710), _item("감자 5kg", 6460)]

    # By first-match parsing the 10kg sack read as 40g, the cheapest per gram.
    assert choose_item(items, "감자", basis="unit_price").title == "수미감자 소 (조림용 40g 미만) 10kg"


def test_unit_price_picks_lowest_price_per_gram():
    items = [_item("양파 10kg", 14580), _item("양파 5kg", 6890), _item("양파 2kg", 6920)]

    assert choose_item(items, "양파", basis="unit_price").title == "양파 5kg"


def test_non_default_basis_falls_back_to_relevance_without_sizes():
    items = [_item("계란 특란 30구", 18660), _item("계란 대란 30구", 17600)]

    assert choose_item(items, "계란", basis="unit_price").title == "계란 대란 30구"
    assert choose_item(items, "달걀", "2개", basis="min_spend").title == "계란 대란 30구"
