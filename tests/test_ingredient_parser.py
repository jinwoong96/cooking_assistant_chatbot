from cooking_assistant_chatbot.pricing.ingredient_parser import parse_ingredients


def _names(raw: str, recipe_name: str = "") -> list[str]:
    return [p.name for p in parse_ingredients(raw, recipe_name=recipe_name)]


def test_drops_leading_line_that_duplicates_the_recipe_name():
    raw = (
        "새우두부계란찜\n"
        "연두부 75g(3/4모), 칵테일새우 20g(5마리)\n"
        "고명\n"
        "시금치 10g(3줄기)"
    )

    names = _names(raw, recipe_name="새우두부계란찜")

    assert "새우두부계란찜" not in names


def test_drops_leading_line_duplicating_recipe_name_despite_spacing_difference():
    # Real data quirk: RCP_NM has spaces ("새우 두부 계란찜") but the first
    # ingredients_raw line glues them together ("새우두부계란찜").
    raw = "새우두부계란찜\n연두부 75g(3/4모), 칵테일새우 20g(5마리)"

    names = _names(raw, recipe_name="새우 두부 계란찜")

    assert "새우두부계란찜" not in names
    assert names == ["연두부", "칵테일새우"]


def test_drops_standalone_label_line():
    raw = "무염버터 5g(1작은술)\n고명\n시금치 10g(3줄기)"

    assert _names(raw) == ["무염버터", "시금치"]


def test_strips_leading_jaeryo_prefix():
    raw = "재료 토란(40g), 표고버섯(20g), 붉은 고추(1g)"

    assert _names(raw) == ["토란", "표고버섯", "붉은 고추"]


def test_strips_colon_delimited_section_label():
    raw = "함초 25g, 알배추 50g\n국물 : 오렌지주스 50g, 유자청 30g"

    assert _names(raw) == ["함초", "알배추", "오렌지주스", "유자청"]


def test_strips_leading_bare_label_word_before_ingredient():
    raw = "육수 닭가슴살(50g), 대파(20g)\n양념 저염된장(5g), 다진 마늘(5g)"

    assert _names(raw) == ["닭가슴살", "대파", "저염된장", "다진 마늘"]


def test_handles_unspecified_quantity_suffixes_with_and_without_space():
    raw = "소금적당량, 후추 적당량, 참깨 약간"

    ingredients = parse_ingredients(raw)

    assert [(i.name, i.quantity_text) for i in ingredients] == [
        ("소금", "적당량"),
        ("후추", "적당량"),
        ("참깨", "약간"),
    ]


def test_extracts_quantity_text_for_parenthesized_and_plain_amounts():
    raw = "닭가슴살(60g), 애호박 30g"

    ingredients = parse_ingredients(raw)

    assert ingredients[0].quantity_text == "(60g)"
    assert ingredients[1].quantity_text == "30g"


def test_strips_leading_serving_size_bracket():
    raw = "[1인분]조선부추 50g, 날콩가루 7g(1⅓작은술)"

    assert _names(raw) == ["조선부추", "날콩가루"]


def test_empty_input_returns_empty_list():
    assert parse_ingredients("") == []


def test_strips_black_circle_bullet_with_colon_label():
    raw = "●멸치육수 : 국물용 멸치, 다시마"

    assert _names(raw) == ["국물용 멸치", "다시마"]


def test_strips_bullet_dot_with_colon_label():
    raw = "•필수 재료 : 주꾸미, 청양고추"

    assert _names(raw) == ["주꾸미", "청양고추"]


def test_strips_multi_word_colon_label():
    raw = "치커리 샐러드 : 치커리\n올리브마늘 드레싱 : 올리브유"

    assert _names(raw) == ["치커리", "올리브유"]


def test_splits_unicode_fraction_quantity_off_the_name():
    result = parse_ingredients("게살(½컵), 우유 ⅔컵")

    assert [(i.name, i.quantity_text) for i in result] == [("게살", "(½컵)"), ("우유", "⅔컵")]


def test_drops_cookware_listed_as_an_ingredient():
    result = parse_ingredients("뚝배기, 달걀 3개, 꼬치, 꼬치어묵 2개")

    assert [i.name for i in result] == ["달걀", "꼬치어묵"]
