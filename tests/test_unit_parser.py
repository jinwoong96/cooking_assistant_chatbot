from cooking_assistant_chatbot.pricing.unit_parser import parse_quantity


def test_parses_grams():
    q = parse_quantity("닭가슴살(60g)")
    assert q.value == 60
    assert q.unit == "g"


def test_parses_kilograms_converted_to_grams():
    q = parse_quantity("다인 냉동 다진 마늘 1kg")
    assert q.value == 1000
    assert q.unit == "g"


def test_parses_milliliters():
    q = parse_quantity("오뚜기 옛날 참기름 320ml")
    assert q.value == 320
    assert q.unit == "ml"


def test_parses_liters_converted_to_milliliters():
    q = parse_quantity("생수 2L")
    assert q.value == 2000
    assert q.unit == "ml"


def test_returns_none_for_count_based_quantity():
    assert parse_quantity("황금란 5구") is None


def test_returns_none_for_vague_quantity():
    assert parse_quantity("약간") is None
    assert parse_quantity("적당량") is None


def test_returns_none_when_no_quantity_present():
    assert parse_quantity("이마트 소소한 하루 청양고추") is None


def test_ignores_trailing_count_words_after_a_valid_weight_match():
    q = parse_quantity("모들채소 대파썰기 100g 1팩 1개")
    assert q.value == 100
    assert q.unit == "g"


def test_matches_weight_immediately_followed_by_korean_particle():
    q = parse_quantity("500g짜리 소포장")
    assert q.value == 500
    assert q.unit == "g"


def test_does_not_match_unit_embedded_in_a_longer_latin_word():
    assert parse_quantity("50grams of flour") is None


def test_package_quantity_takes_the_largest_amount():
    from cooking_assistant_chatbot.pricing.unit_parser import parse_package_quantity

    assert parse_package_quantity("수미감자 소 (조림용 40g 미만) 10kg").value == 10000
    assert parse_package_quantity("올리브오일 140g (10g x 14포)").value == 140
    assert parse_package_quantity("진간장 1.7L + 500ml").value == 1700
    assert parse_package_quantity("계란 30구") is None
