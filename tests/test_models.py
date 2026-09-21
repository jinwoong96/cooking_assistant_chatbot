from cooking_assistant_chatbot.data.models import Recipe

RAW_ROW = {
    "RCP_SEQ": "28",
    "RCP_NM": "새우 두부 계란찜",
    "RCP_WAY2": "찌기",
    "RCP_PAT2": "반찬",
    "RCP_PARTS_DTLS": "연두부 75g(3/4모), 칵테일새우 20g(5마리)",
    "HASH_TAG": "연두부",
    "RCP_NA_TIP": "간을 줄이는 팁",
    "ATT_FILE_NO_MAIN": "http://example.com/main.png",
    "ATT_FILE_NO_MK": "http://example.com/thumb.png",
    "INFO_WGT": "",
    "INFO_ENG": "220",
    "INFO_CAR": "3",
    "INFO_PRO": "14",
    "INFO_FAT": "17",
    "INFO_NA": "99",
    "MANUAL01": "1. 새우를 데친다.",
    "MANUAL_IMG01": "",
    "MANUAL02": "2. 재료를 섞는다.",
    "MANUAL_IMG02": "http://example.com/step2.png",
    "MANUAL03": "",
    "MANUAL_IMG03": "http://example.com/unused.png",
}


def test_from_api_row_maps_known_fields():
    recipe = Recipe.from_api_row(RAW_ROW)

    assert recipe.rcp_seq == "28"
    assert recipe.name == "새우 두부 계란찜"
    assert recipe.category == "반찬"
    assert recipe.energy_kcal == "220"


def test_from_api_row_only_keeps_non_empty_steps_in_order():
    recipe = Recipe.from_api_row(RAW_ROW)

    assert recipe.steps == ["1. 새우를 데친다.", "2. 재료를 섞는다."]
    # image for the empty MANUAL03 step must not leak in
    assert recipe.step_image_urls == ["", "http://example.com/step2.png"]


def test_from_api_row_defaults_missing_fields_to_empty_string():
    recipe = Recipe.from_api_row({"RCP_SEQ": "1", "RCP_NM": "빈 레시피"})

    assert recipe.ingredients_raw == ""
    assert recipe.steps == []


def test_servings_parses_bracket_prefix():
    recipe = Recipe(rcp_seq="1", name="삼겹살쌈", ingredients_raw="[ 2인분 ] 삼겹살(200g), 상추(50g)")

    assert recipe.servings == 2


def test_servings_parses_bracket_prefix_without_spaces():
    recipe = Recipe(rcp_seq="1", name="부추찜", ingredients_raw="[1인분]조선부추 50g")

    assert recipe.servings == 1


def test_servings_is_none_when_not_stated():
    recipe = Recipe(rcp_seq="1", name="김치찌개", ingredients_raw="김치 200g, 돼지고기 150g")

    assert recipe.servings is None
