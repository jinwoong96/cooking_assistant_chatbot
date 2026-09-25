from __future__ import annotations

import re

from pydantic import BaseModel

# Section-label words seen in real RCP_PARTS_DTLS data (e.g. "육수 닭가슴살(50g)"
# where "육수" is a label, not an ingredient). Colon-delimited labels (e.g.
# "국물 : 오렌지주스 50g") are handled separately and don't need to be listed here.
_LABEL_WORDS = {
    "재료",
    "육수",
    "국물",
    "소스",
    "양념",
    "양념장",
    "고명",
    "밑간",
    "절임물",
    "튀김옷",
    "곁들임",
    "곁들임채소",
    "드레싱",
    "소스재료",
}

# Cookware that some recipes (mostly the 만개의레시피 crawl) list alongside
# ingredients. Priced as groceries, "뚝배기" added a 2,100원 pot to a 계란찜.
# Exact-name matches only, so "꼬치어묵" etc. are unaffected.
_COOKWARE = {
    "뚝배기",
    "냄비",
    "프라이팬",
    "팬",
    "웍",
    "그릇",
    "볼",
    "전자레인지",
    "에어프라이어",
    "오븐",
    "찜기",
    "랩",
    "호일",
    "쿠킹호일",
    "종이호일",
    "키친타월",
    "이쑤시개",
    "꼬치",
    "꼬치용 꼬치",
    "나무젓가락",
    "종이컵",
    "지퍼백",
    "위생장갑",
}

# Unicode vulgar fractions count as the start of a quantity too: without
# them "게살(½컵)" kept its amount glued onto the name (25 names in the DB).
_DIGIT_RE = re.compile(r"[\d½⅓⅔¼¾⅛⅜⅝⅞]")
_LEADING_BRACKET_RE = re.compile(r"^\[[^\]]*\]\s*")
_LEADING_LABEL_RE = re.compile(r"^재료\s*")
_LINE_LABEL_RE = re.compile(r"^([가-힣A-Za-z0-9][가-힣A-Za-z0-9 ]{0,14})\s*[:：]\s*(.+)$")
_TRAILING_CHARS = " (（-:：·"
_UNSPECIFIED_QUANTITY_SUFFIXES = ("적당량", "약간")


class ParsedIngredient(BaseModel):
    name: str
    quantity_text: str = ""


def _clean_name(name: str) -> str:
    return name.strip().rstrip(_TRAILING_CHARS).strip()


def _normalize_spacing(text: str) -> str:
    """Strip all whitespace for comparison — the source data sometimes
    repeats the recipe name as the first ingredients_raw line with different
    spacing (e.g. name "새우 두부 계란찜" vs. first line "새우두부계란찜"),
    which would otherwise dodge the duplicate-name-line check below and let
    the glued-together dish name slip in as a bogus "ingredient"."""
    return text.strip().replace(" ", "")


def _split_name_and_quantity(token: str) -> tuple[str, str]:
    match = _DIGIT_RE.search(token)
    if match:
        idx = match.start()
        if idx > 0 and token[idx - 1] == "(":
            idx -= 1
        return _clean_name(token[:idx]), token[idx:].strip()
    for suffix in _UNSPECIFIED_QUANTITY_SUFFIXES:
        if token.endswith(suffix):
            return _clean_name(token[: -len(suffix)]), suffix
    return _clean_name(token), ""


def _strip_leading_label_word(token: str) -> str:
    if " " not in token:
        return token
    first, rest = token.split(" ", 1)
    if first in _LABEL_WORDS and rest.strip():
        return rest
    return token


def _strip_line_label_prefix(line: str) -> str:
    line = line.strip().lstrip("-·●•*").strip()
    match = _LINE_LABEL_RE.match(line)
    if match:
        return match.group(2)
    return line


def parse_ingredients(ingredients_raw: str, recipe_name: str = "") -> list[ParsedIngredient]:
    """Best-effort rule-based extraction of ingredient names + raw quantity text.

    The source data (RCP_PARTS_DTLS) is inconsistent: it sometimes repeats the
    recipe name as the first line, sometimes prefixes with "재료"/"[1인분]", and
    uses a mix of "라벨 : 재료들" and bare "라벨\n재료" section markers. This
    parser handles the common cases seen in the real dataset; genuinely
    ambiguous cases (e.g. multi-word labels with no colon) may leak into the
    ingredient name. Good enough for MVP search-term extraction — an LLM-based
    correction pass for ambiguous cases is deferred to a later feature.
    """
    if not ingredients_raw:
        return []

    text = _LEADING_BRACKET_RE.sub("", ingredients_raw.strip())
    text = _LEADING_LABEL_RE.sub("", text)

    lines = text.split("\n")
    if lines and recipe_name and _normalize_spacing(lines[0]) == _normalize_spacing(recipe_name):
        lines = lines[1:]

    results: list[ParsedIngredient] = []
    for line in lines:
        line = _strip_line_label_prefix(line)
        if not line.strip():
            continue
        for raw_token in line.split(","):
            token = raw_token.strip()
            if not token or token in _LABEL_WORDS:
                continue
            token = _strip_leading_label_word(token)
            name, quantity_text = _split_name_and_quantity(token)
            if not name or name in _COOKWARE:
                continue
            results.append(ParsedIngredient(name=name, quantity_text=quantity_text))
    return results
