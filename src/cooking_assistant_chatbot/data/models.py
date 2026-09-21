from __future__ import annotations

import re

from pydantic import BaseModel, Field

_MANUAL_STEPS = 20
_SERVINGS_RE = re.compile(r"^\[\s*(\d+)\s*인분\s*\]")


class Recipe(BaseModel):
    """A single recipe as returned by the 식약처 COOKRCP01 Open API.

    Nutrition fields are kept as raw strings: the source data is inconsistent
    (some rows use empty strings, ranges, or "-" instead of a number), and MVP
    doesn't need them as numbers yet.
    """

    rcp_seq: str
    name: str
    category: str = ""
    cooking_method: str = ""
    ingredients_raw: str = ""
    hash_tag: str = ""
    na_tip: str = ""
    main_image_url: str = ""
    thumbnail_image_url: str = ""
    weight_info: str = ""
    energy_kcal: str = ""
    carbohydrate_g: str = ""
    protein_g: str = ""
    fat_g: str = ""
    sodium_mg: str = ""
    steps: list[str] = Field(default_factory=list)
    step_image_urls: list[str] = Field(default_factory=list)

    @property
    def servings(self) -> int | None:
        """Number of servings this recipe makes, when the source data says so.

        Only ~3.5% of recipes in the dataset state this, as a "[N인분]"
        prefix on `ingredients_raw` (e.g. "[ 2인분 ] 삼겹살(200g), ..."). No
        other field in the source API reliably gives a serving count
        (`INFO_WGT` is grams *per* serving, not how many servings there are,
        and is itself empty most of the time). Returns None rather than
        guessing when the source doesn't say.
        """
        match = _SERVINGS_RE.match(self.ingredients_raw.strip())
        return int(match.group(1)) if match else None

    @classmethod
    def from_api_row(cls, row: dict[str, str]) -> "Recipe":
        steps = []
        step_image_urls = []
        for i in range(1, _MANUAL_STEPS + 1):
            step = row.get(f"MANUAL{i:02d}", "").strip()
            if step:
                steps.append(step)
                step_image_urls.append(row.get(f"MANUAL_IMG{i:02d}", "").strip())

        return cls(
            rcp_seq=row.get("RCP_SEQ", ""),
            name=row.get("RCP_NM", "").strip(),
            category=row.get("RCP_PAT2", "").strip(),
            cooking_method=row.get("RCP_WAY2", "").strip(),
            ingredients_raw=row.get("RCP_PARTS_DTLS", "").strip(),
            hash_tag=row.get("HASH_TAG", "").strip(),
            na_tip=row.get("RCP_NA_TIP", "").strip(),
            main_image_url=row.get("ATT_FILE_NO_MAIN", "").strip(),
            thumbnail_image_url=row.get("ATT_FILE_NO_MK", "").strip(),
            weight_info=row.get("INFO_WGT", "").strip(),
            energy_kcal=row.get("INFO_ENG", "").strip(),
            carbohydrate_g=row.get("INFO_CAR", "").strip(),
            protein_g=row.get("INFO_PRO", "").strip(),
            fat_g=row.get("INFO_FAT", "").strip(),
            sodium_mg=row.get("INFO_NA", "").strip(),
            steps=steps,
            step_image_urls=step_image_urls,
        )
