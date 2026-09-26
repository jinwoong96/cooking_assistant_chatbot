"""User-registered recipes, browsed as a board (게시판).

Deliberately kept out of the `recipes` table: the user wants these readable
as posts, not mixed into search / recommendations / the ingredient index /
cost lookups. A separate table means no index rebuild can pick them up by
accident either.

"Who registered it" is the name typed into the form (the browser remembers
it). This is a single-password personal app with no real accounts, so
edit/delete permission is "same name as the author", not verified identity.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS user_recipes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    servings INTEGER,
    ingredients_json TEXT NOT NULL,
    steps_json TEXT NOT NULL,
    nutrition_json TEXT NOT NULL,
    author TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_user_recipes_updated ON user_recipes(updated_at);
"""

_STEP_NUMBER_RE = re.compile(r"^\s*\d+\s*[.)]\s*")

NUTRITION_FIELDS = {
    "energy_kcal": ("열량", "kcal"),
    "carbohydrate_g": ("탄수화물", "g"),
    "protein_g": ("단백질", "g"),
    "fat_g": ("지방", "g"),
    "sodium_mg": ("나트륨", "mg"),
}


@dataclass
class RecipeDraft:
    """The registration template: what the form (or the chat LLM) fills in."""

    name: str = ""
    servings: int | None = None
    ingredients: list[str] = field(default_factory=list)
    """One ingredient per entry, "재료 분량" ("간장 2큰술", "소금 약간")."""
    steps: list[str] = field(default_factory=list)
    nutrition: dict[str, str] = field(default_factory=dict)
    """Optional; keys from NUTRITION_FIELDS, values as typed ("350")."""

    def cleaned(self) -> "RecipeDraft":
        return RecipeDraft(
            name=" ".join(self.name.split()),
            servings=int(self.servings) if self.servings and self.servings > 0 else None,
            ingredients=[" ".join(i.split()) for i in self.ingredients if i.strip()],
            # Numbering is added when shown; a typed "1." would double it.
            steps=[_STEP_NUMBER_RE.sub("", s).strip() for s in self.steps if s.strip()],
            nutrition={
                k: str(v).strip()
                for k, v in self.nutrition.items()
                if k in NUTRITION_FIELDS and str(v or "").strip()
            },
        )


@dataclass
class UserRecipe:
    id: int
    draft: RecipeDraft
    author: str
    created_at: str
    updated_at: str


def validate(draft: RecipeDraft, author: str) -> list[str]:
    """Korean error messages for the form; empty when it can be saved."""
    errors = []
    if not author.strip():
        errors.append("위의 '내 이름'을 적어주세요 (등록자로 저장돼요).")
    if not draft.name:
        errors.append("메뉴 이름을 적어주세요.")
    if not draft.ingredients:
        errors.append("재료를 한 줄에 하나씩 적어주세요.")
    if not draft.steps:
        errors.append("조리 순서를 한 줄에 한 단계씩 적어주세요.")
    for value in draft.nutrition.values():
        try:
            float(value)
        except ValueError:
            errors.append(f"영양성분은 숫자로 적어주세요 ({value}).")
            break
    return errors


class PermissionDenied(Exception):
    pass


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_recipe(row) -> UserRecipe:
    return UserRecipe(
        id=row[0],
        draft=RecipeDraft(
            name=row[1],
            servings=row[2],
            ingredients=json.loads(row[3]),
            steps=json.loads(row[4]),
            nutrition=json.loads(row[5]),
        ),
        author=row[6],
        created_at=row[7],
        updated_at=row[8],
    )


_COLUMNS = (
    "id, name, servings, ingredients_json, steps_json, nutrition_json, author, created_at, updated_at"
)


def get(conn: sqlite3.Connection, recipe_id: int) -> UserRecipe | None:
    row = conn.execute(
        f"SELECT {_COLUMNS} FROM user_recipes WHERE id = ?", (recipe_id,)
    ).fetchone()
    return _row_to_recipe(row) if row else None


def search(conn: sqlite3.Connection, query: str = "", limit: int = 50) -> list[UserRecipe]:
    """Newest first; `query` matches the title or the author (spaces ignored)."""
    key = "%" + "".join(query.split()) + "%"
    rows = conn.execute(
        f"SELECT {_COLUMNS} FROM user_recipes "
        "WHERE REPLACE(name, ' ', '') LIKE ? OR REPLACE(author, ' ', '') LIKE ? "
        "ORDER BY created_at DESC, id DESC LIMIT ?",
        (key, key, limit),
    ).fetchall()
    return [_row_to_recipe(r) for r in rows]


def save(
    conn: sqlite3.Connection, draft: RecipeDraft, author: str, recipe_id: int | None = None
) -> UserRecipe:
    """Create (recipe_id None) or update a post; raises ValueError with the
    form's error messages, or PermissionDenied for someone else's post."""
    author = author.strip()
    draft = draft.cleaned()
    errors = validate(draft, author)
    if errors:
        raise ValueError(" ".join(errors))
    values = (
        draft.name,
        draft.servings,
        json.dumps(draft.ingredients, ensure_ascii=False),
        json.dumps(draft.steps, ensure_ascii=False),
        json.dumps(draft.nutrition, ensure_ascii=False),
    )
    now = _now()
    if recipe_id is None:
        cursor = conn.execute(
            "INSERT INTO user_recipes (name, servings, ingredients_json, steps_json, nutrition_json,"
            " author, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (*values, author, now, now),
        )
        recipe_id = int(cursor.lastrowid)
    else:
        existing = get(conn, recipe_id)
        if existing is None:
            raise ValueError("없는 레시피예요.")
        if existing.author != author:
            raise PermissionDenied("등록한 사람만 수정할 수 있어요.")
        conn.execute(
            "UPDATE user_recipes SET name = ?, servings = ?, ingredients_json = ?, steps_json = ?,"
            " nutrition_json = ?, updated_at = ? WHERE id = ?",
            (*values, now, recipe_id),
        )
    conn.commit()
    return get(conn, recipe_id)


def delete(conn: sqlite3.Connection, recipe_id: int, author: str) -> None:
    existing = get(conn, recipe_id)
    if existing is None:
        return
    if existing.author != author.strip():
        raise PermissionDenied("등록한 사람만 삭제할 수 있어요.")
    conn.execute("DELETE FROM user_recipes WHERE id = ?", (recipe_id,))
    conn.commit()


def render(recipe: UserRecipe) -> str:
    """A board post as markdown."""
    d = recipe.draft
    date = recipe.created_at[:10]
    lines = [f"## {d.name}", f"✍️ {recipe.author} · {date}"]
    if recipe.updated_at[:16] != recipe.created_at[:16]:
        lines[-1] += f" (수정 {recipe.updated_at[:10]})"
    if d.servings:
        lines.append(f"**{d.servings}인분**")
    lines.append("\n### 재료\n" + "\n".join(f"- {i}" for i in d.ingredients))
    lines.append("\n### 조리 순서\n" + "\n".join(f"{n}. {s}" for n, s in enumerate(d.steps, 1)))
    if d.nutrition:
        parts = [
            f"{NUTRITION_FIELDS[k][0]} {v}{NUTRITION_FIELDS[k][1]}" for k, v in d.nutrition.items()
        ]
        lines.append("\n### 영양성분 (작성자 입력)\n" + " · ".join(parts))
    return "\n".join(lines)
