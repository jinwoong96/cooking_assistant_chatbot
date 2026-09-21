from __future__ import annotations

import sqlite3

from ..pricing.ingredient_parser import parse_ingredients
from .db import get_recipes_by_ids
from .models import Recipe

SCHEMA = """
CREATE TABLE IF NOT EXISTS recipe_ingredients (
    recipe_rcp_seq TEXT NOT NULL,
    ingredient_name TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_recipe_ingredients_name ON recipe_ingredients(ingredient_name);
CREATE INDEX IF NOT EXISTS idx_recipe_ingredients_recipe ON recipe_ingredients(recipe_rcp_seq);
"""

def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def build_ingredient_index(conn: sqlite3.Connection) -> int:
    """(Re)build recipe_ingredients from every recipe's ingredients_raw.

    A lightweight normalized junction table rather than a graph DB — enough
    to answer "which recipes contain ALL of these ingredients" precisely,
    which semantic/embedding search can only approximate. Re-run this after
    ingesting or crawling new recipes.
    """
    ensure_schema(conn)
    conn.execute("DELETE FROM recipe_ingredients")

    rows = conn.execute("SELECT rcp_seq, name, ingredients_raw FROM recipes").fetchall()
    entries = [
        (rcp_seq, ingredient.name)
        for rcp_seq, name, ingredients_raw in rows
        for ingredient in parse_ingredients(ingredients_raw, recipe_name=name)
    ]
    conn.executemany(
        "INSERT INTO recipe_ingredients (recipe_rcp_seq, ingredient_name) VALUES (?, ?)",
        entries,
    )
    conn.commit()
    return len(entries)


def find_recipes_by_ingredients(
    conn: sqlite3.Connection, ingredient_terms: list[str], limit: int = 5
) -> list[Recipe]:
    """Recipes containing ALL of `ingredient_terms` (substring match against
    the parsed ingredient name), ranked by fewest total ingredients first —
    a recipe using only what was asked about ranks above one needing lots of
    other ingredients too.
    """
    terms = [t.strip() for t in ingredient_terms if t.strip()]
    if not terms:
        return []

    matched_sets = []
    for term in terms:
        rows = conn.execute(
            "SELECT DISTINCT recipe_rcp_seq FROM recipe_ingredients WHERE ingredient_name LIKE ?",
            (f"%{term}%",),
        ).fetchall()
        matched_sets.append({row[0] for row in rows})

    matching_ids = set.intersection(*matched_sets)
    if not matching_ids:
        return []

    placeholders = ",".join("?" for _ in matching_ids)
    ranked = conn.execute(
        f"""
        SELECT recipe_rcp_seq, COUNT(*) AS ingredient_count
        FROM recipe_ingredients
        WHERE recipe_rcp_seq IN ({placeholders})
        GROUP BY recipe_rcp_seq
        ORDER BY ingredient_count ASC
        LIMIT ?
        """,
        (*matching_ids, limit),
    ).fetchall()

    return get_recipes_by_ids(conn, [row[0] for row in ranked])
