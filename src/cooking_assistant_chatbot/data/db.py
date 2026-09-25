from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .chat_store import SCHEMA as CHAT_SCHEMA
from .models import Recipe

SCHEMA = """
CREATE TABLE IF NOT EXISTS recipes (
    rcp_seq TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    cooking_method TEXT,
    ingredients_raw TEXT,
    hash_tag TEXT,
    na_tip TEXT,
    main_image_url TEXT,
    thumbnail_image_url TEXT,
    weight_info TEXT,
    energy_kcal TEXT,
    carbohydrate_g TEXT,
    protein_g TEXT,
    fat_g TEXT,
    sodium_mg TEXT,
    steps_json TEXT NOT NULL,
    step_image_urls_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_recipes_name ON recipes(name);

CREATE TABLE IF NOT EXISTS price_cache (
    ingredient_name TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    price INTEGER NOT NULL,
    brand TEXT,
    link TEXT,
    fetched_at TEXT NOT NULL
);
"""


def get_connection(db_path: str) -> sqlite3.Connection:
    """Open a connection, usable from any thread.

    `check_same_thread=False` because the Gradio UI runs each chat turn in a
    worker thread from a pool, not the thread that created this connection.
    We don't do concurrent writes from multiple threads at once (single-user
    local app), so sqlite3's own file-level locking is enough here.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA + CHAT_SCHEMA)
    return conn


def row_to_recipe(row: sqlite3.Row) -> Recipe:
    return Recipe(
        rcp_seq=row["rcp_seq"],
        name=row["name"],
        category=row["category"] or "",
        cooking_method=row["cooking_method"] or "",
        ingredients_raw=row["ingredients_raw"] or "",
        hash_tag=row["hash_tag"] or "",
        na_tip=row["na_tip"] or "",
        main_image_url=row["main_image_url"] or "",
        thumbnail_image_url=row["thumbnail_image_url"] or "",
        weight_info=row["weight_info"] or "",
        energy_kcal=row["energy_kcal"] or "",
        carbohydrate_g=row["carbohydrate_g"] or "",
        protein_g=row["protein_g"] or "",
        fat_g=row["fat_g"] or "",
        sodium_mg=row["sodium_mg"] or "",
        steps=json.loads(row["steps_json"]),
        step_image_urls=json.loads(row["step_image_urls_json"]),
    )


def get_all_recipes(conn: sqlite3.Connection) -> list[Recipe]:
    rows = conn.execute("SELECT * FROM recipes").fetchall()
    return [row_to_recipe(row) for row in rows]


def get_recipes_by_ids(conn: sqlite3.Connection, rcp_seqs: list[str]) -> list[Recipe]:
    """Return recipes for the given ids, in the same order as `rcp_seqs`."""
    if not rcp_seqs:
        return []
    placeholders = ",".join("?" for _ in rcp_seqs)
    rows = conn.execute(
        f"SELECT * FROM recipes WHERE rcp_seq IN ({placeholders})", rcp_seqs
    ).fetchall()
    by_id = {row["rcp_seq"]: row_to_recipe(row) for row in rows}
    return [by_id[seq] for seq in rcp_seqs if seq in by_id]


def get_recipe_by_name(conn: sqlite3.Connection, name: str) -> Recipe | None:
    """Exact name match, ignoring spaces ("김치 찌개" == "김치찌개") — the
    source data itself is inconsistent about spacing in names."""
    row = conn.execute(
        "SELECT * FROM recipes WHERE REPLACE(name, ' ', '') = ? LIMIT 1",
        ("".join(name.split()),),
    ).fetchone()
    return row_to_recipe(row) if row else None


def upsert_recipes(conn: sqlite3.Connection, recipes: list[Recipe]) -> None:
    conn.executemany(
        """
        INSERT INTO recipes (
            rcp_seq, name, category, cooking_method, ingredients_raw,
            hash_tag, na_tip, main_image_url, thumbnail_image_url,
            weight_info, energy_kcal, carbohydrate_g, protein_g, fat_g,
            sodium_mg, steps_json, step_image_urls_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(rcp_seq) DO UPDATE SET
            name=excluded.name,
            category=excluded.category,
            cooking_method=excluded.cooking_method,
            ingredients_raw=excluded.ingredients_raw,
            hash_tag=excluded.hash_tag,
            na_tip=excluded.na_tip,
            main_image_url=excluded.main_image_url,
            thumbnail_image_url=excluded.thumbnail_image_url,
            weight_info=excluded.weight_info,
            energy_kcal=excluded.energy_kcal,
            carbohydrate_g=excluded.carbohydrate_g,
            protein_g=excluded.protein_g,
            fat_g=excluded.fat_g,
            sodium_mg=excluded.sodium_mg,
            steps_json=excluded.steps_json,
            step_image_urls_json=excluded.step_image_urls_json
        """,
        [
            (
                r.rcp_seq,
                r.name,
                r.category,
                r.cooking_method,
                r.ingredients_raw,
                r.hash_tag,
                r.na_tip,
                r.main_image_url,
                r.thumbnail_image_url,
                r.weight_info,
                r.energy_kcal,
                r.carbohydrate_g,
                r.protein_g,
                r.fat_g,
                r.sodium_mg,
                json.dumps(r.steps, ensure_ascii=False),
                json.dumps(r.step_image_urls, ensure_ascii=False),
            )
            for r in recipes
        ],
    )
    conn.commit()
