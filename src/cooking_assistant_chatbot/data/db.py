from __future__ import annotations

import json
import sqlite3
from pathlib import Path

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
"""


def get_connection(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


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
