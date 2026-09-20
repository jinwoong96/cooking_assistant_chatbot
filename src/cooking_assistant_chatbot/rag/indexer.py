from __future__ import annotations

import sqlite3

import chromadb
from chromadb.api.types import EmbeddingFunction

from ..data.models import Recipe

COLLECTION_NAME = "recipes"


def recipe_document_text(recipe: Recipe) -> str:
    parts = [recipe.name]
    if recipe.category:
        parts.append(f"카테고리: {recipe.category}")
    if recipe.ingredients_raw:
        parts.append(f"재료: {recipe.ingredients_raw}")
    if recipe.hash_tag:
        parts.append(f"해시태그: {recipe.hash_tag}")
    return "\n".join(parts)


def build_index(
    recipes: list[Recipe],
    chroma_path: str,
    embedding_function: EmbeddingFunction,
    batch_size: int = 64,
) -> int:
    """(Re)build the recipe collection from scratch and return how many recipes were indexed."""
    client = chromadb.PersistentClient(path=chroma_path)
    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        COLLECTION_NAME, embedding_function=embedding_function
    )

    count = 0
    for i in range(0, len(recipes), batch_size):
        batch = recipes[i : i + batch_size]
        collection.upsert(
            ids=[r.rcp_seq for r in batch],
            documents=[recipe_document_text(r) for r in batch],
            metadatas=[{"name": r.name, "category": r.category} for r in batch],
        )
        count += len(batch)
    return count


def build_index_from_sqlite(
    sqlite_path: str,
    chroma_path: str,
    embedding_function: EmbeddingFunction,
    batch_size: int = 64,
) -> int:
    from ..data.db import get_all_recipes

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        recipes = get_all_recipes(conn)
    finally:
        conn.close()
    return build_index(recipes, chroma_path, embedding_function, batch_size=batch_size)
