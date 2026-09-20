from __future__ import annotations

import sqlite3

import chromadb
from chromadb.api.types import EmbeddingFunction

from ..data.db import get_recipes_by_ids
from ..data.models import Recipe
from .indexer import COLLECTION_NAME


class RecipeSearcher:
    """Semantic search over the indexed recipe collection.

    Chroma only stores ids/embeddings/metadata; the full Recipe (steps,
    nutrition, etc.) is looked up from SQLite by id after the vector search
    narrows down the candidates.
    """

    def __init__(
        self,
        chroma_path: str,
        sqlite_path: str,
        embedding_function: EmbeddingFunction,
    ):
        self._sqlite_path = sqlite_path
        client = chromadb.PersistentClient(path=chroma_path)
        self._collection = client.get_collection(
            COLLECTION_NAME, embedding_function=embedding_function
        )

    def search(self, query: str, top_k: int = 5) -> list[Recipe]:
        result = self._collection.query(query_texts=[query], n_results=top_k)
        ids = result["ids"][0] if result["ids"] else []
        if not ids:
            return []

        conn = sqlite3.connect(self._sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            return get_recipes_by_ids(conn, ids)
        finally:
            conn.close()
