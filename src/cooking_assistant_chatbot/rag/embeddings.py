from __future__ import annotations

import threading
from typing import Sequence

from chromadb.api.types import Documents, EmbeddingFunction


class BGEEmbeddingFunction(EmbeddingFunction[Documents]):
    """Chroma-compatible embedding function backed by a local sentence-transformers model.

    Defaults to BAAI/bge-m3, chosen for good Korean retrieval quality while
    running fully locally (no cloud API, no cost). The model is loaded lazily
    so importing this module doesn't trigger a multi-GB download/model load.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._load_lock = threading.Lock()

    def _get_model(self):
        # Locked: the app preloads this on a background thread at startup, and
        # a first request arriving mid-load must wait rather than load it twice.
        with self._load_lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    def load(self) -> None:
        """Load the model now instead of on the first search. On first use
        this took ~40s here — the bulk of a cold first reply."""
        self._get_model()

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        embeddings = self._get_model().encode(list(input), normalize_embeddings=True)
        return embeddings.tolist()

    def name(self) -> str:
        return f"bge-embedding-function-{self._model_name}"

    def get_config(self) -> dict:
        return {"model_name": self._model_name, "device": self._device}

    @staticmethod
    def build_from_config(config: dict) -> "BGEEmbeddingFunction":
        return BGEEmbeddingFunction(
            model_name=config.get("model_name", "BAAI/bge-m3"),
            device=config.get("device"),
        )
