from __future__ import annotations

from chromadb.api.types import Documents, EmbeddingFunction

_DIM = 64


class FakeEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deterministic, dependency-free stand-in for BGEEmbeddingFunction in tests.

    Hashes each character into one of `_DIM` buckets and counts occurrences,
    so texts that share more characters end up closer in embedding space.
    Good enough to test that the indexing/search *pipeline* behaves
    correctly, without downloading the real (multi-GB) embedding model.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, input):
        return [self._embed_one(text) for text in input]

    def name(self) -> str:
        return "fake-char-hash-embedding"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "FakeEmbeddingFunction":
        return FakeEmbeddingFunction()

    @staticmethod
    def _embed_one(text: str) -> list[float]:
        vector = [0.0] * _DIM
        for char in text:
            vector[hash(char) % _DIM] += 1.0
        norm = sum(v * v for v in vector) ** 0.5
        if norm == 0:
            return vector
        return [v / norm for v in vector]
