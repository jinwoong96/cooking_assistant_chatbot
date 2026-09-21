from __future__ import annotations

import json
import zlib

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
        # zlib.crc32, not hash(): Python randomizes str hash() per-process
        # (PYTHONHASHSEED), which made this "deterministic" fake flaky
        # across separate test runs. Plain ord() % _DIM is stable but collides
        # badly across Korean's contiguous syllable block (every char exactly
        # _DIM codepoints apart lands in the same bucket); crc32 gives a
        # stable *and* well-distributed bucket instead.
        vector = [0.0] * _DIM
        for char in text:
            vector[zlib.crc32(char.encode("utf-8")) % _DIM] += 1.0
        norm = sum(v * v for v in vector) ** 0.5
        if norm == 0:
            return vector
        return [v / norm for v in vector]


class _FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name: str, arguments: str):
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content: str | None = None, tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message: _FakeMessage):
        self.message = message


class FakeLLMResponse:
    """Stand-in for a litellm.completion() result, shaped just enough to
    match what our code reads off it (response.choices[0].message.*)."""

    def __init__(self, message: _FakeMessage):
        self.choices = [_FakeChoice(message)]


def fake_tool_call_response(name: str, arguments: dict) -> FakeLLMResponse:
    return FakeLLMResponse(
        _FakeMessage(tool_calls=[_FakeToolCall(name, json.dumps(arguments))])
    )


def fake_text_response(content: str) -> FakeLLMResponse:
    return FakeLLMResponse(_FakeMessage(content=content, tool_calls=None))
