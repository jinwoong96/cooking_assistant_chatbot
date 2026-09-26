"""Local Korean TTS via Supertonic (ONNX, CPU-only).

Chosen because it uses no VRAM (qwen3:14b already takes ~9.6 GB of the 16 GB
on this PC) and needs no CUDA. Measured here: a ~13 s Korean sentence takes
~0.6 s to synthesize with supertonic-2, ~2 s with supertonic-3.
"""
from __future__ import annotations

import re
import threading

import numpy as np

_MARKDOWN_PATTERNS = [
    (re.compile(r"```.*?```", re.S), " "),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), " "),
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),
    (re.compile(r"https?://\S+"), " "),
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""),
    (re.compile(r"^\s*[-*+]\s+", re.M), ""),
    (re.compile(r"^\s*\d+[.)]\s+", re.M), ""),
    (re.compile(r"^\s*[-*_]{3,}\s*$", re.M), " "),
    (re.compile(r"\*{1,3}|_{2,3}|~~"), ""),
    (re.compile(r"\|"), " "),
    (re.compile(r"[\U0001F000-\U0001FAFF☀-➿️]"), ""),
]


def to_speech_text(markdown: str) -> str:
    """Strip markdown/emoji/URLs so the TTS doesn't read symbols aloud."""
    text = markdown
    for pattern, replacement in _MARKDOWN_PATTERNS:
        text = pattern.sub(replacement, text)
    lines = [re.sub(r"[ \t]{2,}", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


MIN_SPEED = 0.8
MAX_SPEED = 1.6
"""UI slider range. Measured on a 7.5s sentence (at 1.05): 0.8 -> 9.9s,
1.6 -> 4.9s. Past ~1.6 Korean gets hard to follow while cooking."""


class Speaker:
    """Lazy-loaded Supertonic model + one voice style."""

    def __init__(self, model_name: str, voice: str, speed: float = 1.05) -> None:
        self._model_name = model_name
        self._voice = voice
        self.speed = speed
        """Speaking rate passed to Supertonic; changed live from the UI.
        Shared by every browser session (single-user app)."""
        self._tts = None
        self._style = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._tts is not None:
                return
            from supertonic import TTS

            self._tts = TTS(model=self._model_name)
            self._style = self._tts.get_voice_style(self._voice)

    def synthesize(self, text: str) -> tuple[int, np.ndarray] | None:
        """Returns (sample_rate, int16 mono samples), or None for empty text."""
        text = text.strip()
        if not text:
            return None
        self.load()
        with self._lock:
            wav, _durations = self._tts.synthesize(
                text, voice_style=self._style, lang="ko", speed=self.speed
            )
            sample_rate = self._tts.sample_rate
        samples = np.clip(np.asarray(wav, dtype=np.float32).reshape(-1), -1.0, 1.0)
        return sample_rate, (samples * 32767).astype(np.int16)
