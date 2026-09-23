"""faster-whisper wrapper, ported from the speech_to_text reference project.

CPU int8 only: the AMD RX 9060 XT has no CUDA, and CTranslate2's Windows
ROCm builds crash on RDNA4 (CT2 issues #2016/#2021), so there's no usable
GPU path for faster-whisper here. Measured latency on this PC (reference
project, Ryzen 5 7600), time from end of speech to text: small 1.1s,
medium 2.8s, large-v3-turbo 4.3s — roughly independent of utterance length,
since Whisper always encodes a 30 s window.
"""
from __future__ import annotations

import re
import threading

import numpy as np

STT_MODELS = ("small", "medium", "large-v3-turbo")

# Phrases Whisper habitually emits on silence/noise; in continuous listening
# these would otherwise get sent to the chatbot as if the user said them.
_HALLUCINATIONS = (
    "시청해주셔서 감사합니다",
    "시청해 주셔서 감사합니다",
    "구독과 좋아요",
    "구독 좋아요 부탁드립니다",
    "다음 영상에서 만나요",
    "한글자막 by",
    "자막 제공",
    "mbc 뉴스",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "subtitles by",
    "amara.org",
)
_REPEAT_RE = re.compile(r"^(.)\1{3,}$")


def is_noise_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if any(pattern in stripped.lower() for pattern in _HALLUCINATIONS):
        return True
    compact = re.sub(r"[\s.,!?·…]", "", stripped)
    return not compact or bool(_REPEAT_RE.match(compact))


class Transcriber:
    """Holds one Whisper model; swaps it when a different one is requested."""

    def __init__(self, model_dir: str, cpu_threads: int = 6) -> None:
        self._model_dir = model_dir
        self._cpu_threads = cpu_threads
        self._model = None
        self._model_name: str | None = None
        self._lock = threading.Lock()

    def load(self, model_name: str) -> None:
        if model_name not in STT_MODELS:
            raise ValueError(f"Unknown STT model: {model_name}")
        with self._lock:
            if self._model_name == model_name:
                return
            from faster_whisper import WhisperModel

            self._model = None  # free the old one before loading the next
            self._model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
                cpu_threads=self._cpu_threads,
                download_root=self._model_dir,
            )
            self._model_name = model_name

    def transcribe(self, audio: np.ndarray, model_name: str) -> str:
        """16 kHz mono float32 in, Korean text out ("" for noise/silence)."""
        self.load(model_name)
        with self._lock:
            segments, _info = self._model.transcribe(
                audio,
                language="ko",
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                condition_on_previous_text=False,
                temperature=[0.0, 0.2, 0.4],
                no_speech_threshold=0.6,
            )
            text = " ".join(segment.text.strip() for segment in segments)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return "" if is_noise_text(text) else text
