"""Glue between mic sources, STT, TTS and the chat turn.

One shared controller per app, not per browser session — this is a
single-user personal app, and the PC mic is a single physical device anyway.

Half-duplex by design: while a chat turn is being processed, and while its
spoken reply is playing, mic input is discarded. Otherwise the speakers feed
back into the mic and the bot starts answering itself.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Callable

import numpy as np

from .pc_mic import PcMicListener
from .stt import Transcriber
from .tts import MAX_SPEED, MIN_SPEED, Speaker
from .vad import UtteranceSegmenter, to_16k_mono

logger = logging.getLogger(__name__)

# Extra quiet time after the estimated end of TTS playback, to cover
# browser playback start-up delay and room echo.
PLAYBACK_MARGIN_SEC = 0.8


class VoiceController:
    def __init__(
        self,
        transcriber: Transcriber,
        speaker: Speaker,
        stt_model: str,
        executor: Executor | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._transcriber = transcriber
        self._speaker = speaker
        self.stt_model = stt_model
        self._executor = executor or ThreadPoolExecutor(max_workers=1)
        self._clock = clock
        self._inbox: queue.Queue[str] = queue.Queue()
        self._busy = False
        self._mute_until = 0.0
        self._browser_segmenter = UtteranceSegmenter()
        self.pc_mic = PcMicListener(self._submit_utterance, self.is_muted)

    # ---- warm-up (model loads take a few seconds; do them before first use) ----
    def preload_stt(self) -> None:
        self._executor.submit(self._transcriber.load, self.stt_model)

    def preload_tts(self) -> None:
        # Separate thread: the STT executor has one worker, and a first-run
        # TTS model download would otherwise hold up transcription.
        threading.Thread(target=self._speaker.load, daemon=True).start()

    # ---- half-duplex gating ----
    def is_muted(self) -> bool:
        return self._busy or self._clock() < self._mute_until

    def begin_turn(self) -> None:
        self._busy = True

    def end_turn(self, speech_seconds: float = 0.0) -> None:
        self._busy = False
        if speech_seconds > 0:
            self._mute_until = self._clock() + speech_seconds + PLAYBACK_MARGIN_SEC

    # ---- sources ----
    def feed_browser_audio(self, sample_rate: int, audio: np.ndarray) -> None:
        if self.is_muted():
            self._browser_segmenter.reset()
            return
        for utterance in self._browser_segmenter.feed(to_16k_mono(sample_rate, audio)):
            self._submit_utterance(utterance)

    def stop_browser_audio(self) -> None:
        utterances = [] if self.is_muted() else self._browser_segmenter.flush()
        self._browser_segmenter.reset()
        for utterance in utterances:
            self._submit_utterance(utterance)

    def _submit_utterance(self, audio: np.ndarray) -> None:
        logger.info("utterance %.1fs queued for STT", len(audio) / 16000)
        self._executor.submit(self._transcribe_into_inbox, audio, self.stt_model)

    def _transcribe_into_inbox(self, audio: np.ndarray, model_name: str) -> None:
        started = time.monotonic()
        text = self._transcriber.transcribe(audio, model_name)
        logger.info("STT %.1fs -> %r", time.monotonic() - started, text)
        if text:
            self._inbox.put(text)

    def pop_text(self) -> str | None:
        """Everything recognized since the last turn, joined into one message
        — a pause mid-sentence ("김치찌개... 해먹고 싶어") shouldn't become two
        separate chat turns."""
        parts = []
        while True:
            try:
                parts.append(self._inbox.get_nowait())
            except queue.Empty:
                break
        return " ".join(parts) if parts else None

    # ---- output ----
    def set_tts_speed(self, speed: float) -> None:
        self._speaker.speed = min(max(float(speed), MIN_SPEED), MAX_SPEED)

    def speak(self, text: str) -> tuple[tuple[int, np.ndarray] | None, float]:
        """(Gradio audio value, duration in seconds) for the reply's speech."""
        audio = self._speaker.synthesize(text)
        if audio is None:
            return None, 0.0
        sample_rate, samples = audio
        return audio, len(samples) / sample_rate
