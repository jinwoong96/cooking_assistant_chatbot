"""Energy-based utterance segmentation, source-agnostic.

Ported from the speech_to_text reference project's MicListener, minus the
audio-device plumbing: callers push raw 16 kHz mono float32 audio in, and
get back each finished utterance once the speaker pauses. Both the PC-mic
listener and the browser-mic stream feed the same segmenter.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

# Tuned values from the reference project's settings.json on this PC.
SENSITIVITY_DB = 4.0          # how many dB above the noise floor counts as speech
SILENCE_HOLD_SEC = 0.8        # this much silence ends an utterance
MAX_SEGMENT_SEC = 20.0        # force a cut if someone talks without pausing
MIN_SEGMENT_SEC = 0.35        # shorter than this is treated as a click/noise
ABSOLUTE_FLOOR_DB = -55.0     # never count anything quieter than this as speech
PREROLL_SEC = 0.4             # keep audio from just before onset so the first syllable isn't cut
SPEECH_ONSET_FRAMES = 3       # 90 ms of consecutive loud frames to start an utterance


def dbfs(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)) + 1e-12))
    return 20.0 * math.log10(max(rms, 1e-7))


class UtteranceSegmenter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Drop any half-finished utterance and forget the noise estimate —
        used when listening is muted (e.g. while the bot's reply is playing)
        so speaker audio doesn't bleed into the next utterance."""
        self._pending = np.zeros(0, dtype=np.float32)
        self._noise_floor_db = ABSOLUTE_FLOOR_DB
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, int(PREROLL_SEC * 1000 / FRAME_MS)))
        self._buffer: list[np.ndarray] = []
        self._speaking = False
        self._onset_run = 0
        self._silence_run = 0
        self._speech_frames = 0

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        """Push any amount of audio; returns utterances completed by it."""
        audio = np.concatenate([self._pending, samples.astype(np.float32, copy=False)])
        whole = len(audio) // FRAME_SAMPLES * FRAME_SAMPLES
        self._pending = audio[whole:]
        finished: list[np.ndarray] = []
        for start in range(0, whole, FRAME_SAMPLES):
            utterance = self._feed_frame(audio[start : start + FRAME_SAMPLES])
            if utterance is not None:
                finished.append(utterance)
        return finished

    def flush(self) -> list[np.ndarray]:
        """Emit whatever is buffered — called when listening is stopped."""
        utterance = self._emit(trim_tail=0)
        self.reset()
        return [utterance] if utterance is not None else []

    def _feed_frame(self, frame: np.ndarray) -> np.ndarray | None:
        level = dbfs(frame)
        threshold = max(self._noise_floor_db + SENSITIVITY_DB, ABSOLUTE_FLOOR_DB)
        is_speech = level > threshold

        if not is_speech:
            # Adapt the noise floor only on quiet frames.
            self._noise_floor_db = max(0.95 * self._noise_floor_db + 0.05 * level, -80.0)

        if not self._speaking:
            self._preroll.append(frame)
            self._onset_run = self._onset_run + 1 if is_speech else 0
            if self._onset_run >= SPEECH_ONSET_FRAMES:
                self._speaking = True
                self._silence_run = 0
                self._buffer = list(self._preroll)
                self._speech_frames = self._onset_run
                self._preroll.clear()
            return None

        self._buffer.append(frame)
        if is_speech:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1

        if self._silence_run * FRAME_MS / 1000 >= SILENCE_HOLD_SEC:
            utterance = self._emit(trim_tail=self._silence_run)
            self._buffer, self._speaking = [], False
            self._onset_run = self._silence_run = self._speech_frames = 0
            return utterance
        if len(self._buffer) * FRAME_MS / 1000 >= MAX_SEGMENT_SEC:
            # Still talking — cut here but stay in the speaking state.
            utterance = self._emit(trim_tail=0)
            self._buffer, self._silence_run, self._speech_frames = [], 0, 0
            return utterance
        return None

    def _emit(self, trim_tail: int) -> np.ndarray | None:
        if not self._buffer:
            return None
        # Judge length by frames actually classed as speech, not the whole
        # buffer — preroll + trailing silence would make a click look long.
        if self._speech_frames * FRAME_MS / 1000 < MIN_SEGMENT_SEC:
            return None
        # Keep half the trailing silence; a little tail helps recognition.
        keep = len(self._buffer) - trim_tail // 2
        chunk = self._buffer[:keep] if keep > 0 else self._buffer
        return np.concatenate(chunk).astype(np.float32, copy=False)


def to_16k_mono(sample_rate: int, audio: np.ndarray) -> np.ndarray:
    """Normalize whatever a mic source hands us (int16 or float, mono or
    stereo, any rate — browsers usually send 48 kHz) to 16 kHz mono float32.
    Linear interpolation is plenty for speech recognition."""
    data = np.asarray(audio)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float32, copy=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sample_rate == SAMPLE_RATE or len(data) == 0:
        return data
    target_len = int(round(len(data) * SAMPLE_RATE / sample_rate))
    positions = np.linspace(0, len(data) - 1, num=target_len)
    return np.interp(positions, np.arange(len(data)), data).astype(np.float32)
