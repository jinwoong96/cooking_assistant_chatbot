"""Server-side microphone capture (the PC the app runs on), via sounddevice.

The audio callback only enqueues frames; segmentation runs on a worker
thread — same split as the reference project, so the callback never blocks.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .vad import FRAME_SAMPLES, SAMPLE_RATE, UtteranceSegmenter


@dataclass
class InputDevice:
    index: int
    name: str

    @property
    def label(self) -> str:
        return f"[{self.index}] {self.name}"


def list_input_devices() -> list[InputDevice]:
    import sounddevice as sd

    try:
        return [
            InputDevice(index, str(info["name"]))
            for index, info in enumerate(sd.query_devices())
            if int(info.get("max_input_channels", 0)) > 0
        ]
    except Exception:
        return []


class PcMicListener:
    def __init__(
        self,
        on_utterance: Callable[[np.ndarray], None],
        is_muted: Callable[[], bool],
    ) -> None:
        self._on_utterance = on_utterance
        self._is_muted = is_muted
        self._frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=512)
        self._stream = None
        self._worker: threading.Thread | None = None
        self._segmenter = UtteranceSegmenter()

    @property
    def running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def start(self, device_index: int | None) -> None:
        if self.running:
            return
        import sounddevice as sd

        self._segmenter.reset()
        while not self._frames.empty():
            self._frames.get_nowait()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            device=device_index,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._frames.put(None)
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None

    def _audio_callback(self, indata, _frames, _time, _status) -> None:
        try:
            self._frames.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # dropping a frame or two isn't fatal

    def _loop(self) -> None:
        while True:
            frame = self._frames.get()
            if frame is None:
                break
            if self._is_muted():
                self._segmenter.reset()
                continue
            for utterance in self._segmenter.feed(frame):
                self._on_utterance(utterance)
        if not self._is_muted():
            for utterance in self._segmenter.flush():
                self._on_utterance(utterance)
