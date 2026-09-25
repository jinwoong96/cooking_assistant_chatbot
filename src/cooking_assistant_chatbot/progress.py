from __future__ import annotations

import queue
import threading
from typing import Callable, Generator, TypeVar

T = TypeVar("T")

_DONE = object()


def run_with_progress(
    work: Callable[[Callable[[str], None]], T], tick_sec: float = 1.0
) -> Generator[str, None, T]:
    """Run `work(report)` on a background thread, yielding its latest status.

    Yields each status line `work` reports, and re-yields the current one
    every `tick_sec` while nothing new arrives, so the caller can keep an
    elapsed-time counter moving. The generator's return value (`yield from`
    / StopIteration.value) is `work`'s return value; exceptions from `work`
    re-raise here.
    """
    updates: queue.Queue = queue.Queue()
    outcome: dict = {}

    def target() -> None:
        try:
            outcome["value"] = work(updates.put)
        except BaseException as exc:  # re-raised on the caller's side
            outcome["error"] = exc
        finally:
            updates.put(_DONE)

    threading.Thread(target=target, daemon=True).start()

    status = ""
    while True:
        try:
            item = updates.get(timeout=tick_sec)
        except queue.Empty:
            yield status
            continue
        if item is _DONE:
            break
        status = item
        yield status

    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]
