import threading
import time

import pytest

from cooking_assistant_chatbot.progress import run_with_progress


def _drain(gen):
    statuses = []
    while True:
        try:
            statuses.append(next(gen))
        except StopIteration as done:
            return statuses, done.value


def test_yields_reported_statuses_in_order_and_returns_work_result():
    def work(report):
        report("하나")
        report("둘")
        return 42

    statuses, value = _drain(run_with_progress(work))

    assert statuses == ["하나", "둘"]
    assert value == 42


def test_re_yields_current_status_while_work_is_idle():
    release = threading.Event()

    def work(report):
        report("기다리는 중")
        release.wait(timeout=5)
        return "끝"

    gen = run_with_progress(work, tick_sec=0.05)
    first = next(gen)
    second = next(gen)  # no new report yet -> tick re-yields the same status
    release.set()
    rest, value = _drain(gen)

    assert first == second == "기다리는 중"
    assert value == "끝"


def test_exception_in_work_is_raised_to_the_caller():
    def work(report):
        report("시작")
        raise ValueError("boom")

    gen = run_with_progress(work)

    with pytest.raises(ValueError, match="boom"):
        _drain(gen)


def test_work_runs_off_the_calling_thread():
    caller = threading.get_ident()

    def work(report):
        time.sleep(0.01)
        return threading.get_ident()

    _, worker = _drain(run_with_progress(work))

    assert worker != caller
