"""Rule-based parsing of spoken cooking-mode commands and durations.

Deliberately not an LLM call: commands must answer instantly while hands
are busy. The user scoped cooking mode to step navigation + timers, so
anything else is "unknown" and gets a short hint instead of a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

CommandKind = Literal[
    "next", "prev", "repeat", "first", "timer", "timer_cancel", "timer_check", "exit", "unknown"
]


@dataclass
class Command:
    kind: CommandKind
    seconds: int | None = None
    """For "timer": the requested length, or None for "the step's own"."""


_SINO = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
_NATIVE = {
    "한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4,
    "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10,
    "스무": 20, "스물": 20, "서른": 30, "마흔": 40, "쉰": 50,
}
_UNIT_SECONDS = {"초": 1, "분": 60, "시간": 3600}

# Longest alternatives first so "다섯" wins over "다", "스무" over "스".
_NUMBER_WORD = "|".join(sorted([*_NATIVE, "십", *_SINO], key=len, reverse=True))
_DURATION_RE = re.compile(
    rf"(\d+(?:\.\d+)?|(?:{_NUMBER_WORD})+)\s*(시간|분|초)(?:\s*반)?"
)
_DIGIT_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(시간|분|초)(?:\s*반)?")
_RANGE_RE = re.compile(r"(\d+)\s*~\s*\d+\s*(시간|분|초)")


def _korean_number(word: str) -> int | None:
    """Sino-Korean up to 99 ("삼십오") or a native count ("다섯", "열", "스무")."""
    if word in _NATIVE:
        return _NATIVE[word]
    if word.startswith("열") and word[1:] in _NATIVE:  # 열다섯
        return 10 + _NATIVE[word[1:]]
    if word.startswith(("스무", "스물")) and word[2:] in _NATIVE:
        return 20 + _NATIVE[word[2:]]
    if "십" in word:
        tens, _, ones = word.partition("십")
        if (tens and tens not in _SINO) or (ones and ones not in _SINO):
            return None
        return (_SINO[tens] if tens else 1) * 10 + (_SINO[ones] if ones else 0)
    return _SINO.get(word)


def parse_duration(text: str, spoken: bool = True) -> int | None:
    """First duration in `text`, in seconds ("5분" -> 300, "1시간 반" ->
    5400). A range ("1~2분", "20~30초") gives its low end — better to check
    early than overcook.

    `spoken` also accepts Korean number words ("오 분" -> 300), which STT
    sometimes produces. Off for recipe text, where they'd misfire on
    ordinary words ("이 시간 동안" is not "2 hours")."""
    compact = text.replace(" ", "")
    range_match = _RANGE_RE.search(compact)
    pattern = _DURATION_RE if spoken else _DIGIT_DURATION_RE
    matches = list(pattern.finditer(compact))
    if range_match and (not matches or range_match.start() <= matches[0].start()):
        return int(range_match.group(1)) * _UNIT_SECONDS[range_match.group(2)]
    # Spoken requests add up ("1분 30초" is 90s); recipe text takes its
    # first time only (a step can mention several separate durations).
    total = 0.0
    for match in matches if spoken else matches[:1]:
        raw, unit = match.group(1), match.group(2)
        value = float(raw) if raw[0].isdigit() else _korean_number(raw)
        if value is None:
            continue
        total += value * _UNIT_SECONDS[unit]
        if match.group(0).endswith("반"):
            total += _UNIT_SECONDS[unit] / 2
    return int(total) if total > 0 else None


def _has(text: str, *words: str) -> bool:
    return any(w in text for w in words)


def parse_command(text: str) -> Command:
    t = text.replace(" ", "").rstrip(".!?~")
    if _has(t, "타이머", "알람"):
        if _has(t, "취소", "꺼", "멈춰", "그만", "중지", "없애"):
            return Command("timer_cancel")
        if _has(t, "얼마", "남았", "몇분"):
            return Command("timer_check")
        return Command("timer", parse_duration(t))
    if _has(t, "얼마남았", "몇분남았"):
        return Command("timer_check")
    if _has(t, "요리끝", "요리종료", "요리모드종료", "종료", "그만할래", "끝낼래", "끝내"):
        return Command("exit")
    if _has(t, "처음부터", "첫단계", "맨처음"):
        return Command("first")
    if _has(t, "이전", "전단계", "앞단계", "뒤로", "아까단계"):
        return Command("prev")
    if _has(t, "다시", "한번더", "뭐라고", "다시말해", "못들었"):
        return Command("repeat")
    if _has(t, "다음", "넘어가", "했어", "다했", "됐어", "완료", "넥스트"):
        return Command("next")
    # A bare duration ("5분") is almost certainly a timer request.
    seconds = parse_duration(t)
    if seconds is not None and len(t) <= 8:
        return Command("timer", seconds)
    return Command("unknown")
