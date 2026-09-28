"""Cooking-mode state machine: which step we're on, the running timer, and
what to say for each command. Pure (no Gradio, no audio) so it's testable;
`app.py` holds a `CookingSession` per browser session and speaks the text
this returns."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from ..data.models import Recipe
from ..voice.tts import to_speech_text
from .commands import Command, parse_command, parse_duration

_LEADING_NUMBER_RE = re.compile(r"^\s*\d+\s*[.)]\s*")
_CIRCLED_RE = re.compile(r"[①-⑳]")  # ①..⑳ "see step ①" references
_CHAT_NOISE_RE = re.compile(r"ㅋ+|ㅎ+|ㅠ+|ㅜ+|\^\^|~{2,}")
_MARKDOWN_SPECIAL_RE = re.compile(r"([\\`*_~\[\]<>])")

HELP_TEXT = "'다음', '이전', '다시', '타이머 5분', '요리 끝' 이라고 말해주세요."


def clean_step(text: str) -> str:
    """A recipe step as it should be shown and spoken: no leading "3.",
    line breaks from the source's fixed-width wrapping, circled-number
    cross references, or chat-style ㅋㅋ/^^ from the crawled recipes."""
    text = _LEADING_NUMBER_RE.sub("", text)
    text = _CIRCLED_RE.sub("", text)
    text = _CHAT_NOISE_RE.sub("", text)
    return " ".join(text.split())


def format_duration(seconds: int) -> str:
    hours, rest = divmod(max(seconds, 0), 3600)
    minutes, secs = divmod(rest, 60)
    parts = []
    if hours:
        parts.append(f"{hours}시간")
    if minutes:
        parts.append(f"{minutes}분")
    if secs or not parts:
        parts.append(f"{secs}초")
    return " ".join(parts)


@dataclass(frozen=True)
class CookingSession:
    recipe_name: str
    steps: tuple[str, ...]
    index: int = 0
    timer_deadline: float | None = None
    timer_seconds: int = 0

    @property
    def step(self) -> str:
        return self.steps[self.index]

    @property
    def is_last(self) -> bool:
        return self.index == len(self.steps) - 1

    @property
    def step_duration(self) -> int | None:
        """A time stated in the current step ("5분간 끓인다"), if any."""
        return parse_duration(self.step, spoken=False)

    def timer_remaining(self, now: float) -> int | None:
        if self.timer_deadline is None:
            return None
        return max(0, round(self.timer_deadline - now))


def start(recipe: Recipe) -> tuple[CookingSession | None, str]:
    steps = tuple(s for s in (clean_step(s) for s in recipe.steps) if s)
    if not steps:
        return None, f"{recipe.name}에는 조리 순서가 없어서 요리 모드를 시작할 수 없어요."
    session = CookingSession(recipe_name=recipe.name, steps=steps)
    intro = f"{recipe.name} 요리를 시작할게요. 모두 {len(steps)}단계예요. {HELP_TEXT}"
    return session, f"{intro} {step_speech(session)}"


def step_speech(session: CookingSession) -> str:
    text = f"{session.index + 1}단계. {session.step}"
    duration = session.step_duration
    if duration:
        text += f" 이 단계는 {format_duration(duration)}이에요. '타이머'라고 하면 맞춰드릴게요."
    if session.is_last:
        text += " 마지막 단계예요."
    return to_speech_text(text)


def _set_timer(session: CookingSession, seconds: int, now: float) -> tuple[CookingSession, str]:
    session = replace(session, timer_deadline=now + seconds, timer_seconds=seconds)
    return session, f"{format_duration(seconds)} 타이머를 시작했어요."


def handle(session: CookingSession, text: str, now: float) -> tuple[CookingSession | None, str]:
    """Apply a spoken command. Returns the new session (None = cooking mode
    ended) and what to say back."""
    command: Command = parse_command(text)

    if command.kind == "next":
        if session.is_last:
            return session, "마지막 단계예요. 요리를 마치려면 '요리 끝'이라고 말해주세요."
        session = replace(session, index=session.index + 1)
        return session, step_speech(session)
    if command.kind == "prev":
        if session.index == 0:
            return session, "첫 단계예요. " + step_speech(session)
        session = replace(session, index=session.index - 1)
        return session, step_speech(session)
    if command.kind == "repeat":
        return session, step_speech(session)
    if command.kind == "first":
        session = replace(session, index=0)
        return session, step_speech(session)
    if command.kind == "timer":
        seconds = command.seconds or session.step_duration
        if not seconds:
            return session, "몇 분으로 맞출까요? '타이머 5분'처럼 말해주세요."
        return _set_timer(session, seconds, now)
    if command.kind == "timer_cancel":
        if session.timer_deadline is None:
            return session, "켜진 타이머가 없어요."
        return replace(session, timer_deadline=None, timer_seconds=0), "타이머를 껐어요."
    if command.kind == "timer_check":
        remaining = session.timer_remaining(now)
        if remaining is None:
            return session, "켜진 타이머가 없어요."
        return session, f"{format_duration(remaining)} 남았어요."
    if command.kind == "exit":
        return None, "요리 모드를 마칠게요. 맛있게 드세요!"
    return session, f"잘 못 알아들었어요. {HELP_TEXT}"


def check_timer(session: CookingSession, now: float) -> tuple[CookingSession, str | None]:
    """When the running timer has run out: clear it and say so."""
    remaining = session.timer_remaining(now)
    if remaining is None or remaining > 0:
        return session, None
    done = replace(session, timer_deadline=None, timer_seconds=0)
    return done, f"{format_duration(session.timer_seconds)} 타이머가 끝났어요! {session.index + 1}단계 확인해보세요."


def escape_markdown(text: str) -> str:
    """Recipe text as literal markdown. Crawled steps use "~" freely ("(67g)
    ~ 설탕 1스푼 ~"), and the panel rendered the span between two of them
    as strikethrough."""
    return _MARKDOWN_SPECIAL_RE.sub(r"\\\1", text)


def render(session: CookingSession | None, now: float) -> tuple[str, str]:
    """(step panel markdown, timer markdown) for the cooking-mode UI."""
    if session is None:
        return "", ""
    step = (
        f"### 🍳 {escape_markdown(session.recipe_name)}\n"
        f"**{session.index + 1} / {len(session.steps)}단계**\n\n"
        f"## {escape_markdown(session.step)}"
    )
    remaining = session.timer_remaining(now)
    if remaining is None:
        timer = f"⏱ 타이머 없음 — {HELP_TEXT}"
    else:
        minutes, secs = divmod(remaining, 60)
        timer = f"## ⏱ {minutes:02d}:{secs:02d}"
    return step, timer
