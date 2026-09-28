import pytest

from cooking_assistant_chatbot.cooking import session as cooking
from cooking_assistant_chatbot.cooking.commands import parse_command, parse_duration
from cooking_assistant_chatbot.data.models import Recipe


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("5분", 300),
        ("타이머 10분", 600),
        ("30초", 30),
        ("1시간 반", 5400),
        ("1~2분정도 저어준다", 60),  # a range gives its low end
        ("20~30초간 둔다", 20),
        ("오 분", 300),
        ("삼십 초", 30),
        ("다섯 분", 300),
        ("십오분", 900),
        ("그냥 끓인다", None),
    ],
)
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


def test_recipe_text_does_not_read_korean_words_as_numbers():
    # "이 시간" is "this time", not 2 hours.
    assert parse_duration("이 시간 동안 소스를 만든다", spoken=False) is None
    assert parse_duration("약 20분 정도 더 쪄낸다", spoken=False) == 1200


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("다음", "next"),
        ("다음 단계", "next"),
        ("넘어가자", "next"),
        ("다 했어", "next"),
        ("이전 단계", "prev"),
        ("뒤로", "prev"),
        ("다시 말해줘", "repeat"),
        ("뭐라고?", "repeat"),
        ("처음부터", "first"),
        ("타이머", "timer"),
        ("타이머 꺼줘", "timer_cancel"),
        ("타이머 얼마 남았어?", "timer_check"),
        ("요리 끝", "exit"),
        ("오늘 날씨 어때", "unknown"),
    ],
)
def test_parse_command(text, kind):
    assert parse_command(text).kind == kind


def test_bare_duration_is_a_timer_request():
    command = parse_command("7분")

    assert command.kind == "timer" and command.seconds == 420


def _recipe():
    return Recipe(
        rcp_seq="1",
        name="라면",
        steps=["1. 물 550ml를\n끓인다.", "2. 면과 스프를 넣고 4분간 끓인다. ①참고", "3. 그릇에 담는다ㅋㅋ"],
    )


def test_start_reads_intro_and_first_step_cleaned():
    session, speech = cooking.start(_recipe())

    assert session.steps[0] == "물 550ml를 끓인다."
    assert session.steps[2] == "그릇에 담는다"
    assert "모두 3단계" in speech
    assert "1단계. 물 550ml를 끓인다." in speech


def test_start_without_steps():
    session, speech = cooking.start(Recipe(rcp_seq="1", name="빈 레시피"))

    assert session is None
    assert "조리 순서가 없어서" in speech


def test_navigation_moves_and_stops_at_the_ends():
    session, _ = cooking.start(_recipe())

    session, speech = cooking.handle(session, "이전", now=0)
    assert session.index == 0 and speech.startswith("첫 단계예요.")

    session, speech = cooking.handle(session, "다음", now=0)
    assert session.index == 1
    assert "4분이에요" in speech  # the step's own time is offered as a timer

    session, _ = cooking.handle(session, "다음", now=0)
    session, speech = cooking.handle(session, "다음", now=0)
    assert session.index == 2 and "요리 끝" in speech

    session, _ = cooking.handle(session, "처음부터", now=0)
    assert session.index == 0


def test_timer_uses_the_steps_own_time_and_rings_once():
    session, _ = cooking.start(_recipe())
    session, _ = cooking.handle(session, "다음", now=0)

    session, speech = cooking.handle(session, "타이머", now=100)
    assert speech == "4분 타이머를 시작했어요."
    assert session.timer_remaining(now=160) == 180

    session, alarm = cooking.check_timer(session, now=200)
    assert alarm is None
    session, alarm = cooking.check_timer(session, now=341)
    assert alarm.startswith("4분 타이머가 끝났어요!")
    session, alarm = cooking.check_timer(session, now=400)
    assert alarm is None


def test_timer_without_a_time_asks_for_one():
    session, _ = cooking.start(_recipe())

    session, speech = cooking.handle(session, "타이머", now=0)

    assert "몇 분" in speech and session.timer_deadline is None


def test_timer_check_and_cancel():
    session, _ = cooking.start(_recipe())
    session, _ = cooking.handle(session, "타이머 1분 30초", now=0)

    assert cooking.handle(session, "타이머 얼마 남았어", now=30)[1] == "1분 남았어요."
    session, speech = cooking.handle(session, "타이머 꺼", now=31)
    assert speech == "타이머를 껐어요." and session.timer_deadline is None


def test_exit_and_unknown():
    session, _ = cooking.start(_recipe())

    assert "잘 못 알아들었어요" in cooking.handle(session, "배고프다", now=0)[1]
    ended, speech = cooking.handle(session, "요리 끝", now=0)
    assert ended is None and "마칠게요" in speech


def test_render_shows_step_and_countdown():
    session, _ = cooking.start(_recipe())
    session, _ = cooking.handle(session, "타이머 2분", now=0)

    step, timer = cooking.render(session, now=5)

    assert "1 / 3단계" in step and "물 550ml를 끓인다." in step
    assert "01:55" in timer
    assert cooking.render(None, now=0) == ("", "")


def test_spoken_compound_durations_add_up():
    assert parse_duration("타이머 1분 30초") == 90
    assert parse_duration("1시간 10분") == 4200
    # Recipe text: separate durations in one step, keep the first.
    assert parse_duration("60분 간 1차 발효 후 15분 간 중간 발효", spoken=False) == 3600


def test_render_escapes_tildes_so_they_are_not_strikethrough():
    recipe = Recipe(rcp_seq="1", name="김치볶음밥", steps=["양념 (67g) 설탕 1스푼 (10g) ~ 참기름 2스푼 ~ 섞는다"])
    session, _ = cooking.start(recipe)

    step, _ = cooking.render(session, now=0)

    assert r"\~ 참기름 2스푼 \~" in step
