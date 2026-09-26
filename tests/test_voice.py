import numpy as np

from cooking_assistant_chatbot.voice.controller import PLAYBACK_MARGIN_SEC, VoiceController
from cooking_assistant_chatbot.voice.stt import is_noise_text
from cooking_assistant_chatbot.voice.tts import to_speech_text
from cooking_assistant_chatbot.voice.vad import SAMPLE_RATE, UtteranceSegmenter, to_16k_mono

_rng = np.random.default_rng(0)


def _silence(seconds: float) -> np.ndarray:
    return (_rng.normal(0, 0.0005, int(SAMPLE_RATE * seconds))).astype(np.float32)


def _tone(seconds: float) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# ---- VAD ----

def test_segmenter_emits_one_utterance_after_pause():
    seg = UtteranceSegmenter()
    audio = np.concatenate([_silence(1.0), _tone(1.0), _silence(1.5)])

    utterances = seg.feed(audio)

    assert len(utterances) == 1
    assert 1.0 < len(utterances[0]) / SAMPLE_RATE < 2.0


def test_segmenter_works_when_audio_arrives_in_odd_sized_chunks():
    seg = UtteranceSegmenter()
    audio = np.concatenate([_silence(1.0), _tone(1.0), _silence(1.5)])

    utterances = []
    for start in range(0, len(audio), 1234):
        utterances += seg.feed(audio[start : start + 1234])

    assert len(utterances) == 1


def test_segmenter_ignores_short_clicks():
    seg = UtteranceSegmenter()

    assert seg.feed(np.concatenate([_silence(1.0), _tone(0.2), _silence(1.5)])) == []


def test_segmenter_cuts_nonstop_speech_at_max_length():
    seg = UtteranceSegmenter()

    utterances = seg.feed(np.concatenate([_silence(0.5), _tone(25.0), _silence(1.5)]))

    assert len(utterances) == 2


def test_segmenter_flush_emits_speech_still_in_progress():
    seg = UtteranceSegmenter()
    seg.feed(np.concatenate([_silence(1.0), _tone(1.0)]))

    assert len(seg.flush()) == 1


def test_to_16k_mono_converts_browser_style_int16_stereo_48k():
    stereo = np.zeros((48_000, 2), dtype=np.int16)
    stereo[:, 0] = 16384

    out = to_16k_mono(48_000, stereo)

    assert out.dtype == np.float32
    assert len(out) == 16_000
    assert abs(out.mean() - 0.25) < 0.01  # half-scale left + silent right, averaged


# ---- STT noise filter ----

def test_is_noise_text_catches_whisper_hallucinations_and_repeats():
    assert is_noise_text("시청해주셔서 감사합니다.")
    assert is_noise_text("아아아아아")
    assert is_noise_text("  ...  ")
    assert not is_noise_text("김치찌개 해먹고 싶어")


# ---- TTS text cleanup ----

def test_to_speech_text_strips_markdown_links_and_emoji():
    md = "## 메뉴\n1. **두부**를 썬다 🔪\n- [레시피](https://example.com) 참고\n| a | b |"

    assert to_speech_text(md) == "메뉴\n두부를 썬다\n레시피 참고\na b"


# ---- controller ----

class _SyncExecutor:
    def submit(self, fn, *args):
        fn(*args)


class _StubTranscriber:
    def __init__(self, text="김치찌개 해먹고 싶어"):
        self.text = text
        self.calls = []

    def transcribe(self, audio, model_name):
        self.calls.append(model_name)
        return self.text


class _StubSpeaker:
    def synthesize(self, text):
        return (44_100, np.zeros(44_100 * 2, dtype=np.int16)) if text else None


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _controller(transcriber=None, clock=None):
    return VoiceController(
        transcriber=transcriber or _StubTranscriber(),
        speaker=_StubSpeaker(),
        stt_model="small",
        executor=_SyncExecutor(),
        clock=clock or _Clock(),
    )


def _speech_chunk():
    return np.concatenate([_silence(1.0), _tone(1.0), _silence(1.5)])


def test_browser_audio_flows_to_inbox_via_selected_stt_model():
    transcriber = _StubTranscriber()
    ctrl = _controller(transcriber)
    ctrl.stt_model = "medium"

    ctrl.feed_browser_audio(SAMPLE_RATE, _speech_chunk())

    assert ctrl.pop_text() == "김치찌개 해먹고 싶어"
    assert transcriber.calls == ["medium"]
    assert ctrl.pop_text() is None


def test_pop_text_joins_utterances_split_by_a_pause():
    ctrl = _controller()
    ctrl.feed_browser_audio(SAMPLE_RATE, _speech_chunk())
    ctrl.feed_browser_audio(SAMPLE_RATE, _speech_chunk())

    assert ctrl.pop_text() == "김치찌개 해먹고 싶어 김치찌개 해먹고 싶어"


def test_mic_input_is_dropped_while_a_turn_is_processing():
    ctrl = _controller()
    ctrl.begin_turn()

    ctrl.feed_browser_audio(SAMPLE_RATE, _speech_chunk())

    assert ctrl.pop_text() is None


def test_mic_stays_muted_until_spoken_reply_should_have_finished():
    clock = _Clock()
    ctrl = _controller(clock=clock)
    ctrl.begin_turn()
    ctrl.end_turn(speech_seconds=3.0)

    clock.now += 3.0
    assert ctrl.is_muted()
    clock.now += PLAYBACK_MARGIN_SEC + 0.01
    assert not ctrl.is_muted()


def test_empty_transcription_is_not_queued():
    ctrl = _controller(_StubTranscriber(text=""))

    ctrl.feed_browser_audio(SAMPLE_RATE, _speech_chunk())

    assert ctrl.pop_text() is None


def test_speak_returns_audio_and_duration():
    ctrl = _controller()

    audio, seconds = ctrl.speak("안녕하세요")

    assert audio[0] == 44_100
    assert seconds == 2.0
    assert ctrl.speak("") == (None, 0.0)


def test_set_tts_speed_is_clamped_to_the_slider_range():
    from cooking_assistant_chatbot.voice.tts import MAX_SPEED, MIN_SPEED

    controller = _controller()
    speaker = controller._speaker

    controller.set_tts_speed(1.3)
    assert speaker.speed == 1.3
    controller.set_tts_speed(5)
    assert speaker.speed == MAX_SPEED
    controller.set_tts_speed(0.1)
    assert speaker.speed == MIN_SPEED


def test_speaker_passes_speed_to_supertonic():
    from cooking_assistant_chatbot.voice.tts import Speaker

    calls = {}

    class _FakeTTS:
        sample_rate = 44_100

        def synthesize(self, text, voice_style, lang, speed):
            calls["speed"] = speed
            return np.zeros(441, dtype=np.float32), None

    speaker = Speaker("supertonic-2", "F1", speed=1.4)
    speaker._tts = _FakeTTS()  # skip the real model load

    speaker.synthesize("안녕")

    assert calls["speed"] == 1.4
