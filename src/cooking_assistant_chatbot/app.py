from __future__ import annotations

import logging
import threading
import time

import gradio as gr

from .agent.pipeline import handle_message
from .config import settings
from .cooking import session as cooking
from .progress import run_with_progress
from .data import chat_store
from .data.db import get_connection, get_recipes_by_ids
from .pricing.enuri_client import EnuriClient
from .pricing.selection import DEFAULT_PRICE_BASIS, PRICE_BASIS_LABELS
from .rag.embeddings import BGEEmbeddingFunction
from .rag.search import RecipeSearcher
from .voice.controller import VoiceController
from .voice.pc_mic import list_input_devices
from .voice.stt import STT_MODELS, Transcriber
from .voice.tts import MAX_SPEED, MIN_SPEED, Speaker

logger = logging.getLogger(__name__)

BROWSER_MIC = "브라우저 마이크"
PC_MIC = "PC 마이크"
_VOICE_POLL_SEC = 0.5


def _progress_message(status: str, elapsed_sec: float) -> dict:
    """Placeholder assistant bubble shown while a reply is being built;
    replaced by the real reply when it's done."""
    return {"role": "assistant", "content": f"⏳ {status or '처리 중'} · {int(elapsed_sec)}초"}


_BASIS_BY_LABEL = {label: basis for basis, label in PRICE_BASIS_LABELS.items()}


def _plain_history(history: list[dict]) -> list[dict]:
    """Gradio chat messages → {"role", "content": str} for the LLM."""
    plain = []
    for message in history:
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if isinstance(content, str) and content:
            plain.append({"role": message["role"], "content": content})
    return plain


def _conversation_choices(conn) -> list[tuple[str, int]]:
    """Sidebar radio choices: (title, id), most recently used first."""
    return [(c.title, c.id) for c in chat_store.list_conversations(conn)]


def is_cooking_start_request(text: str) -> bool:
    """"요리 시작", "요리 시작하자", "요리 모드" — typed or spoken."""
    compact = text.replace(" ", "")
    return any(p in compact for p in ("요리시작", "요리모드", "요리할래", "요리하자"))


# Keeps the phone screen on during cooking mode. When the page is hidden
# (screen locked, other app), browsers pause its timers, so the 0.5s voice
# poll stops: no voice commands and no timer alarm. Found while testing
# with a background tab, where gr.Timer never ticked. Screen Wake Lock is
# in iOS Safari 16.4+; the lock drops whenever the page is hidden, so it's
# re-requested when the page becomes visible again.
_WAKE_LOCK_HEAD = """
<script>
window.cookWake = {
  lock: null, wanted: false,
  async on() {
    this.wanted = true;
    try { this.lock = await navigator.wakeLock.request("screen"); } catch (e) {}
  },
  off() {
    this.wanted = false;
    if (this.lock) { this.lock.release().catch(() => {}); this.lock = null; }
  },
};
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && window.cookWake.wanted) window.cookWake.on();
});
</script>
"""

_ERROR_REPLY = "죄송해요, 답변을 만드는 중에 오류가 났어요. 다시 한 번 물어봐주실래요?"
_NO_RECIPE_FOR_COOKING = (
    "이 채팅에서 아직 레시피를 보여드리지 않았어요. 먼저 '○○ 레시피 알려줘'로 레시피와 재료를 "
    "확인한 다음 요리를 시작해주세요."
)


def build_app() -> gr.Blocks:
    """Wire up the shared backend resources once and return a Gradio app.

    Local-only prototype UI, per the earlier design decision to keep the
    frontend minimal and spend effort on the AI/agent side instead.
    """
    conn = get_connection(settings.db_path)
    embedding_function = BGEEmbeddingFunction(settings.embedding_model_name)
    # Load it now (in the background, so the UI still comes up immediately)
    # instead of on the first search, which cost ~40s of the first reply.
    threading.Thread(target=embedding_function.load, daemon=True).start()
    searcher = RecipeSearcher(settings.chroma_db_path, settings.db_path, embedding_function)
    price_client = EnuriClient()
    voice = VoiceController(
        transcriber=Transcriber(model_dir=settings.stt_model_dir),
        speaker=Speaker(settings.tts_model, settings.tts_voice, settings.tts_speed),
        stt_model=settings.stt_model,
    )
    devices = list_input_devices()
    device_labels = [d.label for d in devices]

    # Handlers below return {component: value} dicts, so each one updates only
    # what it changes; the components are created further down in the Blocks.
    # Never return a bare {} for "nothing changed": Gradio reads an empty dict
    # as one output *value* and fails the event ("didn't return enough output
    # values") — on the 0.5s voice poll that meant every idle tick errored.
    def _no_change() -> dict:
        return {cook_state: gr.skip()}

    def _sidebar(conv: int | None, interactive: bool) -> dict:
        return {
            conv_list: gr.update(
                choices=_conversation_choices(conn), value=conv, interactive=interactive
            ),
            new_chat: gr.update(interactive=interactive),
            delete_chat: gr.update(interactive=interactive),
            cook_start: gr.update(
                interactive=interactive and chat_store.get_recipe_seq(conn, conv) is not None
            ),
        }

    def _speak(text: str) -> tuple:
        """TTS for cooking mode (always on there — the point is hands-free),
        muting the mic while it plays so the bot doesn't hear itself."""
        voice.begin_turn()
        seconds = 0.0
        try:
            audio, seconds = voice.speak(text)
        finally:
            voice.end_turn(seconds)
        return audio

    def _cooking_view(session, speech: str | None) -> dict:
        step_md, timer_md = cooking.render(session, time.monotonic())
        update = {
            cook_state: session,
            cook_panel: gr.update(visible=session is not None),
            cook_step: step_md,
            cook_timer: timer_md,
        }
        if speech:
            update[tts_audio] = _speak(speech)
        return update

    def start_cooking(conv: int | None) -> dict:
        recipe_seq = chat_store.get_recipe_seq(conn, conv)
        recipes = get_recipes_by_ids(conn, [recipe_seq]) if recipe_seq else []
        if not recipes:
            return {voice_status: _NO_RECIPE_FOR_COOKING}
        voice.preload_stt()
        session, speech = cooking.start(recipes[0])
        return {**_cooking_view(session, speech), voice_status: ""}

    def cooking_command(session, text: str) -> dict:
        if session is None:
            return _no_change()
        session, speech = cooking.handle(session, text, time.monotonic())
        return _cooking_view(session, speech)

    def run_turn(message: str, conv: int | None, tts_on: bool, basis_label: str):
        """One chat turn in conversation `conv` (None = a new chat, created on
        its first message). History for the agent comes from the saved
        conversation, not from what the browser is showing, so separate
        chats never share context. The sidebar is locked for the duration so
        the reply can't land in a chat the user switched to mid-turn."""
        message = (message or "").strip()
        if not message:
            yield _no_change()
            return
        if is_cooking_start_request(message) and chat_store.get_recipe_seq(conn, conv):
            yield {**start_cooking(conv), msg: ""}
            return
        if conv is None:
            conv = chat_store.create_conversation(conn, chat_store.title_from_message(message))
        prior = chat_store.get_messages(conn, conv)
        user_message = {"role": "user", "content": message}
        chat_store.append_messages(conn, conv, [user_message])
        shown = [*prior, user_message]
        started = time.monotonic()
        yield {
            chatbot: [*shown, _progress_message("요청 이해하는 중", 0)],
            tts_audio: None,
            msg: "",
            conv_id: conv,
            **_sidebar(conv, interactive=False),
        }
        voice.begin_turn()
        speech_seconds = 0.0
        try:
            work = run_with_progress(
                lambda report: handle_message(
                    message,
                    searcher,
                    price_client,
                    conn,
                    history=_plain_history(prior),
                    on_progress=report,
                    price_basis=_BASIS_BY_LABEL.get(basis_label, DEFAULT_PRICE_BASIS),
                )
            )
            reply = None
            while True:
                try:
                    status = next(work)
                except StopIteration as done:
                    reply = done.value
                    break
                except Exception:
                    # Keep the turn (and the sidebar unlock below) going
                    # instead of leaving the UI stuck mid-turn.
                    break
                elapsed = time.monotonic() - started
                yield {chatbot: [*shown, _progress_message(status, elapsed)]}

            text = reply.text if reply is not None else _ERROR_REPLY
            chat_store.append_messages(conn, conv, [{"role": "assistant", "content": text}])
            if reply is not None and reply.recipe_seq:
                chat_store.set_recipe(conn, conv, reply.recipe_seq)
            audio = None
            if tts_on and reply is not None:
                elapsed = time.monotonic() - started
                yield {chatbot: [*shown, _progress_message("음성 만드는 중", elapsed)]}
                audio, speech_seconds = voice.speak(reply.speech)
            yield {
                chatbot: [*shown, {"role": "assistant", "content": text}],
                tts_audio: audio,
                **_sidebar(conv, interactive=True),
            }
        finally:
            voice.end_turn(speech_seconds)

    def poll_voice(conv: int | None, tts_on: bool, basis_label: str, session):
        """Every 0.5s: route recognized speech to cooking mode when it's on
        (commands, no LLM), otherwise to a chat turn; and in cooking mode,
        keep the timer display ticking and ring it when it runs out."""
        text = None if voice.is_muted() else voice.pop_text()
        if text:
            logger.info("voice: %r (cooking mode: %s)", text, session is not None)
        if session is not None:
            if text:
                yield cooking_command(session, text)
                return
            session, alarm = cooking.check_timer(session, time.monotonic())
            if alarm or session.timer_deadline is not None:
                yield _cooking_view(session, alarm)
            else:
                yield _no_change()
            return
        if text:
            yield from run_turn(text, conv, tts_on, basis_label)
        else:
            yield _no_change()

    def open_conversation(conv: int | None) -> dict:
        messages = chat_store.get_messages(conn, conv) if conv is not None else []
        return {chatbot: messages, conv_id: conv, **_sidebar(conv, interactive=True)}

    def new_conversation() -> dict:
        # Created lazily on the first message, so an unused "new chat"
        # doesn't leave an empty entry behind.
        return {chatbot: [], conv_id: None, **_sidebar(None, interactive=True)}

    def delete_current(conv: int | None) -> dict:
        if conv is not None:
            chat_store.delete_conversation(conn, conv)
        return new_conversation()

    def on_source_change(source: str):
        if source != PC_MIC and voice.pc_mic.running:
            voice.pc_mic.stop()
        return (
            gr.update(visible=source == BROWSER_MIC),
            gr.update(visible=source == PC_MIC),
            "",
        )

    def start_pc_mic(device_label: str | None):
        index = next((d.index for d in devices if d.label == device_label), None)
        voice.preload_stt()
        voice.pc_mic.start(index)
        return "🎙 듣는 중 — 말하면 자동으로 전송돼요. 봇이 답하는 동안은 듣지 않아요."

    def stop_pc_mic():
        voice.pc_mic.stop()
        return "⏹ 중지됨"

    def set_stt_model(model_name: str) -> None:
        voice.stt_model = model_name
        voice.preload_stt()

    def on_tts_toggle(enabled: bool) -> None:
        if enabled:
            voice.preload_tts()

    with gr.Blocks(title="요리 챗봇", head=_WAKE_LOCK_HEAD) as app:
        conv_id = gr.State(None)
        cook_state = gr.State(None)
        with gr.Sidebar(open=True):
            new_chat = gr.Button("＋ 새 채팅", variant="primary")
            conv_list = gr.Radio(choices=[], value=None, label="채팅 목록")
            delete_chat = gr.Button("🗑 이 채팅 삭제", size="sm")
        gr.Markdown("# 요리 챗봇\n레시피 추천·만드는 법, 영양성분, 재료비 계산을 물어보세요.")

        with gr.Group(visible=False) as cook_panel:
            cook_step = gr.Markdown()
            cook_timer = gr.Markdown()
            with gr.Row():
                cook_prev = gr.Button("◀ 이전")
                cook_repeat = gr.Button("🔁 다시")
                cook_next = gr.Button("다음 ▶", variant="primary")
            with gr.Row():
                cook_timer_off = gr.Button("⏱ 타이머 끄기", size="sm")
                cook_exit = gr.Button("요리 끝", size="sm")

        chatbot = gr.Chatbot(height=520)
        with gr.Row():
            msg = gr.Textbox(placeholder="메시지를 입력하세요", show_label=False, scale=8)
            send = gr.Button("보내기", variant="primary", scale=1)
        cook_start = gr.Button(
            "🍳 요리 시작 — 이 채팅에서 본 레시피를 단계별로 읽어드려요 (음성: '다음', '이전', '다시', '타이머 5분')",
            interactive=False,
        )
        gr.Examples(
            [
                "된장찌개 어떻게 만들어?",
                "김치찌개 칼로리 얼마야?",
                "오므라이스 재료비 얼마나 들어?",
                "냉장고에 두부랑 계란 있는데 뭐 해먹지?",
            ],
            inputs=msg,
        )
        price_basis = gr.Radio(
            list(PRICE_BASIS_LABELS.values()),
            value=PRICE_BASIS_LABELS[DEFAULT_PRICE_BASIS],
            label="재료 가격 기준 (관련도: 검색 상위 최저가 · 최소 지출: 필요한 양 이상 중 가장 싼 상품 · 단위가격: g/ml당 최저가)",
        )

        with gr.Accordion("음성", open=True):
            with gr.Row():
                mic_source = gr.Radio([BROWSER_MIC, PC_MIC], value=BROWSER_MIC, label="마이크")
                stt_model = gr.Dropdown(
                    list(STT_MODELS), value=settings.stt_model, label="인식 모델 (클수록 정확·느림)"
                )
                tts_on = gr.Checkbox(value=False, label="답변 읽어주기")
                tts_speed = gr.Slider(
                    MIN_SPEED,
                    MAX_SPEED,
                    value=settings.tts_speed,
                    step=0.05,
                    label="음성 속도 (1.05 = 기본, 클수록 빠름)",
                )
            browser_mic = gr.Audio(
                sources=["microphone"],
                streaming=True,
                label="녹음을 누르면 중지할 때까지 계속 듣고, 말이 끝날 때마다 자동으로 전송해요",
            )
            with gr.Group(visible=False) as pc_mic_group:
                with gr.Row():
                    device = gr.Dropdown(
                        device_labels,
                        value=device_labels[0] if device_labels else None,
                        label="입력 장치",
                    )
                    pc_start = gr.Button("🎙 듣기 시작")
                    pc_stop = gr.Button("⏹ 중지")
            voice_status = gr.Markdown("")
            tts_audio = gr.Audio(label="음성 답변", autoplay=True, interactive=False)

        sidebar_outputs = [conv_list, new_chat, delete_chat, cook_start]
        cooking_outputs = [cook_state, cook_panel, cook_step, cook_timer, tts_audio, voice_status]
        turn_outputs = [chatbot, msg, conv_id, *sidebar_outputs, *cooking_outputs]
        turn_opts = dict(concurrency_id="chat_turn", concurrency_limit=1)
        turn_inputs = [msg, conv_id, tts_on, price_basis]
        msg.submit(run_turn, turn_inputs, turn_outputs, **turn_opts)
        send.click(run_turn, turn_inputs, turn_outputs, **turn_opts)

        gr.Timer(_VOICE_POLL_SEC).tick(
            poll_voice,
            [conv_id, tts_on, price_basis, cook_state],
            turn_outputs,
            trigger_mode="once",
            **turn_opts,
        )

        cook_start.click(start_cooking, conv_id, cooking_outputs, **turn_opts)
        cook_start.click(None, None, None, js="() => window.cookWake && window.cookWake.on()")
        cook_exit.click(None, None, None, js="() => window.cookWake && window.cookWake.off()")
        for button, command in [
            (cook_prev, "이전"),
            (cook_repeat, "다시"),
            (cook_next, "다음"),
            (cook_timer_off, "타이머 꺼"),
            (cook_exit, "요리 끝"),
        ]:
            button.click(
                lambda session, command=command: cooking_command(session, command),
                cook_state,
                cooking_outputs,
                **turn_opts,
            )

        # .input, not .change: fires only on the user's own clicks, not when
        # a turn updates the radio's value/choices.
        chat_outputs = [chatbot, conv_id, *sidebar_outputs]
        conv_list.input(open_conversation, conv_list, chat_outputs)
        new_chat.click(new_conversation, None, chat_outputs)
        delete_chat.click(delete_current, conv_id, chat_outputs)
        app.load(open_conversation, conv_id, chat_outputs)
        browser_mic.start_recording(voice.preload_stt)
        browser_mic.stream(
            lambda chunk: voice.feed_browser_audio(*chunk) if chunk is not None else None,
            inputs=browser_mic,
            outputs=None,
            stream_every=0.5,
        )
        browser_mic.stop_recording(voice.stop_browser_audio)

        mic_source.change(
            on_source_change, mic_source, [browser_mic, pc_mic_group, voice_status]
        )
        pc_start.click(start_pc_mic, device, voice_status)
        pc_stop.click(stop_pc_mic, None, voice_status)
        stt_model.change(set_stt_model, stt_model, None)
        tts_on.change(on_tts_toggle, tts_on, None)
        # .release: apply once the slider is let go, not on every drag step.
        tts_speed.release(voice.set_tts_speed, tts_speed, None)

    return app


def resolve_auth() -> tuple[str, str] | None:
    """None (no login prompt) unless a password is configured — see
    config.Settings.app_password."""
    if not settings.app_password:
        return None
    return (settings.app_username, settings.app_password)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    app = build_app()
    app.launch(auth=resolve_auth())


if __name__ == "__main__":
    main()
