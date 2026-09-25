from __future__ import annotations

import threading
import time

import gradio as gr

from .agent.pipeline import handle_message
from .config import settings
from .progress import run_with_progress
from .data.db import get_connection
from .pricing.enuri_client import EnuriClient
from .pricing.selection import DEFAULT_PRICE_BASIS, PRICE_BASIS_LABELS
from .rag.embeddings import BGEEmbeddingFunction
from .rag.search import RecipeSearcher
from .voice.controller import VoiceController
from .voice.pc_mic import list_input_devices
from .voice.stt import STT_MODELS, Transcriber
from .voice.tts import Speaker

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
        speaker=Speaker(settings.tts_model, settings.tts_voice),
        stt_model=settings.stt_model,
    )
    devices = list_input_devices()
    device_labels = [d.label for d in devices]

    def run_turn(message: str, history: list, tts_on: bool, basis_label: str):
        message = (message or "").strip()
        if not message:
            yield gr.skip(), gr.skip(), gr.skip()
            return
        history = [*history, {"role": "user", "content": message}]
        started = time.monotonic()
        yield [*history, _progress_message("요청 이해하는 중", 0)], None, ""
        voice.begin_turn()
        speech_seconds = 0.0
        try:
            prior = _plain_history(history[:-1])
            work = run_with_progress(
                lambda report: handle_message(
                    message,
                    searcher,
                    price_client,
                    conn,
                    history=prior,
                    on_progress=report,
                    price_basis=_BASIS_BY_LABEL.get(basis_label, DEFAULT_PRICE_BASIS),
                )
            )
            while True:
                try:
                    status = next(work)
                except StopIteration as done:
                    reply = done.value
                    break
                elapsed = time.monotonic() - started
                yield [*history, _progress_message(status, elapsed)], gr.skip(), gr.skip()

            audio = None
            if tts_on:
                elapsed = time.monotonic() - started
                yield [*history, _progress_message("음성 만드는 중", elapsed)], gr.skip(), gr.skip()
                audio, speech_seconds = voice.speak(reply.speech)
            yield [*history, {"role": "assistant", "content": reply.text}], audio, ""
        finally:
            voice.end_turn(speech_seconds)

    def poll_voice(history: list, tts_on: bool, basis_label: str):
        text = None if voice.is_muted() else voice.pop_text()
        if not text:
            yield gr.skip(), gr.skip(), gr.skip()
            return
        yield from run_turn(text, history, tts_on, basis_label)

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

    with gr.Blocks(title="요리 챗봇") as app:
        gr.Markdown("# 요리 챗봇\n메뉴 이름을 말하면 레시피와 예상 재료비를 알려드려요.")
        chatbot = gr.Chatbot(height=520)
        with gr.Row():
            msg = gr.Textbox(placeholder="메시지를 입력하세요", show_label=False, scale=8)
            send = gr.Button("보내기", variant="primary", scale=1)
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

        turn_outputs = [chatbot, tts_audio, msg]
        turn_opts = dict(concurrency_id="chat_turn", concurrency_limit=1)
        turn_inputs = [msg, chatbot, tts_on, price_basis]
        msg.submit(run_turn, turn_inputs, turn_outputs, **turn_opts)
        send.click(run_turn, turn_inputs, turn_outputs, **turn_opts)

        gr.Timer(_VOICE_POLL_SEC).tick(
            poll_voice, [chatbot, tts_on, price_basis], turn_outputs, trigger_mode="once", **turn_opts
        )
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

    return app


def resolve_auth() -> tuple[str, str] | None:
    """None (no login prompt) unless a password is configured — see
    config.Settings.app_password."""
    if not settings.app_password:
        return None
    return (settings.app_username, settings.app_password)


def main() -> None:
    app = build_app()
    app.launch(auth=resolve_auth())


if __name__ == "__main__":
    main()
