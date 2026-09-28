from __future__ import annotations

import logging
import threading
import time

import gradio as gr

from .agent.pipeline import handle_message
from .config import settings
from .cooking import session as cooking
from .cooking.commands import parse_command
from .progress import run_with_progress
from .data import chat_store, user_recipes
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


def is_typed_cooking_command(text: str, session) -> bool:
    """A message typed while cooking mode is on that is a cooking command
    ("다음", "타이머 3분"). Voice always goes to cooking mode, but typed text
    used to go to the LLM even then, so "타이머 3분" got a chat reply instead
    of a timer. Anything else typed still goes to chat as a question."""
    return session is not None and parse_command(text).kind != "unknown"


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

    def run_turn(message: str, conv: int | None, tts_on: bool, basis_label: str, session=None):
        """One chat turn in conversation `conv` (None = a new chat, created on
        its first message). History for the agent comes from the saved
        conversation, not from what the browser is showing, so separate
        chats never share context. The sidebar is locked for the duration so
        the reply can't land in a chat the user switched to mid-turn."""
        message = (message or "").strip()
        if not message:
            yield _no_change()
            return
        if is_typed_cooking_command(message, session):
            yield {**cooking_command(session, message), msg: ""}
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
            draft_update = _draft_into_form(reply.recipe_draft) if reply else {}
            audio = None
            if tts_on and reply is not None:
                elapsed = time.monotonic() - started
                yield {chatbot: [*shown, _progress_message("음성 만드는 중", elapsed)]}
                audio, speech_seconds = voice.speak(reply.speech)
            yield {
                chatbot: [*shown, {"role": "assistant", "content": text}],
                tts_audio: audio,
                **_sidebar(conv, interactive=True),
                **draft_update,
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

    # ---- recipe board ----
    def _board_choices(query: str) -> list[tuple[str, int]]:
        return [
            (f"{r.draft.name} · {r.author} · {r.created_at[:10]}", r.id)
            for r in user_recipes.search(conn, query or "")
        ]

    def show_board(query: str = "", status: str = "") -> dict:
        return {
            board_list: gr.update(choices=_board_choices(query), value=None),
            post_group: gr.update(visible=False),
            form_group: gr.update(visible=False),
            form_status: status,
            editing_id: None,
            viewing_id: None,
        }

    def open_post(recipe_id: int | None) -> dict:
        recipe = user_recipes.get(conn, recipe_id) if recipe_id is not None else None
        if recipe is None:
            return show_board()
        return {
            post_group: gr.update(visible=True),
            post_view: user_recipes.render(recipe),
            form_group: gr.update(visible=False),
            form_status: "",
            viewing_id: recipe.id,
        }

    def _form_values(draft) -> dict:
        nutrition = draft.nutrition if draft else {}
        values = {
            form_name: draft.name if draft else "",
            form_servings: draft.servings if draft else None,
            form_ingredients: "\n".join(draft.ingredients) if draft else "",
            form_steps: "\n".join(draft.steps) if draft else "",
        }
        for box, key in zip(form_nutrition, user_recipes.NUTRITION_FIELDS):
            values[box] = float(nutrition[key]) if key in nutrition else None
        return values

    def _draft_into_form(draft) -> dict:
        """After a chat turn: if the model drafted a recipe, open it in the
        registration form on the board tab for the user to check and save."""
        if draft is None:
            return {}
        return {
            tabs: gr.Tabs(selected="board"),
            form_group: gr.update(visible=True),
            post_group: gr.update(visible=False),
            form_status: "채팅 내용으로 양식을 채웠어요. 확인하고 위의 '내 이름'을 적은 뒤 **저장**을 눌러주세요.",
            editing_id: None,
            **_form_values(draft),
        }

    def new_post() -> dict:
        return {
            post_group: gr.update(visible=False),
            form_group: gr.update(visible=True),
            form_status: "",
            editing_id: None,
            **_form_values(None),
        }

    def edit_post(recipe_id: int | None, author: str) -> dict:
        recipe = user_recipes.get(conn, recipe_id) if recipe_id is not None else None
        if recipe is None:
            return show_board()
        if recipe.author != (author or "").strip():
            return {form_status: f"'{recipe.author}'님이 등록한 레시피예요. 위의 '내 이름'이 같아야 수정할 수 있어요."}
        return {
            post_group: gr.update(visible=False),
            form_group: gr.update(visible=True),
            form_status: "",
            editing_id: recipe.id,
            **_form_values(recipe.draft),
        }

    def delete_post(recipe_id: int | None, author: str, confirmed: bool, query: str) -> dict:
        if recipe_id is None:
            return show_board(query)
        if not confirmed:
            return {form_status: "삭제하려면 '삭제하려면 체크'를 먼저 체크해주세요."}
        try:
            user_recipes.delete(conn, recipe_id, author or "")
        except user_recipes.PermissionDenied as error:
            return {form_status: str(error), post_delete_confirm: False}
        return {**show_board(query, "삭제했어요."), post_delete_confirm: False}

    def save_post(recipe_id, author, name, servings, ingredients, steps, *rest):
        *nutrition_values, query = rest
        draft = user_recipes.RecipeDraft(
            name=name or "",
            servings=int(servings) if servings else None,
            ingredients=(ingredients or "").split("\n"),
            steps=(steps or "").split("\n"),
            nutrition={
                key: f"{value:g}"
                for key, value in zip(user_recipes.NUTRITION_FIELDS, nutrition_values)
                if value is not None
            },
        )
        try:
            saved = user_recipes.save(conn, draft, author or "", recipe_id)
        except (ValueError, user_recipes.PermissionDenied) as error:
            return {form_status: f"⚠️ {error}"}
        return {
            **show_board(query, f"'{saved.draft.name}' 레시피를 {'수정' if recipe_id else '등록'}했어요."),
            **open_post(saved.id),
            board_list: gr.update(choices=_board_choices(query), value=saved.id),
        }

    def example_post() -> dict:
        return _form_values(
            user_recipes.RecipeDraft(
                name="자취생 간장계란밥",
                servings=1,
                ingredients=["밥 1공기", "계란 2개", "간장 1큰술", "참기름 약간", "김가루 약간"],
                steps=[
                    "팬에 기름을 두르고 계란을 반숙으로 부친다.",
                    "밥 위에 계란을 올린다.",
                    "간장과 참기름을 두르고 김가루를 뿌려 비빈다.",
                ],
            )
        )

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
        conv_id = gr.State(None)
        cook_state = gr.State(None)
        editing_id = gr.State(None)
        viewing_id = gr.State(None)
        # The registrant's name, remembered in this browser's localStorage.
        # BrowserState encrypts with a random per-launch key unless given a
        # secret, so the name was unreadable after every app restart. A fixed
        # secret is fine here: it's a display name, not a credential.
        author_store = gr.BrowserState(
            "", storage_key="cook_chatbot_author", secret="cook-chatbot-author-name"
        )
        with gr.Sidebar(open=True):
            new_chat = gr.Button("＋ 새 채팅", variant="primary")
            conv_list = gr.Radio(choices=[], value=None, label="채팅 목록")
            delete_chat = gr.Button("🗑 이 채팅 삭제", size="sm")
        gr.Markdown("# 요리 챗봇\n레시피 추천·만드는 법, 영양성분, 재료비 계산을 물어보세요.")

        with gr.Tabs(selected="chat") as tabs:
            with gr.Tab("💬 채팅", id="chat"):
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

            with gr.Tab("📖 레시피 게시판", id="board"):
                gr.Markdown(
                    "직접 만든 레시피를 올리고 찾아보는 곳이에요. 여기 올린 레시피는 채팅의 검색·추천에는 쓰이지 않아요."
                )
                # Always visible: it's both the new post's author and the
                # name edit/delete permission is checked against.
                form_author = gr.Textbox(
                    label="내 이름 (등록자로 저장되고, 수정·삭제는 같은 이름일 때만 돼요)",
                    placeholder="이 브라우저가 기억해요",
                )
                with gr.Row():
                    board_query = gr.Textbox(
                        placeholder="제목이나 작성자로 찾기", show_label=False, scale=6
                    )
                    board_search = gr.Button("🔍 찾기", scale=1)
                    board_new = gr.Button("✏️ 새 레시피 등록", variant="primary", scale=2)
                board_list = gr.Radio(choices=[], value=None, label="레시피 목록 (최신순)")
                with gr.Group(visible=False) as post_group:
                    post_view = gr.Markdown()
                    with gr.Row():
                        post_edit = gr.Button("수정")
                        post_delete_confirm = gr.Checkbox(label="삭제하려면 체크", value=False)
                        post_delete = gr.Button("삭제", variant="stop")
                with gr.Group(visible=False) as form_group:
                    gr.Markdown("### 레시피 등록 양식")
                    form_name = gr.Textbox(label="메뉴 이름", placeholder="예: 자취생 간장계란밥")
                    form_servings = gr.Number(label="몇 인분 (선택)", precision=0, minimum=0)
                    form_ingredients = gr.Textbox(
                        label="재료 (한 줄에 하나씩, '재료 분량')",
                        lines=6,
                        placeholder="밥 1공기\n계란 2개\n간장 1큰술\n참기름 약간",
                    )
                    form_steps = gr.Textbox(
                        label="조리 순서 (한 줄에 한 단계)",
                        lines=6,
                        placeholder="계란을 반숙으로 부친다.\n밥 위에 계란을 올린다.\n간장과 참기름을 두르고 비빈다.",
                    )
                    with gr.Accordion("영양성분 (선택)", open=False):
                        with gr.Row():
                            form_nutrition = [
                                gr.Number(label=f"{label} ({unit})", minimum=0)
                                for label, unit in user_recipes.NUTRITION_FIELDS.values()
                            ]
                    with gr.Row():
                        form_save = gr.Button("💾 저장", variant="primary")
                        form_example = gr.Button("📋 예시로 채우기")
                        form_cancel = gr.Button("취소")
                form_status = gr.Markdown()

        sidebar_outputs = [conv_list, new_chat, delete_chat, cook_start]
        cooking_outputs = [cook_state, cook_panel, cook_step, cook_timer, tts_audio, voice_status]
        turn_outputs = [
            chatbot, msg, conv_id, *sidebar_outputs, *cooking_outputs,
            tabs, form_group, post_group, form_status, editing_id,
            form_name, form_servings, form_ingredients, form_steps, *form_nutrition,
        ]
        turn_opts = dict(concurrency_id="chat_turn", concurrency_limit=1)
        turn_inputs = [msg, conv_id, tts_on, price_basis, cook_state]
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

        form_fields = [form_name, form_servings, form_ingredients, form_steps, *form_nutrition]
        board_outputs = [board_list, post_group, post_view, form_group, form_status, editing_id, viewing_id]
        board_search.click(show_board, board_query, board_outputs)
        board_query.submit(show_board, board_query, board_outputs)
        board_list.input(open_post, board_list, board_outputs)
        board_new.click(new_post, None, [*board_outputs, *form_fields])
        post_edit.click(edit_post, [viewing_id, form_author], [*board_outputs, *form_fields])
        post_delete.click(
            delete_post, [viewing_id, form_author, post_delete_confirm, board_query],
            [*board_outputs, post_delete_confirm],
        )
        form_save.click(
            save_post, [editing_id, form_author, *form_fields, board_query], board_outputs
        )
        form_example.click(example_post, None, form_fields)
        form_cancel.click(show_board, board_query, board_outputs)
        app.load(show_board, board_query, board_outputs)
        app.load(lambda stored: stored or "", author_store, form_author)
        form_author.change(lambda name: name, form_author, author_store)
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
    # head= belongs to launch() in Gradio 6 (on Blocks() it's ignored with a
    # warning — which is how the wake-lock script went missing at first).
    app.launch(auth=resolve_auth(), head=_WAKE_LOCK_HEAD)


if __name__ == "__main__":
    main()
