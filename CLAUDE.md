# Cooking Assistant Chatbot

## Project overview

A personal-use Korean-language chatbot for people living alone. The user names a
dish in natural language; the app finds a recipe and looks up the cheapest price
for each ingredient so the user can cook it affordably. This is also a learning
project for building a local-first RAG + tool-calling AI agent.

Core flow (`agent/pipeline.py` + `agent/tools.py`): a single tool-calling
agent loop. The user message (plus Gradio's prior turns as history) goes to
the LLM with five tools, and the model calls only what the question needs:
- `search_recipes(query)`: RAG/semantic search for recommendations and
  "similar menu" questions.
- `search_recipes_by_ingredients(ingredients)`: exact match; see below.
- `get_recipe(recipe_name)`: ingredients, servings, and steps.
- `get_nutrition(recipe_name)`: COOKRCP01's nutrition fields. The
  만개의레시피 recipes have none, and the tool says so instead of guessing.
- `estimate_ingredient_cost(recipe_name)`: enuri price lookup with
  portioned cost, via `estimate_recipe_price`. This is the slow one
  (1 req/sec per ingredient).

Design notes:
- **Why tools replaced a router.** This used to be an intent router
  (recipe-request vs. general-chat) followed by a fixed
  search -> price-every-ingredient -> compose pipeline. That meant "칼로리
  알려줘" also waited ~100s on price lookups it didn't ask for. The user
  asked for answers that fit what was actually asked, so each capability
  became a tool.
- **Tools take a recipe name, not an id.** `_resolve_recipe` tries an exact
  name match first (spaces ignored, `db.get_recipe_by_name`), then falls
  back to semantic top-1. The result says when it's only a near match.
  Because only final replies are kept in history, follow-ups ("그거
  칼로리는?") work by reusing the name from the previous reply, with no
  extra session state.
- **Loop limits.** Up to `MAX_TOOL_ROUNDS` (4) rounds of tool calls are
  allowed, so compound questions like "추천하고 재료비도" can chain. The
  final round offers no tools, which forces an answer.
- **Temperature 0.3** throughout, per the grounding quirk below. The system
  prompt tells the model to use only tool data and not ask the user
  questions back.
- **Cost results state their coverage.** The portioned-cost total says how
  many ingredients it covers ("18개 중 3개만 계산됨"). Before, a total like
  오므라이스's 697원 read as the whole dish's cost.
- General knowledge questions (e.g. substitutions) are answered with no tool
  call.
- **Ungrounded dish guard** (`pipeline.is_grounded`): a recipe-name tool
  call only runs if at least half of the name's character bigrams appear
  somewhere in the conversation — the current message, history, or this
  turn's earlier tool results.
  - Why: in a brand-new chat, "그거 칼로리는?" made qwen3 call
    get_nutrition("김치찌개") 3 of 4 times. That happened even after a
    prompt rule against guessing, and after removing the '김치찌개' example
    from the tool description.
  - If the guard rejects a call, the model gets a "don't guess, ask which
    dish" result instead. Re-tested: it asked 4/4.

**Chat sessions** (`data/chat_store.py`, sidebar in `app.py`): ChatGPT-style
separate chats.
- Stored in SQLite (`conversations`, `conversation_messages`), so they
  survive restarts and every device (PC, phone via Tailscale) sees the same
  list.
- **Context isolation:** `run_turn` builds the agent's history from the
  saved conversation by id, never from what the browser is showing.
- A chat is created lazily on its first message, titled with that message
  (trimmed, no LLM call).
- The sidebar is locked during a turn, so a reply can't land in a chat the
  user switched to mid-turn.
- A failed turn now shows an error reply instead of leaving the UI stuck.
- Chose this over Gradio's `ChatInterface(save_history=True)`, which stores
  chats in browser localStorage (per device, not shared) and would replace
  the custom Blocks layout (voice controls, price basis).
- Verified in the browser:
  - "그거 칼로리는?" answered from each chat's own context: 된장찌개 in
    the 된장찌개 chat, nothing carried into a new chat.
  - Switching chats restores their history.
  - Chats persist across reload, and delete works.

Stack decisions:
- Backend: Python (FastAPI)
- Frontend: Gradio (prototype UI)
- LLM: **qwen3:14b via Ollama** (settled from the Qwen2.5-14B/Qwen3-14B
  candidates, Q4 quant, ~9.3GB, sized for 16GB VRAM / AMD RX 9060 XT).
  Ollama installed via `winget install Ollama.Ollama`; pull the model with
  `ollama pull qwen3:14b`. Reached through `llm/client.py`, a thin wrapper
  around `litellm.completion(model="ollama_chat/qwen3:14b", ...)` — the
  `ollama_chat/` prefix (not `ollama/`) is required for proper tool-calling
  support. Verified working end-to-end: plain chat and OpenAI-style function
  calling (model correctly emits `tool_calls` with parsed arguments) both
  confirmed against the real local model. Provider switching (local vs.
  cloud) is just changing `settings.llm_model`; RAG/tool-calling/routing
  orchestration itself is hand-built, not LangChain/LangGraph (deliberately
  deferred to a future project).
  **RAM**: Ollama's default mmap model load kept ~9GB of *system RAM*
  resident for qwen3:14b even with all 41 layers on the GPU (`ollama ps`
  says 100% GPU). This got long local test runs killed for low memory
  alongside BGE-M3 and other apps. `llm/client.chat` now sends
  `use_mmap=False` for Ollama models. Measured on this PC:
  - ROCm: free RAM 4.7GB -> 12.3GB, llama-server working set 8.7GB -> 0.7GB.
  - Speed unchanged at ~32 tok/s either way.
  - Vulkan (`OLLAMA_VULKAN`) was also tried: same RAM problem with mmap on,
    same speed. So the backend wasn't the cause; stay on ROCm (the default).
  - The option only takes effect when the model loads. If another app
    already loaded qwen3 with mmap, it stays that way until it's unloaded.
  **Known quirk**: at default settings, qwen3:14b sometimes ignores a long
  data-summarization prompt and free-associates a generic reply instead of
  using the provided recipe/price data — reproduced once, not consistently.
  Fixed (originally for the old recipe-compose step, now applied to the whole
  agent loop in `agent/pipeline.py`) with (a) an
  explicit system prompt stating this is a data-grounded reply, not the
  start of a conversation, and telling it not to ask the user questions back,
  and (b) `temperature=0.3` instead of the default. Re-tested 3x after the
  fix, all correctly formatted. If a similar ignore-the-context failure shows
  up elsewhere, apply the same fix rather than re-diagnosing from scratch.
- Vector store: Chroma, embeddings: BGE-M3
- Structured data: SQLite
- Recipe data: 식약처 COOKRCP01 (1156 recipes) as the base, **supplemented by
  a one-time crawl of 만개의레시피(10000recipe.com)** — 145 recipes across 15
  자취생/casual-dish keywords (떡볶이, 라면, 카레, 파스타, 볶음밥, 김밥,
  오므라이스, 돈까스, 짜장밥, 짬뽕, 토스트, 계란찜, 우동, 덮밥, 마라탕),
  chosen because COOKRCP01 skews toward health/저염식 recipes. 1301 total.
  robots.txt/ToS checked again before running (policies can change — the
  Naver Shopping API died between sessions in this same project); personal,
  non-commercial, not redistributed, self-throttled to 1 req/sec even though
  robots.txt here doesn't specify a Crawl-delay. See
  `data/tenthousand_recipe_crawler.py` / `data/crawl_supplemental_recipes.py`
  to re-run or extend with more keywords — it upserts by a namespaced
  `10000recipe_<id>` key, so re-running is safe (no duplicates). Reads the
  page's schema.org `Recipe` JSON-LD block (same technique as the enuri
  price scraper), which conveniently gives already-clean ingredient strings
  (e.g. "떡 2컵") instead of COOKRCP01's messy free-text blob.
- Deployment: the app still binds to localhost. Remote access (the user's
  phone) goes through **Tailscale Serve**, set up 2026-09-25. No code
  change was needed.
  - Installed with `winget install Tailscale.Tailscale`. The PC joined the
    user's tailnet as `frodan`; the user's iPhone was already on it.
  - `tailscale serve --bg 7860` proxies
    `https://frodan.tailefabc6.ts.net/` -> `http://127.0.0.1:7860`. It
    persists across reboots, and the Tailscale service starts with Windows.
    The chat app itself still has to be running.
  - HTTPS matters: the browser mic (`getUserMedia`) only works in a secure
    context, so a plain `http://100.x.x.x:7860` wouldn't do for voice.
  - It's **tailnet-only**. Funnel (public internet) was deliberately left
    off, so `APP_PASSWORD` is optional. Set it if Funnel is ever turned on.
  - To undo: `tailscale serve --https=443 off`. Check with
    `tailscale serve status`.
  - Verified from this PC through the ts.net URL: HTTP 200 with a valid
    certificate, the page loads, and a chat turn round-trips.
    The user confirmed page access and chat from the iPhone the same day;
    voice (mic, TTS playback) on the phone is still unchecked in `TODO.md`.
- Auth: single-user password gate, not real multi-user support (this is a
  personal single-user tool by design — a "multi-user" deferred item never
  actually fit the product). `app.resolve_auth()` returns `None` (no login
  prompt) unless `APP_PASSWORD` is set in `.env`, in which case Gradio's
  built-in `auth=(username, password)` gates the whole app behind its own
  login form. Verified for real with curl: wrong password -> 400 "Incorrect
  credentials", right password -> 200 + session cookie set. Leave
  `APP_PASSWORD` unset for local-only use; set it before exposing the app
  beyond localhost (e.g. once Tailscale is added).
- Ingredient price lookup: scrapes **에누리(enuri.com)** search results
  (`EnuriClient`, `pricing/enuri_client.py`), reading the page's
  `<script type="application/ld+json">` schema.org block (stable, semantic —
  not the React/Next.js CSS-module markup, which changes every redeploy).
  Results are cached in SQLite (`price_cache` table, 24h TTL, see
  `pricing/cache.py`) so repeat lookups don't re-hit the site. The client
  self-throttles to >=1 req/sec per price.enuri.com's robots.txt Crawl-delay.
  **Product choice is a user-selectable "price basis"**
  (`pricing/selection.py`). The UI radio sets the default. The
  `estimate_ingredient_cost` tool's optional `price_basis` argument
  overrides it when the user asks in chat (e.g. "돈 적게 드는 걸로"; verified
  with qwen3). Each basis is part of the price cache key. The bases:
  - `relevance` (default): the cheapest of enuri's top 5 relevance-ordered
    results, preferring titles that contain the ingredient name. The old
    logic took the cheapest of all ~40 results, and 29 of 76 cached matches
    didn't even name the ingredient (밥 -> latte powder, 고춧가루 ->
    vinegar). The catch: produce at the top is mostly 5-10kg sacks.
  - `min_spend`: the least money out of pocket, among packages that cover
    the recipe's parsed need (or among all candidates if the need can't be
    parsed).
    - This replaced a first try at "smallest package first". enuri's price
      for a small item is often a multi-pack price ("스팸 120g" at 19,350원
      vs "스팸 300g" at 3,430원), so smallest-first came out more expensive
      than `relevance`.
    - Measured: 오므라이스 92,340 -> 59,030원, 된장 두부찌개 63,930 ->
      29,970원.
  - `unit_price`: the lowest price per g/ml. Picks bulk items (25kg salt for
    "소금 약간"); that's the intended trade-off.

  How `min_spend` and `unit_price` pick candidates:
  - They look at all results, but only accept titles where some word *ends
    with* the ingredient name. Korean compounds put the head noun last, so
    수미감자, 진간장 and 흑미밥 pass, while 감자칩 and 대파분태 don't.
  - Package size is the *largest* amount in the title
    (`unit_parser.parse_package_quantity`). With the first match, "감자 소
    (조림용 40g 미만) 10kg" read as 40g.
  - If a basis has nothing usable (e.g. eggs are sold by 구, with no g/ml),
    it falls back to `relevance`.
  All other options were tried and ruled out first, in this order — don't
  re-research from scratch, revisit only if 에누리 itself becomes a problem
  (e.g. starts blocking or its markup changes in a way that breaks scraping):
  1. **네이버쇼핑 검색 API** — officially shut down 2026-07-31 (confirmed via
     developers.naver.com notice #32564), no replacement offered.
  2. **쿠팡파트너스 API** — requires generating 150,000원 in actual affiliate
     sales before the API even activates, plus 10 calls/hour once approved.
  3. **11번가 Open API** — looked promising (documented "상품검색" as a plain
     category, no seller framing) but its account signup turned out to
     require a 사업자등록번호 (business registration number) — not usable by
     an individual.
  4. **옥션/G마켓 (ESM) API** — checked and ruled out without even trying to
     register: their listed functions (AddItem/ReviseItem/GetSellingItemList)
     are for sellers managing their own listings, not general product search.
  다나와 was also considered alongside 에누리 but rejected in favor of it:
  same lack of an official API, but 다나와's robots.txt Crawl-delay is 10s
  vs. 에누리's 1s — 10s/ingredient makes an interactive chatbot response
  impractical (a 10-ingredient recipe would take 100+ seconds).

Deferred to later phases (not in MVP): a real graph DB (Neo4j etc.) for
ingredient/recipe relationships. Real multi-user support (per-user accounts/
data isolation) was considered and deliberately dropped rather than
deferred — this is a personal single-user tool by design, so it never
actually fit; see the Auth bullet above for the lightweight single-password
gate that was built instead. Larger-scale crawling beyond the one
145-recipe supplemental run above is also still deferred —
if more is needed later, extend `DEFAULT_KEYWORDS` in
`crawl_supplemental_recipes.py` rather than re-deriving the legal/scope
reasoning from scratch.

**Resolved (without a graph DB)**: multi-ingredient exact-match recipe
search. Asked "would a graph help accuracy here?" and worked through what a
graph would actually buy over what existed: substitution advice wouldn't
improve (our data has no curated substitution edges, so it'd just be the
LLM's own knowledge either way — same as now); "similar recipe" search
wouldn't meaningfully improve (RAG/embedding search already handles that
reasonably). The one real gap: "what can I cook with exactly X and Y"
was answered by fuzzy semantic search before, which can't guarantee a
recipe actually contains both ingredients. Fixed with a plain SQLite
junction table instead of standing up a graph database —
`data/ingredient_index.py` (`recipe_ingredients` table: recipe_rcp_seq,
ingredient_name, built from the already-parsed ingredient names) +
`find_recipes_by_ingredients()` (set-intersection across per-ingredient
substring matches, ranked by fewest total ingredients). Rebuild after any
data change with `python -m cooking_assistant_chatbot.data.build_ingredient_index`
(not automatic — same pattern as the RAG index build). Wired into
the agent as the `search_recipes_by_ingredients` tool, alongside the
existing RAG search (now `search_recipes`) for mood/style
questions.

Also found and fixed a real ingredient_parser bug while building this: the
duplicate-recipe-name-as-first-line check compared strings exactly, so a
name like "새우 두부 계란찜" (spaced) vs. the ingredients_raw first line
"새우두부계란찜" (glued, no spaces — a real data inconsistency) wouldn't
match, letting the glued name slip in as a bogus "ingredient" that could
spuriously match unrelated searches (e.g. it contains both "두부" and
"계란" as substrings). Fixed by comparing with whitespace stripped
(`_normalize_spacing`). The parser also now:
- treats unicode fractions (½⅓⅔¼¾…) as quantity starts. "게살(½컵)" used to
  keep its amount in the name; 25 names in the DB were affected.
- drops exact-match cookware names (`_COOKWARE`: 뚝배기, 꼬치, 랩…). The
  crawled recipes sometimes list them, and a 2,100원 pot once got priced into
  a 계란찜.

The spacing fix affected 3/1301 recipes — rebuild the ingredient
index after pulling this fix.

**Resolved**: LLM-based correction for ambiguous ingredient names.
`ingredient_parser.py` itself got two real bug fixes first (found by scanning
the DB for 3+-word parsed names, most of which turned out fixable with
plain rules): `●`/`•` bullet markers weren't in the label-strip charset, and
the colon-label regex didn't allow spaces in the label (so "치커리 샐러드 :
치커리" wasn't recognized as a label). That dropped 3+-word names from 309 to
130 across the dataset.

For what's left, checking live search results showed the "empty results ⇒
ambiguous" assumption was wrong: several bad names (e.g. "고기 삶는 재료
양파", "톳 무침양념 설탕") return a *confidently wrong* product instead of
nothing — so correction can't be a fallback that only fires on empty
results. Design: `pricing/price_lookup._looks_ambiguous()` flags any parsed
name with 3+ words and runs it through `agent/ingredient_correction.
correct_ingredient_name()` (an LLM tool-call) *before* searching; a plain
empty-result fallback also still triggers correction for shorter names that
slip through. The correction tool can also report "this isn't a real
purchasable ingredient" (verified against "간 맞출 때", a cooking
instruction the parser had mistakenly captured as an ingredient — correctly
returned `None` rather than searching for it). `correct_name` is injected
into `estimate_recipe_price()` as a callback rather than importing the LLM
client directly, so `pricing/` stays LLM-free and unit-testable without
mocking chat.

**Model reliability note**: qwen3:14b needed `temperature=0.0` for this
(higher values gave inconsistent results run-to-run on the *identical*
input, including once wrongly flagging "설탕"/sugar as not-purchasable at
temperature 0.2 — that specific error disappeared at 0.0). Even at
temperature 0, tested against 9 real ambiguous names from the DB, it
correctly cleaned ~5/9 and left ~4/9 unchanged rather than inventing a wrong
correction. Treated as an acceptable ceiling rather than something to keep
prompt-tuning — see `ingredient_correction.correct_ingredient_name`'s
docstring. Don't re-attempt few-shot/prompt iteration expecting much more
from this model size; a bigger model or a different technique would be the
next lever, not more prompt tweaking.

Also confirmed the result is context-sensitive, not just per-input
deterministic: 묵은지가지말이's "고기 삶는 재료 양파" returned "양파"
against a short hand-written test context, but unchanged against that
recipe's real (longer) `ingredients_raw` as context, at temperature=0 both
times. Ran end-to-end through `estimate_recipe_price()` on the real recipe:
the unchanged name matched a bay-leaf product ("...고기삶을때..." in its
listing text coincidentally matched), a confidently-wrong price rather than
no price. Not a new regression — naive keyword search already had this
failure mode before this feature existed (e.g. "물" matching bottled water);
this feature only ever reduces how often it happens, never increases it.

Known gap: corrections aren't cached — only successful price lookups are
(under the *original* ambiguous name, so a cache hit does skip re-correcting
on repeat). A recipe with several ambiguous names not yet in cache means
several extra LLM calls on top of the existing per-recipe latency. Not fixed
now; revisit if this becomes the dominant cost once used day to day.

**Servings count**: `Recipe.servings` (a computed property in `data/models.py`,
not a stored column) parses a leading "[N인분]" marker from `ingredients_raw`
when present. Only ~3.5% of recipes (40/1156) state this — no other field in
COOKRCP01 reliably gives a serving count (`INFO_WGT` is grams *per* serving,
not how many servings). Returns None otherwise; `agent/tools.format_recipe_detail` explicitly tells the model not to guess or
mention a serving count when it's None, verified against both a recipe that
has one (reported "(2인분)") and one that doesn't (reported "제공되지
않음", didn't fabricate a number).
**Resolved**: the "total price overstates real cost" quirk noted below was
addressed by adding a portioned-cost estimate alongside the full purchase
price (`pricing/unit_parser.py` + `IngredientPrice.portioned_cost` /
`RecipePriceEstimate.total_portioned_cost` in `pricing/price_lookup.py`).
`parse_quantity()` extracts a weight (g/kg) or volume (ml/L) from free text;
when both the recipe's needed amount and the product title's package size
parse to the *same* unit (both weight or both volume), portioned cost =
package price × (recipe amount / package amount). Deliberately not attempted
for count-based amounts ("1개", "5구") or vague ones ("약간", "적당량"), or
when the two sides are different unit kinds (e.g. recipe needs grams of
sesame oil but the product is sold in ml) — no unit conversion between
weight and volume is done, even though it's sometimes physically possible
(density), because that's guessing, not parsing. Ingredients this couldn't
be computed for still show their full purchase price, just no portioned
figure; `ingredients_missing_portioned_cost` lists which ones. Verified
against the real DB: for 닭고기김치찌개, total purchase price was 51,927원
but total portioned cost was only 5,074원 — 5 of 18 ingredients (물, 참기름,
청주 — unit mismatch; 청양고추 — no size in title; 달걀 — sold by count)
couldn't get a portioned figure, which is expected and surfaced to the user
rather than silently dropped.
Latency: a recipe with ~18 uncached ingredients takes ~100s end-to-end. The
enuri 1 req/sec throttle dominates, plus two sequential local-LLM calls.
Parallelizing price lookups would violate the crawl-delay's intent, so the
fix was UI-side instead:
- **Progress display**: `handle_message(on_progress=...)` reports short
  Korean status lines — "요청 이해하는 중", "'X' 레시피 찾는 중",
  "재료 가격 조회 중 (3/9 · 두부)" via `estimate_recipe_price(on_progress=...)`,
  and "답변 작성 중". Each tool in `agent/tools.run_tool` reports its own.
  - `progress.run_with_progress()` runs the turn on a worker thread and
    yields the latest status, re-yielding every second while idle.
  - `app.run_turn` shows it as a placeholder assistant bubble
    ("⏳ … · 42초") and replaces it with the real reply when done.
- **Cold start**: timing the stages exposed a ~40s "레시피 찾는 중" on the
  first request after startup. That was BGE-M3 loading lazily on the first
  search. It's now preloaded on a background thread at startup
  (`BGEEmbeddingFunction.load()`, locked against a concurrent first
  request). Measured on a first request after startup:
  - Recipe search took 0.2s (was ~40s).
  - The remaining time is the price lookups for uncached ingredients and
    the LLM compose step.
  - A fully cached repeat request took 29.5s.
- Not addressed: Ollama unloads qwen3:14b after idle (default keep_alive ~5
  min), so the first LLM call after a break also pays a model load.

## Voice I/O (`voice/`)

Both STT and TTS run locally on CPU. Open verification items for the user
live in `TODO.md`.

- **STT**: faster-whisper, CPU int8 (`voice/stt.py`). Ported from the user's
  `C:\Project_Files\speech_to_text` project, which already measured this PC
  (small 1.1s / medium 2.8s / large-v3-turbo 4.3s from end of speech to
  text). No usable AMD GPU path: CTranslate2's Windows ROCm builds crash on
  RDNA4, and whisper.cpp's Vulkan route needs a fragile source build on
  Windows. Default `medium`: a round-trip test (Supertonic-generated Korean →
  VAD → STT) had small mishear "마라탕" and turbo drop "찌개" from
  "김치찌개", while medium got all three right. That's synthetic speech, so
  the pick still needs confirming with a real voice.
- **VAD** (`voice/vad.py`): the reference project's energy-based segmenter,
  rewritten push-style so the browser stream and the PC mic both use it.
  Tuned constants were copied from the reference project's settings.json.
- **Mic sources**, selectable in the UI: browser (`gr.Audio(streaming=True)`,
  so it keeps working once the app is reached remotely via Tailscale) or PC
  mic (`voice/pc_mic.py`, sounddevice + device picker).
- **`VoiceController`** (`voice/controller.py`) is one shared instance per
  app, not per session. That's fine because this is a single-user app and
  the PC mic is a single device.
  - Recognized text goes into an inbox, and a 0.5s `gr.Timer` pops it and
    runs a chat turn.
  - It's half-duplex: mic input is dropped while a turn is processing and
    until the spoken reply should have finished. That duration comes from
    the TTS audio length plus a 0.8s margin. Without this, the speakers feed
    the mic and the bot answers itself.
- **TTS**: Supertonic (`voice/tts.py`), ONNX and CPU-only. It was chosen
  because qwen3:14b already occupies ~9.6GB of the 16GB VRAM.
  - Speed here: a 13s sentence synthesizes in ~0.6s on supertonic-2 and ~2s
    on supertonic-3. Default is supertonic-2 / voice F1, pending the user's
    listening check.
  - Rejected candidates: Kokoro and Piper have no Korean voice; Zonos is
    Linux/NVIDIA-only; Qwen3-TTS, CosyVoice3, Chatterbox and S1-mini are
    CUDA-oriented.
  - MeloTTS (Korean, MIT) is the fallback if Supertonic's quality isn't good
    enough, with the caveat that its mecab dependencies often conflict on
    Windows.
- **What gets spoken**: `handle_message()` returns a `Reply(text, speech)`.
  - Every reply speaks its full text with markdown and emoji stripped
    (`to_speech_text`). There used to be a fixed recipe template
    (`recipe_speech_summary`). It was dropped with the tool refactor, at the
    user's choice: replies now fit the question, so they're usually short
    enough to read in full.
  - A future real-time cooking-assistant mode is expected to read full
    (short) replies.
- **Verified end-to-end against the running app via `gradio_client`**: a
  Supertonic-generated wav was posted to the browser-mic stream endpoint,
  then `/poll_voice` triggered the turn.
  - STT produced the right text, and general_chat called the exact-ingredient
    search tool. Reply plus TTS arrived in 45s.
  - The recipe path via text took 96s and produced a 15s spoken summary.
  - `gradio_client` raises `KeyError: 'process_streaming'` when posting to a
    streaming-input endpoint. That's a client-side limitation: the server
    processes the chunk anyway.
  - Not yet verified with a real voice or real mic in the browser (the
    Chrome extension was disconnected); tracked in `TODO.md`.

## Development environment

- Dependency/venv management: **uv**. Use `uv add <package>` to add dependencies,
  `uv run <cmd>` to run things inside the project venv. Do not use bare `pip`.

## Paid API approval gate

- The local Ollama model and the 에누리 price scraper are free (rate-limited
  only, by our own throttle) and can be used/tested freely without asking.
- Any code path that calls a **paid cloud LLM API** (e.g. Claude, OpenAI, or any
  other billed provider reached through LiteLLM) must get the user's explicit
  approval **before** that test/run happens. Never run a paid-API test
  speculatively "just to check" — ask first, every time, even if a similar call
  was approved earlier in the session.

## Git workflow

- One feature = one branch (branch off `main`, e.g. `feature/recipe-rag-search`).
- Do not commit directly to `main`.
- When a feature is ready, open a PR against `main` on
  https://github.com/jinwoong96/cooking_assistant_chatbot and tell the user
  it's ready for review (summarize what changed and how it was tested).
- Only merge a PR after the user confirms testing is complete. Do not
  self-merge based on passing CI/automated tests alone.
