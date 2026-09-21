# Cooking Assistant Chatbot

## Project overview

A personal-use Korean-language chatbot for people living alone. The user names a
dish in natural language; the app finds a recipe and looks up the cheapest price
for each ingredient so the user can cook it affordably. This is also a learning
project for building a local-first RAG + tool-calling AI agent.

Core flow (implemented in `agent/router.py` + `agent/pipeline.py`): user
message -> intent router (single LLM tool-call that both classifies
recipe-request vs. general-chat *and*, for recipe requests, extracts the menu
name in one shot) -> RAG recipe search -> ingredient extraction/normalization
(rule-based, with LLM fallback for ambiguous cases still deferred) ->
lookup of ingredient prices by scraping 에누리(enuri.com) -> a final LLM call
composes the reply from that data. general_chat is a real (if simple) LLM
passthrough already, not a stub — only *specialized* general-conversation
handling (e.g. substitution advice grounded in the recipe DB) is deferred.

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
  **Known quirk**: at default settings, qwen3:14b sometimes ignores a long
  data-summarization prompt and free-associates a generic reply instead of
  using the provided recipe/price data — reproduced once, not consistently.
  Fixed for the recipe-compose step in `agent/pipeline.py` with (a) an
  explicit system prompt stating this is a data-grounded reply, not the
  start of a conversation, and telling it not to ask the user questions back,
  and (b) `temperature=0.3` instead of the default. Re-tested 3x after the
  fix, all correctly formatted. If a similar ignore-the-context failure shows
  up elsewhere, apply the same fix rather than re-diagnosing from scratch.
- Vector store: Chroma, embeddings: BGE-M3
- Structured data: SQLite
- Recipe data: public datasets first (식약처 COOKRCP01, 농식품 공공데이터 레시피 API),
  supplemented only by small-scale, personal-use crawling if needed (사이트
  약관/robots.txt 확인 후 소량만; see prior research on legal risk before adding
  any crawler)
- Deployment: localhost only for now; if remote access is needed later, add
  Tailscale + Gradio `auth=` rather than redesigning anything
- Ingredient price lookup: scrapes **에누리(enuri.com)** search results
  (`EnuriClient`, `pricing/enuri_client.py`), reading the page's
  `<script type="application/ld+json">` schema.org block (stable, semantic —
  not the React/Next.js CSS-module markup, which changes every redeploy).
  Results are cached in SQLite (`price_cache` table, 24h TTL, see
  `pricing/cache.py`) so repeat lookups don't re-hit the site. The client
  self-throttles to >=1 req/sec per price.enuri.com's robots.txt Crawl-delay.
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

Deferred to later phases (not in MVP): graph-DB-based ingredient/recipe
relationship search, general free-form cooking conversation (router exists,
handler logic doesn't yet), larger-scale crawling, multi-user auth, LLM-based
correction pass for ambiguous ingredient names (current parser is rule-based
only; see `pricing/ingredient_parser.py` docstring for known edge cases).
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
Known latency quirk: a recipe with ~18 uncached ingredients took ~100s+
end-to-end in real browser testing (enuri's 1 req/sec throttle dominates,
plus two sequential local-LLM calls). Not fixed for MVP — parallelizing
price lookups would violate the crawl-delay's intent even across multiple
client instances, so the fix, if pursued, should be UI-side (streaming/
progress indication) rather than trying to go faster.

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
