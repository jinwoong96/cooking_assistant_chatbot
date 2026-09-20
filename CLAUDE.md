# Cooking Assistant Chatbot

## Project overview

A personal-use Korean-language chatbot for people living alone. The user names a
dish in natural language; the app finds a recipe and looks up the cheapest price
for each ingredient so the user can cook it affordably. This is also a learning
project for building a local-first RAG + tool-calling AI agent.

Core flow: user message -> intent router (recipe/price request vs. general chat,
only the router is built for MVP) -> RAG recipe search -> ingredient extraction/
normalization (rule-based, with LLM fallback for ambiguous cases) -> tool-calling
lookup of ingredient prices by scraping 에누리(enuri.com) -> combined response.

Stack decisions:
- Backend: Python (FastAPI)
- Frontend: Gradio (prototype UI)
- LLM: local model (Qwen2.5-14B or Qwen3-14B, Q4_K_M) served via Ollama, sized for
  16GB VRAM (AMD RX 9060 XT). Provider switching (local vs. cloud) goes through
  LiteLLM; RAG/tool-calling/routing orchestration is hand-built, not LangChain/
  LangGraph (deliberately deferred to a future project).
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
Known product-level quirk to revisit: total recipe price sums the cheapest
*purchasable package* for every ingredient (e.g. buying a whole bottle of
water or 200g of garlic to use 10g), which overstates real marginal cost —
fine for MVP, but worth reconsidering (e.g. excluding common pantry staples,
or showing cost-per-recipe-use) once this is actually used day to day.

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
