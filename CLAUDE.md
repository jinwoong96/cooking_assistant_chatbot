# Cooking Assistant Chatbot

## Project overview

A personal-use Korean-language chatbot for people living alone. The user names a
dish in natural language; the app finds a recipe and looks up the cheapest price
for each ingredient so the user can cook it affordably. This is also a learning
project for building a local-first RAG + tool-calling AI agent.

Core flow: user message -> intent router (recipe/price request vs. general chat,
only the router is built for MVP) -> RAG recipe search -> ingredient extraction/
normalization (rule-based, with LLM fallback for ambiguous cases) -> tool-calling
lookup of ingredient prices via the 11번가 Open API -> combined response.

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
- Ingredient price lookup: **11번가 Open API** (`ElevenStClient`). The original
  plan was the Naver Shopping search API, but it was officially shut down
  2026-07-31 (confirmed via developers.naver.com notice #32564) with no
  replacement. Other candidates considered and rejected for now:
  - 쿠팡파트너스 API: requires generating 150,000원 in actual affiliate sales
    before the API is even activated, plus a 10 calls/hour, 10 items/call
    limit once approved — not usable for a not-yet-launched personal project.
  - 다나와/에누리 등 가격비교 사이트 크롤링: no official API, would need the
    same kind of ToS/robots.txt legal review already done for recipe crawling
    (commerce data, likely higher legal sensitivity than recipe text).
  Revisit these if 11번가 turns out to be too limited once actually used —
  don't re-research from scratch, this section already has the tradeoffs.

Deferred to later phases (not in MVP): graph-DB-based ingredient/recipe
relationship search, general free-form cooking conversation (router exists,
handler logic doesn't yet), larger-scale crawling, multi-user auth, LLM-based
correction pass for ambiguous ingredient names (current parser is rule-based
only; see `pricing/ingredient_parser.py` docstring for known edge cases).

## Development environment

- Dependency/venv management: **uv**. Use `uv add <package>` to add dependencies,
  `uv run <cmd>` to run things inside the project venv. Do not use bare `pip`.

## Paid API approval gate

- The local Ollama model and the 11번가 Open API are free (rate-limited only)
  and can be used/tested freely without asking.
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
