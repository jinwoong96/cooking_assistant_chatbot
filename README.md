# cooking_assistant_chatbot

자취생을 위한 개인용 요리 챗봇. 메뉴명을 말하면 레시피를 찾아주고, 레시피에 들어가는
재료들의 최저가를 조회해서 합리적인 가격에 직접 해먹을 수 있도록 도와줍니다.
로컬 LLM 기반 RAG + tool calling 에이전트를 학습하기 위한 개인 프로젝트이기도 합니다.

설계/개발 규칙은 `CLAUDE.md` 참고.

## 개발 환경

```bash
uv sync
```

## 레시피 데이터 수집

`.env.example`을 `.env`로 복사한 뒤, [식품안전나라 OpenAPI](https://various.foodsafetykorea.go.kr)에서
발급받은 인증키를 `FOOD_SAFETY_API_KEY`에 넣으세요. 키가 없어도 `sample` 값으로 동작 확인은
가능하지만, 실제 전체 레시피(1000건+)를 받으려면 개인 키가 필요합니다.

```bash
uv run python -m cooking_assistant_chatbot.data.ingest
```

`data/app.db` (SQLite)에 레시피가 적재됩니다.

## 레시피 데이터 보완 (선택)

식약처 데이터가 저염식/건강식 위주라, 자취생 일상 메뉴(떡볶이·라면·카레 등)를
[만개의레시피](https://www.10000recipe.com)에서 소량 보완 수집합니다 (개인 비상업 용도,
robots.txt 준수, 초당 1회로 자체 제한 — 자세한 내용은 `CLAUDE.md` 참고).

```bash
uv run python -m cooking_assistant_chatbot.data.crawl_supplemental_recipes
```

## 레시피 검색 인덱스 빌드

```bash
uv run python -m cooking_assistant_chatbot.rag.build_index
uv run python -m cooking_assistant_chatbot.data.build_ingredient_index
```

두 번째 명령은 "냉장고에 있는 재료로 뭐 해먹지?" 같은 질문에 재료를 정확히
매칭해서 검색하기 위한 색인입니다. 레시피 데이터가 바뀔 때마다 둘 다 다시
실행하세요.

## 챗봇 실행

미리 준비할 것:
1. [Ollama](https://ollama.com) 설치 후 `ollama pull qwen3:14b`
2. 위 레시피 수집 + 인덱스 빌드 완료

```bash
uv run python -m cooking_assistant_chatbot.app
```

`http://127.0.0.1:7860`에서 접속할 수 있습니다.
