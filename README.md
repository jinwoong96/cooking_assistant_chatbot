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
