from __future__ import annotations

import gradio as gr

from .agent.pipeline import handle_message
from .config import settings
from .data.db import get_connection
from .pricing.enuri_client import EnuriClient
from .rag.embeddings import BGEEmbeddingFunction
from .rag.search import RecipeSearcher


def build_app() -> gr.ChatInterface:
    """Wire up the shared backend resources once and return a Gradio app.

    Local-only prototype UI, per the earlier design decision to keep the
    frontend minimal and spend effort on the AI/agent side instead.
    """
    conn = get_connection(settings.db_path)
    embedding_function = BGEEmbeddingFunction(settings.embedding_model_name)
    searcher = RecipeSearcher(settings.chroma_db_path, settings.db_path, embedding_function)
    price_client = EnuriClient()

    def respond(message: str, history: list) -> str:
        return handle_message(message, searcher, price_client, conn, history=history)

    return gr.ChatInterface(
        fn=respond,
        title="요리 챗봇",
        description="메뉴 이름을 말하면 레시피와 예상 재료비를 알려드려요.",
        examples=["김치찌개 해먹고 싶어", "된장찌개 레시피 알려줘", "떡볶이 어떻게 만들어?"],
    )


def main() -> None:
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()
