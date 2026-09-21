from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    food_safety_api_key: str = "sample"
    db_path: str = "data/app.db"
    chroma_db_path: str = "data/chroma_db"
    embedding_model_name: str = "BAAI/bge-m3"
    llm_model: str = "ollama_chat/qwen3:14b"
    ollama_base_url: str = "http://localhost:11434"
    app_username: str = "me"
    app_password: str = ""
    """Empty (default) = no login prompt, for local-only use. Set both this
    and app_username in .env once the app is exposed beyond localhost (e.g.
    via Tailscale) — see CLAUDE.md's deployment notes."""


settings = Settings()
