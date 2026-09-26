from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    food_safety_api_key: str = "sample"
    db_path: str = "data/app.db"
    chroma_db_path: str = "data/chroma_db"
    embedding_model_name: str = "BAAI/bge-m3"
    llm_model: str = "ollama_chat/qwen3:14b"
    ollama_base_url: str = "http://localhost:11434"
    stt_model: str = "medium"
    """Round-trip test (TTS-generated Korean → VAD → STT) on this PC: small
    1.1s but misheard "마라탕", large-v3-turbo 4.5s but dropped "찌개" from
    "김치찌개", medium 3.1s and got all three right. Selectable in the UI."""
    stt_model_dir: str = "data/models/whisper"
    """faster-whisper download_root. Point it at an existing download (e.g.
    the speech_to_text project's models/hf) to skip re-downloading."""
    tts_model: str = "supertonic-2"
    tts_voice: str = "F1"
    tts_speed: float = 1.05
    """Default speaking rate (Supertonic's own default is 1.05). Adjustable
    live from the UI slider; this is what it starts at."""
    app_username: str = "me"
    app_password: str = ""
    """Empty (default) = no login prompt, for local-only use. Set both this
    and app_username in .env once the app is exposed beyond localhost (e.g.
    via Tailscale) — see CLAUDE.md's deployment notes."""


settings = Settings()
