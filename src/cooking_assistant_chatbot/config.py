from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    food_safety_api_key: str = "sample"
    db_path: str = "data/app.db"
    chroma_db_path: str = "data/chroma_db"
    embedding_model_name: str = "BAAI/bge-m3"
    eleven_st_api_key: str = ""


settings = Settings()
