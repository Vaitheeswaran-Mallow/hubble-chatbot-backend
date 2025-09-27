from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    open_api_key: str
    ws_upstream_url: str | None = None
    database_url: str = "sqlite:///./hubble.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


config = Settings()
