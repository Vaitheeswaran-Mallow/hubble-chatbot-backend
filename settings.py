from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    open_api_key: str
    ws_upstream_url: str | None = None
    database_url: str = "sqlite:///./hubble.db"
    cors_allow_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


config = Settings()
