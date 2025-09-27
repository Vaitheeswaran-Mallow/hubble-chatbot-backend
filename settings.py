from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    open_api_key: str
    ws_upstream_url: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


config = Settings()
