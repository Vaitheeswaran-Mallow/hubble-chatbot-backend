from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    open_api_key: str = ""
    database_url: str = "sqlite:///./hubble.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


config = Settings()
