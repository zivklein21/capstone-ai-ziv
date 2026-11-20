# src/capstone_ai_ziv/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str = ""
    model_name: str = "claude-sonnet-5"
    max_tokens: int = 4096

    model_config: dict = {"env_file": ".env", "env_prefix": ""}

settings = Settings()