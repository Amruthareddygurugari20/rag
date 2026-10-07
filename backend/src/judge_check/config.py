"""Runtime configuration, read from environment variables (prefix ``JUDGE_CHECK_``) or a .env file.

Settings are added here stage by stage, only when something reads them.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="JUDGE_CHECK_",
        # Looked up relative to the working directory: works from the repo root or backend/.
        env_file=(".env", "../.env"),
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://judge_check:judge_check@localhost:5433/judge_check"
    test_database_url: str = (
        "postgresql+psycopg://judge_check:judge_check@localhost:5433/judge_check_test"
    )
    ollama_base_url: str = "http://localhost:11434"


@lru_cache
def get_settings() -> Settings:
    return Settings()
