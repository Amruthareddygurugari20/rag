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
    # Default generation model (stage 2). 7B: the smallest size we *expect* to produce
    # plausible RAG answers. Expectation only: eligibility is measured (D-032 A).
    generation_model: str = "qwen2.5:7b"
    # Convenience default for which generators stage 5 samples for plausibility review.
    # Not a gate: only generation_policy.is_eval_eligible (measured plausibility) is.
    candidate_generator_models: list[str] = ["qwen2.5:7b", "llama3.1:8b"]
    # Optional Azure OpenAI provider (D-011, D-029). Unset = not available.
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-10-21"


@lru_cache
def get_settings() -> Settings:
    return Settings()
