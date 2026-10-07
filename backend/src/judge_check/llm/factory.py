"""Build an LLMClient from settings: `ollama` (default) or `azure_openai`."""

from __future__ import annotations

from judge_check.config import get_settings
from judge_check.llm.azure import AzureOpenAIClient
from judge_check.llm.base import LLMClient
from judge_check.llm.ollama import OllamaClient

PROVIDERS = ("ollama", "azure_openai")


def make_client(provider: str, model: str) -> LLMClient:
    s = get_settings()
    if provider == "ollama":
        return OllamaClient(model, s.ollama_base_url)
    if provider == "azure_openai":
        # model = the Azure deployment name
        return AzureOpenAIClient(
            model,
            s.azure_openai_endpoint,
            s.azure_openai_api_key,
            api_version=s.azure_openai_api_version,
        )
    raise ValueError(f"unknown provider {provider!r}; choose from {PROVIDERS}")
