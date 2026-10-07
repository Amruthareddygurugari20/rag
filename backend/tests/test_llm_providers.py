"""Providers, tested against a recording fake transport: exact request shape, parsing,
costs and errors. No network, no model."""

import pytest

from judge_check.llm.azure import AzureOpenAIClient
from judge_check.llm.base import LLMError, Message
from judge_check.llm.ollama import OllamaClient

MSGS = [Message("system", "Be brief."), Message("user", "Capital of France?")]
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}


class FakeTransport:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[tuple] = []

    def __call__(self, url, body, headers, timeout):
        self.calls.append((url, body, headers, timeout))
        return self.response


def test_ollama_request_and_parse() -> None:
    t = FakeTransport(
        {"message": {"role": "assistant", "content": "Paris"}, "prompt_eval_count": 12,
         "eval_count": 3}
    )  # fmt: skip
    c = OllamaClient("qwen2.5:3b", "http://localhost:11434/", transport=t)
    out = c.complete(MSGS, temperature=0.0, max_tokens=64, seed=7, json_schema=SCHEMA)
    url, body, headers, _ = t.calls[0]
    assert url == "http://localhost:11434/api/chat"
    assert body["model"] == "qwen2.5:3b" and body["stream"] is False
    assert body["messages"][1] == {"role": "user", "content": "Capital of France?"}
    assert body["options"] == {"temperature": 0.0, "num_predict": 64, "seed": 7}
    assert body["format"] == SCHEMA
    assert headers == {}
    assert (out.text, out.input_tokens, out.output_tokens, out.cost_usd) == ("Paris", 12, 3, 0.0)
    assert out.provider == "ollama" and out.latency_ms >= 0


def test_ollama_omits_optional_fields() -> None:
    t = FakeTransport({"message": {"content": "x"}})
    OllamaClient("m", transport=t).complete(MSGS)
    body = t.calls[0][1]
    assert "format" not in body and "seed" not in body["options"]


def test_ollama_malformed_response_raises() -> None:
    with pytest.raises(LLMError, match="unexpected Ollama response"):
        OllamaClient("m", transport=FakeTransport({"error": "model not found"})).complete(MSGS)


def azure(t, **kw):
    return AzureOpenAIClient("gpt-judge", "https://x.openai.azure.com/", "KEY", transport=t, **kw)


AZURE_OK = {
    "choices": [{"message": {"role": "assistant", "content": "Paris"}}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
}


def test_azure_request_and_cost() -> None:
    t = FakeTransport(AZURE_OK)
    out = azure(t, usd_per_1m_input=2.0, usd_per_1m_output=8.0).complete(
        MSGS, max_tokens=64, seed=1, json_schema=SCHEMA
    )
    url, body, headers, _ = t.calls[0]
    assert url == (
        "https://x.openai.azure.com/openai/deployments/gpt-judge/chat/completions"
        "?api-version=2024-10-21"
    )
    assert headers == {"api-key": "KEY"}
    assert body["max_tokens"] == 64 and body["seed"] == 1
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert out.cost_usd == pytest.approx((1000 * 2.0 + 500 * 8.0) / 1e6)


def test_azure_cost_is_none_without_prices() -> None:
    assert azure(FakeTransport(AZURE_OK)).complete(MSGS).cost_usd is None


def test_azure_requires_credentials() -> None:
    with pytest.raises(ValueError, match="endpoint and an API key"):
        AzureOpenAIClient("d", "", "", transport=FakeTransport({}))


def test_azure_malformed_response_raises() -> None:
    with pytest.raises(LLMError, match="unexpected Azure OpenAI response"):
        azure(FakeTransport({"choices": []})).complete(MSGS)
