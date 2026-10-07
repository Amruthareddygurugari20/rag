"""Providers, tested against a recording fake transport: exact request shape, parsing,
model attribution, costs and errors. No network, no model."""

import pytest

from judge_check.llm.azure import AzureOpenAIClient
from judge_check.llm.base import Completion, LLMError, Message
from judge_check.llm.ollama import OllamaClient

MSGS = [Message("system", "Be brief."), Message("user", "Capital of France?")]
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
DIGEST = "sha256:a1b2c3"


class FakeTransport:
    """Routes GET /api/tags and POST calls to canned responses; records every call."""

    def __init__(self, post: dict, tags: dict | None = None) -> None:
        self.post, self.tags = post, tags
        self.calls: list[tuple] = []

    def __call__(self, method, url, body, headers, timeout):
        self.calls.append((method, url, body, headers, timeout))
        if method == "GET":
            return self.tags or {"models": []}
        return self.post


OLLAMA_TAGS = {"models": [{"name": "qwen2.5:3b", "digest": DIGEST}]}
OLLAMA_OK = {"model": "qwen2.5:3b", "message": {"content": "Paris"}, "prompt_eval_count": 12,
             "eval_count": 3}  # fmt: skip


def test_ollama_request_parse_and_attribution() -> None:
    t = FakeTransport(OLLAMA_OK, OLLAMA_TAGS)
    out = OllamaClient("qwen2.5:3b", "http://localhost:11434/", transport=t).complete(
        MSGS, temperature=0.0, max_tokens=64, seed=7, json_schema=SCHEMA
    )
    (m1, url1, *_), (m2, url2, body, headers, _) = t.calls
    assert (m1, url1) == ("GET", "http://localhost:11434/api/tags")
    assert (m2, url2) == ("POST", "http://localhost:11434/api/chat")
    assert body["model"] == "qwen2.5:3b" and body["stream"] is False
    assert body["options"] == {"temperature": 0.0, "num_predict": 64, "seed": 7}
    assert body["format"] == SCHEMA and headers == {}
    assert out.model_version == f"qwen2.5:3b@{DIGEST}"
    assert (out.model_requested, out.model_reported) == ("qwen2.5:3b", "qwen2.5:3b")
    assert (out.temperature, out.seed, out.max_tokens) == (0.0, 7, 64)
    assert (out.text, out.input_tokens, out.output_tokens, out.cost_usd) == ("Paris", 12, 3, 0.0)


def test_ollama_always_sends_a_seed() -> None:
    t = FakeTransport(OLLAMA_OK, OLLAMA_TAGS)
    out = OllamaClient("qwen2.5:3b", transport=t).complete(MSGS)
    assert t.calls[1][2]["options"]["seed"] == 0 and out.seed == 0
    assert "format" not in t.calls[1][2]


def test_ollama_untagged_name_matches_latest() -> None:
    tags = {"models": [{"name": "llama3:latest", "digest": DIGEST}]}
    t = FakeTransport({"model": "llama3", "message": {"content": "x"}}, tags)
    assert OllamaClient("llama3", transport=t).complete(MSGS).model_version == (
        f"llama3:latest@{DIGEST}"
    )


def test_ollama_refuses_without_digest() -> None:
    t = FakeTransport(OLLAMA_OK, {"models": [{"name": "other:1b", "digest": DIGEST}]})
    with pytest.raises(LLMError, match="no digest"):
        OllamaClient("qwen2.5:3b", transport=t).complete(MSGS)
    assert all(c[0] == "GET" for c in t.calls)  # refused before generating anything


def test_ollama_malformed_response_raises() -> None:
    t = FakeTransport({"error": "boom"}, OLLAMA_TAGS)
    with pytest.raises(LLMError, match="unexpected Ollama response"):
        OllamaClient("qwen2.5:3b", transport=t).complete(MSGS)


def azure(t, **kw):
    return AzureOpenAIClient("gpt-judge", "https://x.openai.azure.com/", "KEY", transport=t, **kw)


AZURE_OK = {
    "model": "gpt-4o-2024-08-06",
    "system_fingerprint": "fp_abc123",
    "choices": [{"message": {"role": "assistant", "content": "Paris"}}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
}


def test_azure_request_attribution_and_cost() -> None:
    t = FakeTransport(AZURE_OK)
    out = azure(t, usd_per_1m_input=2.0, usd_per_1m_output=8.0).complete(
        MSGS, max_tokens=64, seed=1, json_schema=SCHEMA
    )
    method, url, body, headers, _ = t.calls[0]
    assert method == "POST"
    assert url == (
        "https://x.openai.azure.com/openai/deployments/gpt-judge/chat/completions"
        "?api-version=2024-10-21"
    )
    assert headers == {"api-key": "KEY"}
    assert body["max_tokens"] == 64 and body["seed"] == 1
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert out.model_requested == "gpt-judge"  # our deployment name
    assert out.model_reported == "gpt-4o-2024-08-06"  # what actually answered
    assert out.model_version == "gpt-4o-2024-08-06+fp_abc123"
    assert out.cost_usd == pytest.approx((1000 * 2.0 + 500 * 8.0) / 1e6)


def test_azure_cost_is_none_without_prices() -> None:
    assert azure(FakeTransport(AZURE_OK)).complete(MSGS).cost_usd is None


def test_azure_refuses_response_without_model() -> None:
    no_model = {k: v for k, v in AZURE_OK.items() if k != "model"}
    with pytest.raises(LLMError, match="no 'model'"):
        azure(FakeTransport(no_model)).complete(MSGS)


def test_azure_requires_credentials() -> None:
    with pytest.raises(ValueError, match="endpoint and an API key"):
        AzureOpenAIClient("d", "", "", transport=FakeTransport({}))


def test_azure_malformed_response_raises() -> None:
    with pytest.raises(LLMError, match="unexpected Azure OpenAI response"):
        azure(FakeTransport({"choices": []})).complete(MSGS)


@pytest.mark.parametrize("missing", ["model_version", "model_reported", "provider"])
def test_completion_cannot_exist_without_attribution(missing: str) -> None:
    fields = dict(
        text="x", provider="ollama", model_requested="m", model_reported="m",
        model_version="m@sha256:1", temperature=0.0, seed=0, max_tokens=8, latency_ms=1,
        input_tokens=None, output_tokens=None, cost_usd=None, raw={},
    )  # fmt: skip
    fields[missing] = ""
    with pytest.raises(ValueError, match=missing):
        Completion(**fields)
