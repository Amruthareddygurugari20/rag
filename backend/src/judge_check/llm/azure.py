"""Azure OpenAI (optional). Chat Completions REST API on a deployment.

Model identity: the deployment name is ours and stays fixed while Azure upgrades the model
behind it. Every completion records the dated model the response reports, plus its
`system_fingerprint` (backend configuration) when present. A response without a model is
refused (D-030).

Prices are not hard-coded: they change and depend on the contract. Pass them in (USD per
1M tokens) to get a cost per call; without them `cost_usd` is None, never a guess.
"""

from __future__ import annotations

from typing import Any

from judge_check.llm.base import (
    DEFAULT_SEED,
    Completion,
    LLMError,
    Message,
    Transport,
    http_json,
    timed,
)

DEFAULT_API_VERSION = "2024-10-21"


class AzureOpenAIClient:
    provider = "azure_openai"

    def __init__(
        self,
        deployment: str,
        endpoint: str,
        api_key: str,
        *,
        api_version: str = DEFAULT_API_VERSION,
        usd_per_1m_input: float | None = None,
        usd_per_1m_output: float | None = None,
        timeout: float = 120.0,
        transport: Transport = http_json,
    ) -> None:
        if not (endpoint and api_key):
            raise ValueError("Azure OpenAI needs an endpoint and an API key")
        self.model = deployment
        self._url = (
            f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={api_version}"
        )
        self._api_key = api_key
        self._prices = (usd_per_1m_input, usd_per_1m_output)
        self.timeout = timeout
        self._transport = transport

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int = DEFAULT_SEED,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        body: dict[str, Any] = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema, "strict": True},
            }
        raw, ms = timed(
            self._transport, "POST", self._url, body, {"api-key": self._api_key}, self.timeout
        )
        try:
            text = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected Azure OpenAI response: {str(raw)[:300]}") from exc
        reported = raw.get("model")
        if not reported:
            raise LLMError("Azure OpenAI response has no 'model'; cannot attribute it")
        fingerprint = raw.get("system_fingerprint")
        version = f"{reported}+{fingerprint}" if fingerprint else reported
        usage = raw.get("usage") or {}
        n_in, n_out = usage.get("prompt_tokens"), usage.get("completion_tokens")
        p_in, p_out = self._prices
        cost = None
        if None not in (n_in, n_out, p_in, p_out):
            cost = (n_in * p_in + n_out * p_out) / 1_000_000
        return Completion(
            text=text,
            provider=self.provider,
            model_requested=self.model,
            model_reported=reported,
            model_version=version,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            latency_ms=ms,
            input_tokens=n_in,
            output_tokens=n_out,
            cost_usd=cost,
            raw=raw,
        )
