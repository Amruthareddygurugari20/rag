"""Ollama (local, free). https://github.com/ollama/ollama/blob/main/docs/api.md

Model identity: an Ollama tag ("qwen2.5:3b") is a moving pointer; `ollama pull` can swap
the weights behind it. The content digest from /api/tags identifies the weights exactly,
so every completion records `tag@digest`. If the digest can't be found, the call is
refused: an unattributable judgement is worse than none (D-030).
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


def _full_tag(model: str) -> str:
    return model if ":" in model else f"{model}:latest"


class OllamaClient:
    provider = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        *,
        timeout: float = 300.0,
        transport: Transport = http_json,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport

    def model_digest(self) -> str:
        """Digest of the local weights for this tag, looked up on every call (cheap and
        local) so a re-pull between calls is visible in the records."""
        tags = self._transport("GET", f"{self.base_url}/api/tags", None, {}, 30.0)
        want = _full_tag(self.model)
        for m in tags.get("models", []):
            if _full_tag(m.get("name", "")) == want and m.get("digest"):
                return m["digest"]
        raise LLMError(f"Ollama has no digest for {want!r}; is it pulled? (`ollama pull`)")

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int = DEFAULT_SEED,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        digest = self.model_digest()
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed},
        }
        if json_schema is not None:
            body["format"] = json_schema  # Ollama constrains decoding to this JSON schema
        raw, ms = timed(
            self._transport, "POST", f"{self.base_url}/api/chat", body, {}, self.timeout
        )
        try:
            text = raw["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"unexpected Ollama response: {str(raw)[:300]}") from exc
        reported = raw.get("model") or self.model
        return Completion(
            text=text,
            provider=self.provider,
            model_requested=self.model,
            model_reported=reported,
            model_version=f"{_full_tag(reported)}@{digest}",
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            latency_ms=ms,
            input_tokens=raw.get("prompt_eval_count"),
            output_tokens=raw.get("eval_count"),
            cost_usd=0.0,  # local inference: no per-token price
            raw=raw,
        )
