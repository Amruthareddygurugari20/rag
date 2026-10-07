"""Ollama (local, free). https://github.com/ollama/ollama/blob/main/docs/api.md (/api/chat)."""

from __future__ import annotations

from typing import Any

from judge_check.llm.base import Completion, LLMError, Message, Transport, http_post_json, timed


class OllamaClient:
    provider = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        *,
        timeout: float = 300.0,
        transport: Transport = http_post_json,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        options: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
        if seed is not None:
            options["seed"] = seed
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": options,
        }
        if json_schema is not None:
            body["format"] = json_schema  # Ollama constrains decoding to this JSON schema
        raw, ms = timed(self._transport, f"{self.base_url}/api/chat", body, {}, self.timeout)
        try:
            text = raw["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"unexpected Ollama response: {str(raw)[:300]}") from exc
        return Completion(
            text=text,
            provider=self.provider,
            model=self.model,
            latency_ms=ms,
            input_tokens=raw.get("prompt_eval_count"),
            output_tokens=raw.get("eval_count"),
            cost_usd=0.0,  # local inference: no per-token price
            raw=raw,
        )
