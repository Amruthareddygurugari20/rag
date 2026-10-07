"""The provider interface. Everything that calls an LLM goes through `LLMClient.complete`.

One interface for every provider is what lets stage 4 hold the prompt fixed and vary only
the model: the judge harness never knows which provider it's talking to.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class Completion:
    text: str
    provider: str
    model: str
    latency_ms: int  # wall clock around the HTTP call, measured by us
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None  # 0.0 for local models; None when prices aren't configured
    raw: dict[str, Any] = field(repr=False)  # the provider's full response, kept for audit


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    provider: str
    model: str

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion: ...


# (url, json body, headers, timeout seconds) -> parsed JSON response
Transport = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


def http_post_json(
    url: str, body: dict[str, Any], headers: dict[str, str], timeout: float
) -> dict[str, Any]:
    """Default transport: standard-library HTTP, so providers add no dependencies."""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise LLMError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"cannot reach {url}: {exc.reason}") from exc


def timed(transport: Transport, *args: Any) -> tuple[dict[str, Any], int]:
    t0 = time.perf_counter()
    response = transport(*args)
    return response, round((time.perf_counter() - t0) * 1000)
