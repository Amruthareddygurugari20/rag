"""The single path from a prompt to a recorded LLM call (D-031).

Generation (stage 2) and judging (stage 4) both go through `run_prompt`, and only here is
`LLMClient.complete` called; tests/test_llm_runner.py enforces that by scanning the source.
So two judges can differ only in the client passed in: the prompt text, how it's rendered,
the decoding parameters and what gets recorded are the same code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from judge_check.llm.base import DEFAULT_SEED, Completion, LLMClient, Message
from judge_check.prompts import PromptTemplate


@dataclass(frozen=True)
class CallRecord:
    prompt: PromptTemplate
    messages: list[Message]
    completion: Completion


def run_prompt(
    client: LLMClient,
    prompt: PromptTemplate,
    variables: dict[str, str],
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    seed: int = DEFAULT_SEED,
    json_schema: dict[str, Any] | None = None,
) -> CallRecord:
    messages = prompt.render(**variables)
    completion = client.complete(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        json_schema=json_schema,
    )
    return CallRecord(prompt, messages, completion)
