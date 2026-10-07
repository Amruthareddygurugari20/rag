"""The single call path (D-031): run_prompt is the only caller of LLMClient.complete."""

import ast
from pathlib import Path

from judge_check.llm.base import Completion
from judge_check.llm.runner import run_prompt
from judge_check.prompts import load_prompt

SRC = Path(__file__).resolve().parents[1] / "src" / "judge_check"
ALLOWED = {SRC / "llm" / "runner.py"}


def test_complete_is_called_only_from_the_runner() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "complete"
                and path not in ALLOWED
            ):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert offenders == [], f"LLM calls must go through llm/runner.py: {offenders}"


class RecordingClient:
    provider, model = "fake", "fake-model"

    def __init__(self) -> None:
        self.kwargs = None

    def complete(self, messages, **kwargs):
        self.kwargs = {"messages": messages, **kwargs}
        return Completion(
            text='{"answer": "Paris", "citations": ["C1"], "answerable": true}',
            provider="fake", model_requested="fake-model", model_reported="fake-model",
            model_version="fake-model@sha256:0", temperature=kwargs["temperature"],
            seed=kwargs["seed"], max_tokens=kwargs["max_tokens"], latency_ms=1,
            input_tokens=None, output_tokens=None, cost_usd=0.0, raw={},
        )  # fmt: skip


def test_run_prompt_renders_passes_params_and_records() -> None:
    client = RecordingClient()
    p = load_prompt("grounded_answer", 1)
    rec = run_prompt(
        client, p, {"context": "[C1] x", "question": "q?"}, temperature=0.3, max_tokens=99,
        seed=5,
    )  # fmt: skip
    assert client.kwargs["temperature"] == 0.3 and client.kwargs["seed"] == 5
    assert client.kwargs["max_tokens"] == 99
    assert rec.prompt.sha256 == p.sha256 and rec.messages == client.kwargs["messages"]
    assert rec.completion.model_version == "fake-model@sha256:0"
