"""Which generated answers may enter the evaluation set (D-032).

Generated answers exist in the design to bring *realistic, fluent* RAG output into the
evaluation set, so judges are measured on the kind of text they face in production. A
model too small to produce plausible answers defeats that: its answers are trivially
rejectable, the human-labelled arm looks *easier* than the constructed arm, and the
realism gap D-032 measures comes out backwards.

So eligibility is an explicit allowlist (settings `eval_generator_models`), with a hard
deny-list of smoke-test models that no configuration can override. Stage 5's adjudication
queue must filter generated answers through `is_eval_eligible`; it is the only gate.
"""

from __future__ import annotations

from collections.abc import Iterable

# Used in CI to prove the Ollama path works end to end. Never a source of eval answers.
SMOKE_TEST_MODELS = frozenset({"qwen2.5:0.5b"})


def _tag(model: str) -> str:
    return model if ":" in model else f"{model}:latest"


def is_eval_eligible(model_requested: str, allowlist: Iterable[str]) -> bool:
    """True only for an allowlisted, non-smoke-test generator model."""
    tag = _tag(model_requested)
    if tag in {_tag(m) for m in SMOKE_TEST_MODELS}:
        return False
    return tag in {_tag(m) for m in allowlist}
