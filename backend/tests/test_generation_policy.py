import pytest

from judge_check.config import Settings
from judge_check.generation_policy import SMOKE_TEST_MODELS, is_eval_eligible


def test_smoke_test_model_is_never_eligible_even_if_allowlisted() -> None:
    for m in SMOKE_TEST_MODELS:
        assert not is_eval_eligible(m, allowlist=[m])


@pytest.mark.parametrize(
    ("model", "eligible"),
    [("qwen2.5:7b", True), ("llama3.1:8b", True), ("qwen2.5:3b", False), ("mystery", False)],
)
def test_default_allowlist(model: str, eligible: bool) -> None:
    allow = Settings(_env_file=None).eval_generator_models
    assert is_eval_eligible(model, allow) is eligible


def test_untagged_name_means_latest() -> None:
    assert is_eval_eligible("phi4", ["phi4:latest"])
    assert not is_eval_eligible("phi4", ["phi4:14b"])


def test_default_generation_model_is_eval_eligible() -> None:
    s = Settings(_env_file=None)
    assert is_eval_eligible(s.generation_model, s.eval_generator_models)
