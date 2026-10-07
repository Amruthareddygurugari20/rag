"""Prompt registry: released prompts are immutable and rendering is strict."""

import json
import shutil
from pathlib import Path

import pytest

import judge_check.prompts as prompts
from judge_check.prompts import all_prompts, load_prompt, lock_problems


def test_repository_prompts_match_the_lock() -> None:
    # Fails if a released prompt was edited in place, or a new one wasn't locked.
    assert lock_problems() == []


def test_render_is_strict() -> None:
    p = load_prompt("grounded_answer", 1)
    assert p.placeholders() == {"context", "question"}
    msgs = p.render(context="[C1] Paris is in France.", question="Where is Paris?")
    assert [m.role for m in msgs] == ["system", "user"]
    assert "[C1] Paris is in France." in msgs[1].content
    with pytest.raises(ValueError, match="expected variables"):
        p.render(context="x")  # missing
    with pytest.raises(ValueError, match="expected variables"):
        p.render(context="x", question="y", extra="z")  # unexpected


@pytest.fixture
def scratch_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shutil.copytree(prompts.PROMPT_DIR, tmp_path / "p", ignore=shutil.ignore_patterns("*.py"))
    monkeypatch.setattr(prompts, "PROMPT_DIR", tmp_path / "p")
    monkeypatch.setattr(prompts, "LOCK_FILE", tmp_path / "p" / "PROMPTS.lock.json")
    return tmp_path / "p"


def test_editing_a_released_prompt_is_detected(scratch_prompts: Path) -> None:
    f = scratch_prompts / "generation" / "grounded_answer.v1.toml"
    f.write_text(f.read_text().replace("one or two sentences", "three sentences"))
    assert any("was edited; create v2" in p for p in lock_problems())


def test_adding_to_the_lock_never_rewrites_existing_entries(scratch_prompts: Path) -> None:
    f = scratch_prompts / "generation" / "grounded_answer.v1.toml"
    f.write_text(f.read_text() + "\n# tampered\n")
    before = json.loads(prompts.LOCK_FILE.read_text())
    assert prompts.add_new_prompts_to_lock() == []
    assert json.loads(prompts.LOCK_FILE.read_text()) == before  # still the original hash
    assert any("was edited" in p for p in lock_problems())


def test_new_version_must_be_locked(scratch_prompts: Path) -> None:
    src = scratch_prompts / "generation" / "grounded_answer.v1.toml"
    (scratch_prompts / "generation" / "grounded_answer.v2.toml").write_text(
        src.read_text().replace("version = 1", "version = 2")
    )
    assert lock_problems() == ["grounded_answer.v2: not in the lock; release it with "
                               "`judge-check prompts lock`"]  # fmt: skip
    assert prompts.add_new_prompts_to_lock() == ["grounded_answer.v2"]
    assert lock_problems() == []


def test_file_name_must_match_name_and_version(scratch_prompts: Path) -> None:
    src = scratch_prompts / "generation" / "grounded_answer.v1.toml"
    shutil.copy(src, scratch_prompts / "generation" / "misnamed.v9.toml")
    with pytest.raises(ValueError, match="must be named grounded_answer.v1.toml"):
        all_prompts()


def test_literal_braces_survive_rendering() -> None:
    # The JSON example in the instructions must render as literal braces, not placeholders.
    sys_msg = load_prompt("grounded_answer", 1).render(context="c", question="q")[0].content
    assert '{"answer": string, "citations": [label, ...], "answerable": boolean}' in sys_msg
