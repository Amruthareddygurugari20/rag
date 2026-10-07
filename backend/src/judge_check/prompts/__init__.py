"""Versioned prompt templates, stored as data, not code (D-031).

Each prompt is a TOML file `<kind>/<name>.v<version>.toml` with `system` and `user`
templates using {named} placeholders. `PROMPTS.lock.json` pins the SHA-256 of every
released file. A test fails if a released file changes, so a prompt edit always means a new
version, and every stored call row (which records name, version and hash) stays
reproducible.
"""

from __future__ import annotations

import hashlib
import json
import string
import tomllib
from dataclasses import dataclass
from pathlib import Path

from judge_check.llm.base import Message

PROMPT_DIR = Path(__file__).resolve().parent
LOCK_FILE = PROMPT_DIR / "PROMPTS.lock.json"


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: int
    kind: str  # "generation" | "judge" | ...
    system: str
    user: str
    sha256: str  # of the file bytes: identifies this exact text

    @property
    def key(self) -> str:
        return f"{self.name}.v{self.version}"

    def placeholders(self) -> set[str]:
        return {
            field
            for part in (self.system, self.user)
            for _, field, _, _ in string.Formatter().parse(part)
            if field
        }

    def render(self, **variables: str) -> list[Message]:
        """Fill the placeholders. Missing *or unexpected* variables are errors: a silently
        ignored variable usually means the caller and the prompt disagree."""
        expected = self.placeholders()
        if set(variables) != expected:
            raise ValueError(
                f"{self.key}: expected variables {sorted(expected)}, got {sorted(variables)}"
            )
        return [
            Message("system", self.system.format(**variables)),
            Message("user", self.user.format(**variables)),
        ]


def _load_file(path: Path) -> PromptTemplate:
    raw = path.read_bytes()
    data = tomllib.loads(raw.decode())
    expected_name = f"{data['name']}.v{data['version']}.toml"
    if path.name != expected_name:
        raise ValueError(f"{path}: file must be named {expected_name}")
    return PromptTemplate(
        name=data["name"],
        version=int(data["version"]),
        kind=data["kind"],
        system=data["system"],
        user=data["user"],
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def all_prompts() -> dict[str, PromptTemplate]:
    out = {}
    for path in sorted(PROMPT_DIR.glob("*/*.toml")):
        p = _load_file(path)
        if p.key in out:
            raise ValueError(f"duplicate prompt {p.key}")
        out[p.key] = p
    return out


def load_prompt(name: str, version: int) -> PromptTemplate:
    prompts = all_prompts()
    key = f"{name}.v{version}"
    if key not in prompts:
        raise KeyError(f"no prompt {key}; have {sorted(prompts)}")
    return prompts[key]


def lock_problems() -> list[str]:
    """Differences between the prompt files and the lock. Empty list = consistent."""
    lock = json.loads(LOCK_FILE.read_text()) if LOCK_FILE.exists() else {}
    prompts = all_prompts()
    problems = []
    for key, sha in lock.items():
        if key not in prompts:
            problems.append(f"{key}: locked but its file is gone (released prompts are kept)")
        elif prompts[key].sha256 != sha:
            problems.append(
                f"{key}: released prompt was edited; create v{prompts[key].version + 1}"
            )
    for key in prompts.keys() - lock.keys():
        problems.append(f"{key}: not in the lock; release it with `judge-check prompts lock`")
    return problems


def add_new_prompts_to_lock() -> list[str]:
    """Lock prompts that aren't locked yet. Never rewrites an existing entry."""
    lock = json.loads(LOCK_FILE.read_text()) if LOCK_FILE.exists() else {}
    added = [k for k in all_prompts() if k not in lock]
    for key in added:
        lock[key] = all_prompts()[key].sha256
    LOCK_FILE.write_text(json.dumps(dict(sorted(lock.items())), indent=2) + "\n")
    return added
