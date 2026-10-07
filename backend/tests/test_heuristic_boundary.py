"""D-033: heuristics can't leak into scoring. Source scans + schema check.

Each scan is also run on a planted violation, so a scan that silently matches nothing
can't pass as a green test.
"""

import ast
from pathlib import Path

from judge_check.db import Base

SRC = Path(__file__).resolve().parents[1] / "src" / "judge_check"
DIAGNOSTICS_IMPORTERS = {SRC / "cli.py"}  # human-readable summaries only
# guards.py: widened deliberately in D-035. Guards use containment only to WITHHOLD a label
# (necessary conditions); they can never award one.
CONTAINS_ANSWER_USERS = {
    SRC / "datasets" / "hotpotqa.py",
    SRC / "diagnostics.py",
    SRC / "variants" / "guards.py",
}


def real_sources() -> dict[Path, str]:
    return {p: p.read_text() for p in sorted(SRC.rglob("*.py"))}


def imports_diagnostics(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module, *(f"{node.module}.{a.name}" for a in node.names)]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        if any(n == "judge_check.diagnostics" or n.startswith("judge_check.diagnostics.")
               for n in names):  # fmt: skip
            lines.append(node.lineno)
    return lines


def uses_contains_answer(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        hit = (
            (isinstance(node, ast.Name) and node.id == "contains_answer")
            or (isinstance(node, ast.Attribute) and node.attr == "contains_answer")
            or (
                isinstance(node, ast.ImportFrom)
                and any(a.name == "contains_answer" for a in node.names)
            )
        )
        if hit:
            lines.append(node.lineno)
    return lines


def offenders(sources: dict[Path, str], detector, allowed: set[Path]) -> list[str]:
    out = []
    for path, text in sources.items():
        if path in allowed:
            continue
        out += [f"{path.name}:{ln}" for ln in detector(ast.parse(text))]
    return out


def test_only_display_code_imports_diagnostics() -> None:
    assert offenders(real_sources(), imports_diagnostics, DIAGNOSTICS_IMPORTERS) == []


def test_contains_answer_is_confined() -> None:
    assert offenders(real_sources(), uses_contains_answer, CONTAINS_ANSWER_USERS) == []


def test_scans_catch_planted_violations() -> None:
    planted = {
        SRC / "metrics.py": "from judge_check.diagnostics import cites_gold\n",
        SRC / "report.py": "import judge_check.diagnostics as d\n",
        SRC / "scoring.py": (
            "from judge_check.datasets.hotpotqa import contains_answer\n"
            "label = contains_answer(ref, ans)\n"
        ),
        # The widening is one file, not the package: another variants module must still fail.
        SRC / "variants" / "constructors.py": (
            "from judge_check.datasets.hotpotqa import contains_answer\n"
        ),
    }
    assert offenders(planted, imports_diagnostics, DIAGNOSTICS_IMPORTERS) == [
        "metrics.py:1",
        "report.py:1",
    ]
    assert offenders(planted, uses_contains_answer, CONTAINS_ANSWER_USERS) == [
        "scoring.py:1",
        "scoring.py:2",
        "constructors.py:1",
    ]


def test_no_table_stores_a_heuristic() -> None:
    suspicious = ("reference_in", "contains_ref", "cites_gold", "heuristic", "string_match")
    columns = [
        f"{t.name}.{c.name}"
        for t in Base.metadata.tables.values()
        for c in t.columns
        if any(s in c.name for s in suspicious)
    ]
    assert columns == [], f"heuristic values must never be persisted: {columns}"
