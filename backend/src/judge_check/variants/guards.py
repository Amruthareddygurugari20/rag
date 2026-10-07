"""Guards that must ALL pass before a mutated variant may be labelled (D-035).

Guards can only *withhold* a label. Nothing here can make a variant correct. This is the one
module outside the dataset builder and diagnostics that may use `contains_answer`
(normalised whole-word containment), deliberately widened in D-035 and checked by
tests/test_heuristic_boundary.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from judge_check.datasets.hotpotqa import contains_answer, normalize_answer
from judge_check.variants.mutate import Substitution
from judge_check.variants.valuetypes import VType, vtype


@dataclass(frozen=True)
class GuardResult:
    name: str
    passed: bool
    detail: str


class GuardFailure(Exception):
    def __init__(self, results: list[GuardResult]) -> None:
        self.results = results
        failed = [r for r in results if not r.passed]
        super().__init__("; ".join(f"{r.name}: {r.detail}" for r in failed))

    @property
    def first_failed(self) -> str:
        return next(r.name for r in self.results if not r.passed)


def mutation_guards(
    sub: Substitution, original: str, kind: VType, gold_texts: list[str]
) -> list[GuardResult]:
    """The five D-035 guards for a value mutation. Runs them all (for the report)."""
    rep = sub.replacement
    gold = " ".join(gold_texts)
    results = [
        GuardResult(
            "different",
            normalize_answer(rep) != normalize_answer(original),
            f"{original!r} -> {rep!r}",
        ),
        GuardResult("same_type", vtype(rep) is kind, f"{kind} -> {vtype(rep)}"),
        GuardResult(
            "absent_from_gold",
            not contains_answer(rep, gold),
            f"{rep!r} {'APPEARS' if contains_answer(rep, gold) else 'does not appear'} "
            "in the gold evidence",
        ),
        GuardResult(
            "original_removed",
            not contains_answer(original, sub.result),
            f"{original!r} {'still present' if contains_answer(original, sub.result) else 'gone'}",
        ),
        GuardResult("rest_unchanged", _only_spans_changed(sub), "text outside replaced spans"),
    ]
    return results


def _only_spans_changed(sub: Substitution) -> bool:
    """Rebuild the result from the source and spans independently, and compare."""
    if not sub.spans:
        return False  # nothing replaced: not a mutation
    pieces, last = [], 0
    for s, e in sub.spans:
        if s < last:
            return False  # overlapping spans
        pieces += [sub.source[last:s], sub.replacement]
        last = e
    pieces.append(sub.source[last:])
    return "".join(pieces) == sub.result


def require(results: list[GuardResult]) -> list[dict]:
    """Raise GuardFailure unless every guard passed; return the report for storage."""
    if not all(r.passed for r in results):
        raise GuardFailure(results)
    return [asdict(r) for r in results]
