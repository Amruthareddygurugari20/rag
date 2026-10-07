"""Guards that must ALL pass before a mutated variant may be labelled (D-035).

Guards can only *withhold* a label. Nothing here can make a variant correct. This is the one
module outside the dataset builder and diagnostics that may use `contains_answer`
(normalised whole-word containment), deliberately widened in D-035 and checked by
tests/test_heuristic_boundary.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TypeVar

from judge_check.datasets.hotpotqa import contains_answer, normalize_answer
from judge_check.variants.mutate import Substitution
from judge_check.variants.valuetypes import VType, vtype

T = TypeVar("T")


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


# --- containment checks for the other deterministic types (D-035 addendum D) ----------------
# Same rule as above: each can only withhold a label (discard), never award one.


# No generic "mentions(value, text)" is exported: that would be contains_answer under another
# name and would hollow out the source scan. Each helper below has one job, and every job is
# EXCLUSION (drop a sentence, a value or a chunk), never acceptance.


def sentences_not_stating(answer: str, sentences: list[T]) -> list[T]:
    """Type 7: the sentences (objects with .text) that don't state the answer."""
    return [s for s in sentences if not contains_answer(answer, s.text)]  # type: ignore[attr-defined]


def values_apart_from(answer: str, values: list[tuple]) -> list[tuple]:
    """Type 7b: drop (kind, value) pairs that overlap the answer either way round, so a
    mutation of a supporting detail can't touch the answer."""
    return [
        (k, v) for k, v in values
        if not contains_answer(v, answer) and not contains_answer(answer, v)
    ]  # fmt: skip


def chunks_mentioning_none(chunks: list[tuple[int, str]], values: list[str]):
    """Type 9: the (id, text) chunks that mention none of `values`."""
    return [(c, t) for c, t in chunks if not any(contains_answer(v, t) for v in values)]


def presence_guard(name: str, value: str, text: str, *, present: bool) -> GuardResult:
    found = contains_answer(value, text)
    return GuardResult(
        name, found is present, f"{value!r} {'present' if found else 'absent'} (need "
        f"{'present' if present else 'absent'})"
    )  # fmt: skip


def citation_guards(
    cited: list[tuple[int, str]], gold_chunk_ids: set[int], answer: str, gold_titles: list[str]
) -> list[GuardResult]:
    """Type 9: every cited chunk must be outside the gold chunks and must mention neither the
    answer nor any gold entity. The second condition is what makes the citation *wrong* for
    yes/no answers, where answer absence alone says nothing."""
    ids = [c for c, _ in cited]
    leaks = [
        (c, v) for c, t in cited for v in [answer, *gold_titles] if contains_answer(v, t)
    ]  # fmt: skip
    return [
        GuardResult("cites_something", bool(cited), f"{len(cited)} chunks cited"),
        GuardResult(
            "disjoint_from_gold",
            not set(ids) & gold_chunk_ids,
            f"overlap {sorted(set(ids) & gold_chunk_ids)}",
        ),
        GuardResult("no_support_in_cited", not leaks, f"mentions {leaks}"),
    ]
