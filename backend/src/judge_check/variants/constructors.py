"""Variant constructors, text level (D-035). Database persistence lives elsewhere.

Every constructor returns an Outcome: either a labelled variant with its guard report, or a
discard with the reason. Discards are data (reported per type and reason), never silent.
Containment checks go only through `guards` (D-033/D-035: this module may not use
contains_answer itself; the source scan enforces it).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from judge_check.variants.guards import GuardFailure, mutation_guards, require
from judge_check.variants.mutate import find_spans, replacement_for, rng_for, substitute
from judge_check.variants.valuetypes import vtype

MAX_ATTEMPTS = 5  # different replacement draws before a mutation is given up


@dataclass(frozen=True)
class Outcome:
    variant_type: str
    is_correct: bool
    text: str | None  # None for a discard
    sub_kind: str | None = None  # type 5: entity/year/number/date; type 7: missing/wrong
    discard_reason: str | None = None
    guard_report: list[dict] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def discarded(self) -> bool:
        return self.text is None


def wrong_detail(
    base_text: str,
    answer: str,
    gold_texts: list[str],
    entity_candidates: list[str],
    *,
    seed: int,
    question_key: str,
) -> Outcome:
    """Type 5: the base text with the answer value replaced everywhere, guarded."""
    vt, label = "right_topic_wrong_detail", False
    kind = vtype(answer)
    if kind is None:
        # Untyped answers are never mutated, not even a value inside them (D-035 C).
        return Outcome(vt, label, None, None, "answer_type_unparseable",
                       params={"answer": answer})  # fmt: skip
    spans = find_spans(base_text, answer)
    if not spans:
        return Outcome(vt, label, None, kind, "answer_not_in_base_text", params={"answer": answer})
    rng = rng_for(seed, question_key, vt)
    attempts: list[dict] = []
    for _ in range(MAX_ATTEMPTS):
        try:
            rep = replacement_for(answer, kind, rng, entity_candidates)
        except LookupError as exc:
            return Outcome(vt, label, None, kind, f"no_replacement: {exc}",
                           guard_report=attempts, params={"answer": answer})  # fmt: skip
        sub = substitute(base_text, spans, rep)
        results = mutation_guards(sub, answer, kind, gold_texts)
        try:
            report = require(results)
        except GuardFailure:
            attempts.append({"replacement": rep, "guards": [asdict(r) for r in results]})
            continue
        return Outcome(
            vt, label, sub.result, kind, None, report,
            {"original": answer, "replacement": rep, "rejected_attempts": attempts},
        )  # fmt: skip
    failed = attempts[-1]["guards"]
    first = next(g["name"] for g in failed if not g["passed"])
    return Outcome(vt, label, None, kind, f"guard_failed:{first}", guard_report=attempts,
                   params={"answer": answer})  # fmt: skip
