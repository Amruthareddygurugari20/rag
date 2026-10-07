"""Variant constructors, text level (D-035). Database persistence lives elsewhere.

Every constructor returns an Outcome: either a labelled variant with its guard report, or a
discard with the reason. Discards are data (reported per type and reason), never silent.
Containment checks go only through `guards` (D-033/D-035: this module may not use
contains_answer itself; the source scan enforces it).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from judge_check.variants.guards import (
    GuardFailure,
    GuardResult,
    chunks_mentioning_none,
    citation_guards,
    mutation_guards,
    presence_guard,
    require,
    sentences_not_stating,
    values_apart_from,
)
from judge_check.variants.material import Material, Sentence, base_text, cite, lead
from judge_check.variants.mutate import (
    find_spans,
    replacement_for,
    rng_for,
    strip_disambiguation,
    substitute,
)
from judge_check.variants.valuetypes import DATE_DMY_RE, DATE_MDY_RE, YEAR_RE, VType, vtype

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
    cited: tuple[int, ...] = ()  # chunk ids; the judge prompt renders them (addendum D)

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


# --- the other deterministic types, from a Material (D-035 table; addendum D) ----------------

VERBOSE_RATIO = 2.5  # D-035: verbose text is at least this many times the base length

# Bump a constructor's version whenever its output for the same input could change.
VERSIONS = {
    "correct": 1, "verbose_correct": 1, "terse_correct": 1,
    "right_topic_wrong_detail": 1, "partially_correct": 1, "right_answer_wrong_citation": 1,
}  # fmt: skip


def _no_gold(vt: str, label: bool, m: Material) -> Outcome | None:
    if not m.gold or not m.answer.strip():
        return Outcome(vt, label, None, None, "no_gold_evidence_or_answer")
    return None


def correct(m: Material) -> Outcome:
    """Type 1: the base text, citing the gold chunks."""
    vt = "correct"
    return _no_gold(vt, True, m) or Outcome(vt, True, base_text(m), cited=m.gold_chunk_ids)


def verbose_correct(m: Material) -> Outcome:
    """Type 2: the base text plus further true sentences from the gold paragraphs (not the
    gold sentences again), in document order, until it is VERBOSE_RATIO x the base length."""
    vt = "verbose_correct"
    if (bad := _no_gold(vt, True, m)) is not None:
        return bad
    base = base_text(m)
    extras = [
        s for p in m.gold_paragraphs for i, s in enumerate(p.sentences)
        if i not in p.gold and s.text.strip()
    ]  # fmt: skip
    text, added = base, []
    for s in extras:
        if len(text) >= VERBOSE_RATIO * len(base):
            break
        text, added = f"{text} {s.text}", [*added, s]
    ratio = len(text) / len(base)
    params = {"base_chars": len(base), "chars": len(text), "ratio": round(ratio, 3),
              "added_sentences": len(added)}  # fmt: skip
    if ratio < VERBOSE_RATIO:
        return Outcome(vt, True, None, None, "too_little_extra_context", params=params)
    return Outcome(vt, True, text, params=params, cited=cite([*m.gold, *added]))


def terse_correct(m: Material) -> Outcome:
    """Type 3: the reference answer alone. It cites every gold chunk, not just one holding the
    answer string: a multi-hop answer is supported by the chain, not by one sentence."""
    vt = "terse_correct"
    return _no_gold(vt, True, m) or Outcome(vt, True, lead(m.answer), cited=m.gold_chunk_ids)


def wrong_detail_from(m: Material, *, seed: int) -> Outcome:
    """Type 5 on the base text; cites the gold chunks, which contradict the swapped value."""
    out = wrong_detail(base_text(m), m.answer, m.gold_texts, list(m.entity_candidates),
                       seed=seed, question_key=m.key)  # fmt: skip
    return out if out.discarded else replace(out, cited=m.gold_chunk_ids)


def partially_missing(m: Material) -> Outcome:
    """Type 7a: only the gold sentences that do NOT state the answer. True but incomplete."""
    vt, sk = "partially_correct", "missing"
    if (bad := _no_gold(vt, False, m)) is not None:
        return replace(bad, sub_kind=sk)
    kept = sentences_not_stating(m.answer, list(m.gold))
    if not kept:
        return Outcome(vt, False, None, sk, "every_gold_sentence_states_answer")
    if len(kept) == len(m.gold):
        # Nothing was left out (e.g. yes/no answers): that's the full evidence, not a part.
        return Outcome(vt, False, None, sk, "no_gold_sentence_states_answer")
    text = " ".join(s.text for s in kept)
    results = [
        presence_guard("answer_absent", m.answer, text, present=False),
        GuardResult("dropped_some", len(kept) < len(m.gold), f"{len(kept)}/{len(m.gold)} kept"),
    ]
    try:
        report = require(results)
    except GuardFailure as exc:
        return Outcome(vt, False, None, sk, f"guard_failed:{exc.first_failed}",
                       [asdict(r) for r in results])  # fmt: skip
    return Outcome(vt, False, text, sk, None, report,
                   {"kept": len(kept), "gold": len(m.gold)}, cite(kept))  # fmt: skip


_BARE_NUMBER = re.compile(r"(?<![\w.,])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w]|[.,]\d)")


def detail_values(sentences: list[Sentence], titles: list[str], answer: str) -> list[tuple]:
    """Typed values in sentences that don't state the answer: (kind, value), in a stable
    order. Dates win over the years and numbers inside them; years over bare numbers."""
    found: dict[tuple, None] = {}
    for s in sentences:
        taken: list[tuple[int, int]] = []

        def free(a: int, b: int) -> bool:
            return all(b <= x or a >= y for x, y in taken)  # noqa: B023

        for kind, rx in ((VType.DATE, DATE_MDY_RE), (VType.DATE, DATE_DMY_RE),
                         (VType.YEAR, re.compile(rf"(?<!\w){YEAR_RE.pattern}(?!\w)")),
                         (VType.NUMBER, _BARE_NUMBER)):  # fmt: skip
            for x in rx.finditer(s.text):
                if free(x.start(), x.end()) and vtype(x.group()) is kind:
                    taken.append((x.start(), x.end()))
                    found[(kind, x.group())] = None
        for t in titles:
            name = strip_disambiguation(t)
            if vtype(name) is VType.ENTITY and find_spans(s.text, name):
                found[(VType.ENTITY, name)] = None
    return values_apart_from(answer, list(found))


def partially_wrong(m: Material, *, seed: int) -> Outcome:
    """Type 7b: the base text with the answer kept and one supporting value (in a gold
    sentence that doesn't state the answer) replaced everywhere. Same five guards as type 5,
    plus: the answer must still be stated."""
    vt, sk = "partially_correct", "wrong"
    if (bad := _no_gold(vt, False, m)) is not None:
        return replace(bad, sub_kind=sk)
    base = base_text(m)
    others = sentences_not_stating(m.answer, list(m.gold))
    titles = [p.title for p in m.gold_paragraphs]
    values = detail_values(others, titles, m.answer)
    if not values:
        return Outcome(vt, False, None, sk, "no_typed_detail_outside_answer")
    rng = rng_for(seed, m.key, vt, sk)
    order = [values.pop(int(rng.random() * len(values))) for _ in range(len(values))]
    attempts: list[dict] = []
    for kind, value in order:
        spans = find_spans(base, value)
        for _ in range(MAX_ATTEMPTS):
            try:
                rep = replacement_for(value, kind, rng, list(m.entity_candidates))
            except LookupError:
                break  # this value can't be replaced: try the next one
            sub = substitute(base, spans, rep)
            results = [
                *mutation_guards(sub, value, kind, m.gold_texts),
                presence_guard("answer_kept", m.answer, sub.result, present=True),
            ]
            try:
                report = require(results)
            except GuardFailure:
                attempts.append({"value": value, "replacement": rep,
                                 "guards": [asdict(r) for r in results]})  # fmt: skip
                continue
            return Outcome(vt, False, sub.result, sk, None, report,
                           {"original": value, "value_kind": str(kind), "replacement": rep,
                            "rejected_attempts": attempts},
                           m.gold_chunk_ids)  # fmt: skip
    first = next((g["name"] for g in attempts[-1]["guards"] if not g["passed"]), None) \
        if attempts else None  # fmt: skip
    reason = f"guard_failed:{first}" if first else "no_replacement"
    return Outcome(vt, False, None, sk, reason, attempts, {"values": [v for _, v in order]})


def wrong_citation(m: Material, *, seed: int) -> Outcome:
    """Type 9: the base text (right answer), citing as many distractor chunks as it should
    cite gold chunks. A chunk that mentions the answer or a gold entity is never chosen."""
    vt = "right_answer_wrong_citation"
    if (bad := _no_gold(vt, False, m)) is not None:
        return bad
    titles = [strip_disambiguation(p.title) for p in m.gold_paragraphs]
    gold_ids = set(m.gold_chunk_ids)
    eligible = [
        (c, t) for c, t in chunks_mentioning_none(list(m.distractor_chunks), [m.answer, *titles])
        if c not in gold_ids
    ]  # fmt: skip
    need = len(m.gold_chunk_ids)
    params = {"need": need, "eligible": len(eligible),
              "distractor_chunks": len(m.distractor_chunks)}  # fmt: skip
    if len(eligible) < need:
        return Outcome(vt, False, None, None, "too_few_unrelated_distractor_chunks",
                       params=params)  # fmt: skip
    rng = rng_for(seed, m.key, vt)
    pool = sorted(eligible)
    chosen = [pool.pop(int(rng.random() * len(pool))) for _ in range(need)]
    # Re-checked by the guards on the final choice, not trusted from the filter above.
    results = citation_guards(chosen, gold_ids, m.answer, titles)
    try:
        report = require(results)
    except GuardFailure as exc:
        return Outcome(vt, False, None, None, f"guard_failed:{exc.first_failed}",
                       [asdict(r) for r in results], params)  # fmt: skip
    return Outcome(vt, False, base_text(m), None, None, report, params,
                   tuple(c for c, _ in chosen))  # fmt: skip


def deterministic_variants(m: Material, *, seed: int) -> list[Outcome]:
    """Every deterministic type for one question, discards included (they're data)."""
    return [
        correct(m), verbose_correct(m), terse_correct(m),
        wrong_detail_from(m, seed=seed),
        partially_missing(m), partially_wrong(m, seed=seed),
        wrong_citation(m, seed=seed),
    ]  # fmt: skip
