"""Types 1, 2, 3, 7 and 9 (D-035 table; addendum D), on planted material and on the whole
demo set. Oracles come from tests/oracles.py, not from the implementation."""

from collections import Counter

import pytest

from judge_check.datasets.demo import chunk_in_memory, load_demo
from judge_check.variants.build import DocView, discard_report, material_for
from judge_check.variants.constructors import (
    VERBOSE_RATIO,
    correct,
    deterministic_variants,
    partially_missing,
    partially_wrong,
    terse_correct,
    verbose_correct,
    wrong_citation,
)
from judge_check.variants.material import Material, Paragraph, Sentence
from tests.oracles import states

S1 = Sentence("The Hawks were coached by Ann Lee in 1998.", (1,))
S2 = Sentence("Ann Lee was born in Bergen.", (2,))
X1 = Sentence("Ann Lee later moved to Oslo with her family and two dogs.", (2,))
X2 = Sentence("She retired from coaching after a long and successful career.", (3,))
X3 = Sentence("Bergen is the second largest city in Norway by population.", (4,))


def mat(answer="Bergen", gold=(S1, S2), distractors=None, cands=("Trondheim", "Stavanger")):
    paras = (
        Paragraph("Hawks (team)", (S1, X1, X2), frozenset({0})),
        Paragraph("Ann Lee", (S2, X3), frozenset({0})),
    )
    if distractors is None:
        distractors = (
            (10, "Trondheim is a city in central Norway."),
            (11, "Stavanger hosts an oil museum."),
            (12, "Oslo has a fjord."),
        )
    return Material("q1", answer, tuple(gold), paras, distractors, cands)  # fmt: skip


# --- accept types ------------------------------------------------------------------------------


def test_correct_is_answer_then_gold_sentences() -> None:
    out = correct(mat())
    assert out.text == "Bergen. " + S1.text + " " + S2.text  # rebuilt by hand
    assert out.is_correct and out.cited == (1, 2)


def test_terse_is_the_answer_citing_the_whole_chain() -> None:
    out = terse_correct(mat())
    assert out.text == "Bergen." and out.cited == (1, 2)


def test_verbose_adds_only_non_gold_paragraph_sentences_and_stops_at_the_ratio() -> None:
    m = mat()
    out = verbose_correct(m)
    base = "Bergen. " + S1.text + " " + S2.text
    if out.discarded:  # oracle: then even every extra sentence can't reach the ratio
        total = len(base) + sum(len(s.text) + 1 for s in (X1, X2, X3))
        assert total < VERBOSE_RATIO * len(base)
        return
    assert out.text.startswith(base + " ")
    added = out.text[len(base) + 1 :]
    extras = [X1.text, X2.text, X3.text]
    n = out.params["added_sentences"]
    assert added == " ".join(extras[:n])  # in document order, no gold sentence repeated
    assert len(out.text) >= VERBOSE_RATIO * len(base)
    assert len(base) + sum(len(e) + 1 for e in extras[: n - 1]) < VERBOSE_RATIO * len(base)


def test_verbose_reaches_ratio_when_paragraphs_allow() -> None:
    long = Sentence("Filler about the team's history and stadium. " * 6, (9,))
    m = mat()
    m = Material(m.key, m.answer, m.gold,
                 (Paragraph("Hawks (team)", (S1, long, long), frozenset({0})),),
                 m.distractor_chunks, m.entity_candidates)  # fmt: skip
    out = verbose_correct(m)
    assert not out.discarded and out.params["ratio"] >= VERBOSE_RATIO and 9 in out.cited


def test_verbose_discards_rather_than_padding() -> None:
    m = mat()
    m = Material(m.key, m.answer, m.gold, (Paragraph("p", (S1, S2), frozenset({0, 1})),),
                 m.distractor_chunks, m.entity_candidates)  # fmt: skip
    out = verbose_correct(m)
    assert out.discarded and out.discard_reason == "too_little_extra_context"


# --- type 7 ------------------------------------------------------------------------------------


def test_partial_missing_drops_exactly_the_answer_sentences() -> None:
    out = partially_missing(mat())
    assert out.text == S1.text and out.cited == (1,) and not out.is_correct
    assert not states(out.text, "Bergen")


def test_partial_missing_needs_something_dropped_and_something_kept() -> None:
    yes_no = partially_missing(mat(answer="yes"))
    assert yes_no.discard_reason == "no_gold_sentence_states_answer"  # full evidence, no part
    only = partially_missing(mat(gold=(S2,)))
    assert only.discard_reason == "every_gold_sentence_states_answer"


@pytest.mark.parametrize("seed", range(25))
def test_partial_wrong_keeps_answer_and_changes_one_supporting_value(seed: int) -> None:
    m = mat()
    out = partially_wrong(m, seed=seed)
    assert not out.discarded, out.discard_reason
    base = "Bergen. " + S1.text + " " + S2.text
    orig, rep = out.params["original"], out.params["replacement"]
    # The typed values in S1 (the sentence not stating the answer): a year, and the two gold
    # paragraph titles it mentions ("Hawks (team)" -> "Hawks", "Ann Lee").
    assert orig in {"1998", "Ann Lee", "Hawks"}
    assert states(out.text, "Bergen")  # answer still stated
    assert not states(out.text, orig) and rep not in " ".join(m.gold_texts)
    assert out.text == base.replace(orig, rep)  # oracle: plain replace-all
    assert not out.is_correct and out.sub_kind == "wrong"


def test_partial_wrong_never_touches_a_value_inside_the_answer() -> None:
    s = Sentence("The 1998 Hawks won in 1998.", (1,))
    out = partially_wrong(mat(answer="1998 Hawks", gold=(s, S2)), seed=0)
    # "1998" and "Hawks" overlap the answer; "Ann Lee" is the only other candidate.
    assert out.discarded or out.params["original"] == "Ann Lee"
    assert out.discarded or states(out.text, "1998 Hawks")


def test_partial_wrong_without_typed_detail_is_discarded() -> None:
    s = Sentence("They played with great spirit.", (1,))
    out = partially_wrong(mat(gold=(s, S2)), seed=0)
    assert out.discard_reason == "no_typed_detail_outside_answer"


# --- type 9 ------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_wrong_citation_cites_unrelated_distractor_chunks(seed: int) -> None:
    out = wrong_citation(mat(), seed=seed)
    assert out.text == correct(mat()).text  # same right answer
    assert len(out.cited) == 2 and not set(out.cited) & {1, 2, 3, 4}
    assert set(out.cited) <= {10, 11, 12}


def test_wrong_citation_skips_chunks_mentioning_answer_or_gold_entity() -> None:
    d = ((10, "Bergen has rain."), (11, "Ann Lee met the mayor."), (12, "Oslo has a fjord."),
         (13, "Hawks (team) moved stadium."), (14, "Stavanger hosts a museum."))  # fmt: skip
    for seed in range(20):
        out = wrong_citation(mat(distractors=d), seed=seed)
        assert set(out.cited) == {12, 14}  # oracle: by inspection of the five texts
    out = wrong_citation(mat(distractors=d[:3]), seed=0)
    assert out.discard_reason == "too_few_unrelated_distractor_chunks"


# --- whole demo set: label invariants, checked by the independent oracle ------------------------

STATES_ANSWER = {
    "correct": True, "verbose_correct": True, "terse_correct": True,
    "right_topic_wrong_detail": False, "right_answer_wrong_citation": True,
    ("partially_correct", "missing"): False, ("partially_correct", "wrong"): True,
}  # fmt: skip


def demo_in_memory() -> list:
    """[(Material, Outcome)] for every demo question and deterministic type; chunk ids are
    positions in chunk_in_memory's order. Shared with the database test."""
    docs, qs = load_demo()
    by_doc: dict = {}
    for i, (d, s, e, t) in enumerate(chunk_in_memory(docs)):
        by_doc.setdefault(d, []).append((i, s, e, t))
    views = {d.id: DocView(d.title, d.text, d.sentence_spans or []) for d in docs}
    out = []
    for q in qs:
        ev = [(e.document_id, e.start, e.end) for e in q.gold_evidence]
        m = material_for(q.id, q.reference_answer, ev, q.metadata["distractor_titles"],
                         views, by_doc)  # fmt: skip
        out += [(m, o) for o in deterministic_variants(m, seed=20261007)]
    return out


@pytest.fixture(scope="module")
def demo_outcomes():
    return demo_in_memory()


def test_every_kept_demo_variant_obeys_its_label_invariant(demo_outcomes) -> None:
    checked = 0
    for m, o in demo_outcomes:
        if o.discarded:
            continue
        checked += 1
        key = (o.variant_type, o.sub_kind) if o.variant_type == "partially_correct" else (
            o.variant_type)  # fmt: skip
        assert states(o.text, m.answer) is STATES_ANSWER[key], (m.key, o.variant_type, o.text)
        assert o.cited, (m.key, o.variant_type)
        gold = set(m.gold_chunk_ids)
        if o.variant_type == "right_answer_wrong_citation":
            assert not set(o.cited) & gold
        elif o.variant_type != "partially_correct":
            assert gold <= set(o.cited)
    assert checked > 1000  # not vacuous: most of the 7 x 200 outcomes are kept variants


def test_demo_discard_report_is_recorded(demo_outcomes) -> None:
    report = discard_report(o for _, o in demo_outcomes)
    print("\n" + "\n".join(f"{k}: {dict(v)}" for k, v in report.items()))
    assert sum(sum(c.values()) for c in report.values()) == 7 * 200
    assert report["correct"] == Counter(kept=200)  # every question has gold and an answer


def test_demo_build_is_deterministic(demo_outcomes) -> None:
    m = demo_outcomes[0][0]
    assert deterministic_variants(m, seed=20261007) == deterministic_variants(m, seed=20261007)
