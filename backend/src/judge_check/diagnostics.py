"""HEURISTICS. NOT CORRECTNESS LABELS (D-033).

"Does the reference string appear in the answer?" and "does the answer cite a gold chunk?"
are free, tempting, and almost-correctness. They are wrong often enough to be dangerous: a
correct paraphrase fails the first; an answer that cites the gold chunk and then states
something false passes the second. Ground truth comes only from construction or a human
(D-000, D-032).

Enforced, not just documented:
- These values are computed on the fly for human-readable summaries and never stored.
  No table has a column for them, so scoring code can't read them from the database.
- tests/test_heuristic_boundary.py fails CI if any module other than the CLI's display
  code imports this module, or if `contains_answer` is used anywhere except the dataset
  builder (where it's a data filter, D-013) and here.
"""

from __future__ import annotations

from judge_check.datasets.hotpotqa import contains_answer


def reference_in_answer(reference: str, answer: str) -> bool:
    """Heuristic: normalised reference answer appears as whole words in the answer."""
    return contains_answer(reference, answer)


def cites_gold(cited_chunk_ids: list[int], gold_chunk_ids: set[int]) -> bool:
    """Heuristic: at least one cited chunk overlaps the gold evidence."""
    return bool(set(cited_chunk_ids) & gold_chunk_ids)
