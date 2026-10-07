"""Type balancing for evaluation-set assembly (D-035, addendum B).

Within each group (accept: types 1-4; reject: types 5-9), every type is down-sampled, with a
seed, to the group's smallest type count, so headline FAR/FRR can't be dominated by the type
easiest to generate in bulk.

The stratum is the variant TYPE and nothing finer. Sub-kinds (type 5's entity/year/number/
date, type 7's missing/wrong) are deliberately not balanced: doing that would cut type 5 to
the size of its single date answer. `balance_by_type` therefore takes only a type key, and a
test plants exactly that case.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from judge_check.datasets.hotpotqa import seeded_permutation

T = TypeVar("T")

ACCEPT_TYPES = ("correct", "verbose_correct", "terse_correct", "paraphrased_correct")
REJECT_TYPES = (
    "right_topic_wrong_detail", "hedged_nonanswer", "partially_correct",
    "unsupported_but_plausible", "right_answer_wrong_citation",
)  # fmt: skip


@dataclass(frozen=True)
class Balanced(Generic[T]):
    kept: list[T]
    before: dict[str, int]  # per type, before balancing
    after: dict[str, int]
    missing_types: dict[str, list[str]]  # per group: types with zero variants (reported)


def balance_by_type(
    variants: Sequence[T], type_of: Callable[[T], str], *, seed: int
) -> Balanced[T]:
    by_type: dict[str, list[T]] = defaultdict(list)
    for v in variants:
        by_type[type_of(v)].append(v)
    unknown = set(by_type) - set(ACCEPT_TYPES) - set(REJECT_TYPES)
    if unknown:
        raise ValueError(f"unknown variant types {sorted(unknown)}")
    kept: list[T] = []
    missing: dict[str, list[str]] = {}
    for group, types in (("accept", ACCEPT_TYPES), ("reject", REJECT_TYPES)):
        present = [t for t in types if by_type.get(t)]
        missing[group] = [t for t in types if not by_type.get(t)]
        if not present:
            continue
        floor = min(len(by_type[t]) for t in present)
        for t in present:
            # Version-stable seeded shuffle (D-013), then the first `floor`.
            kept += seeded_permutation(by_type[t], hash_key(seed, t))[:floor]
    count = lambda xs: {t: sum(type_of(v) == t for v in xs) for t in by_type}  # noqa: E731
    return Balanced(kept, count(variants), count(kept), missing)


def hash_key(seed: int, label: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:8], "big")
