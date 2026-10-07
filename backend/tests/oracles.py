"""Independent checkers for variant tests. Deliberately NOT the implementation's helpers
(contains_answer, find_spans, vtype): a bug shared by code and check can't be caught."""

import re

_PUNCT = re.compile(r"[^\w\s]")


def tokens(s: str) -> list[str]:
    """Lowercase, drop punctuation and articles, split on whitespace."""
    return [t for t in _PUNCT.sub("", s.lower()).split() if t not in {"a", "an", "the"}]


def states(text: str, value: str) -> bool:
    """Does `value` occur in `text` as a contiguous token sequence?"""
    h, n = tokens(text), tokens(value)
    return bool(n) and any(h[i : i + len(n)] == n for i in range(len(h) - len(n) + 1))
