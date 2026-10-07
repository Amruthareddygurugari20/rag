"""Conservative value typing for mutations (D-035).

A value gets a type only if a strict parser accepts all of it. Anything else is untyped,
and an untyped answer is never mutated: a type-5 variant for it is discarded with the
reason `answer_type_unparseable`. Guessing a type is how "2016 United States elections"
becomes a "number" and a mutation swaps the wrong kind of thing.
"""

from __future__ import annotations

import re
from enum import StrEnum

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]  # fmt: skip
_M = "|".join(MONTHS)


class VType(StrEnum):
    YEAR = "year"
    DATE = "date"
    NUMBER = "number"
    ENTITY = "entity"


YEAR_RE = re.compile(r"(?:1\d{3}|20\d{2})")
DATE_MDY_RE = re.compile(rf"(?P<month>{_M}) (?P<day>[1-9]|[12]\d|3[01]), (?P<year>\d{{4}})")
DATE_DMY_RE = re.compile(rf"(?P<day>[1-9]|[12]\d|3[01]) (?P<month>{_M}) (?P<year>\d{{4}})")
# A numeral (commas only in correct thousands positions, optional decimals) and at most one
# unit word. "729 at the 2010 census" and "575 acres (2.08 km²)" are deliberately rejected.
NUMBER_RE = re.compile(
    r"(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?: (?P<unit>[A-Za-z][A-Za-z%²³]*\d?))?"
)
_PARTICLES = {"of", "the", "de", "von", "van", "la", "le", "du", "del", "da", "di", "der"}
_CAP_TOKEN = re.compile(r"[A-Z][\w'’.&-]*|[IVXLC]+")


def vtype(value: str) -> VType | None:
    """The strict type of a whole value, or None (untyped)."""
    v = value.strip()
    if YEAR_RE.fullmatch(v):
        return VType.YEAR
    if DATE_MDY_RE.fullmatch(v) or DATE_DMY_RE.fullmatch(v):
        return VType.DATE
    if NUMBER_RE.fullmatch(v):
        return VType.NUMBER
    tokens = v.split()
    if (
        1 <= len(tokens) <= 6
        and _CAP_TOKEN.fullmatch(tokens[0])
        and all(_CAP_TOKEN.fullmatch(t) or t in _PARTICLES for t in tokens)
        and _CAP_TOKEN.fullmatch(tokens[-1])
    ):
        return VType.ENTITY
    return None
