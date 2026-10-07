"""Seeded, type-preserving replacement values, and span-exact substitution (D-035).

Randomness only ever comes from `random.Random(int_seed).random()`, whose sequence Python
guarantees across versions (the same reasoning as the dataset sampler, D-013), and the
integer seed is derived from (seed, question id, purpose) by SHA-256. So the same corpus
always gets the same variants.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass

from judge_check.variants.valuetypes import (
    DATE_DMY_RE,
    DATE_MDY_RE,
    MONTHS,
    NUMBER_RE,
    VType,
    vtype,
)

YEAR_OFFSETS = [d for d in range(-15, 16) if d != 0]
NUMBER_FACTORS = [0.5, 0.75, 1.25, 1.5, 2.0, 3.0]
MONTH_OFFSETS = list(range(1, 12))


def rng_for(seed: int, *parts: object) -> random.Random:
    digest = hashlib.sha256(":".join(map(str, (seed, *parts))).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def pick(rng: random.Random, options: list):
    return options[int(rng.random() * len(options))]


def _format_number(x: float, like: str) -> str:
    decimals = len(like.split(".")[1]) if "." in like else 0
    text = f"{x:,.{decimals}f}" if "," in like else f"{x:.{decimals}f}"
    return text


def replacement_for(value: str, kind: VType, rng: random.Random, candidates: list[str]) -> str:
    """A different value of the same type. `candidates` are used for entities only."""
    v = value.strip()
    if kind is VType.YEAR:
        year, off = int(v), pick(rng, YEAR_OFFSETS)
        # Reflect (never clamp) at the range edges: clamping 1000 - 5 gives back 1000.
        new = year + off if 1000 <= year + off <= 2099 else year - off
        return str(new)
    if kind is VType.NUMBER:
        m = NUMBER_RE.fullmatch(v)
        num = float(m["num"].replace(",", ""))
        first = pick(rng, NUMBER_FACTORS)
        # Rounding can undo a factor (1 x 0.75 -> "1"): try the others in a fixed order.
        for factor in [first] + [f for f in NUMBER_FACTORS if f != first]:
            new = _format_number(num * factor, m["num"])
            if new != m["num"]:
                return f"{new} {m['unit']}" if m["unit"] else new
        raise LookupError(f"no factor changes {m['num']!r} at its precision")
    if kind is VType.DATE:
        m = DATE_MDY_RE.fullmatch(v) or DATE_DMY_RE.fullmatch(v)
        month = MONTHS[(MONTHS.index(m["month"]) + pick(rng, MONTH_OFFSETS)) % 12]
        day = min(int(m["day"]), 28)  # valid in every month
        if DATE_MDY_RE.fullmatch(v):
            return f"{month} {day}, {m['year']}"
        return f"{day} {month} {m['year']}"
    if kind is VType.ENTITY:
        pool = sorted({c for c in candidates if vtype(c) is VType.ENTITY})
        if not pool:
            raise LookupError("no same-type entity candidates")
        return pick(rng, pool)
    raise ValueError(f"cannot mutate type {kind!r}")


def find_spans(text: str, value: str) -> list[tuple[int, int]]:
    """Every case-insensitive, whole-word occurrence of `value` in `text`."""
    pattern = re.compile(rf"(?<!\w){re.escape(value.strip())}(?!\w)", re.IGNORECASE)
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


@dataclass(frozen=True)
class Substitution:
    source: str
    result: str
    spans: list[tuple[int, int]]  # spans in `source` that were replaced
    replacement: str


def substitute(text: str, spans: list[tuple[int, int]], replacement: str) -> Substitution:
    out, last = [], 0
    for s, e in sorted(spans):
        out += [text[last:s], replacement]
        last = e
    out.append(text[last:])
    return Substitution(text, "".join(out), sorted(spans), replacement)


def strip_disambiguation(title: str) -> str:
    """Wikipedia titles: "Derek Dooley (American football)" -> "Derek Dooley"."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
