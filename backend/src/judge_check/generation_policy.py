"""Which generated answers may enter the evaluation set (D-032 addendum A, revised).

Generated answers exist in the design to bring *realistic, fluent* RAG output into the
evaluation set. A generator whose output isn't plausible inverts the measurement: its
answers are trivially rejectable, the human-labelled arm looks *easier* than the
constructed arm, and the realism gap comes out backwards.

The gate is therefore **measured plausibility**, not a list of model names. A list can only
block the models someone thought of; qwen2.5:1.5b or gemma3:270m would pass a deny-list
naming qwen2.5:0.5b. Reviewers in stage 5 mark each sampled generation "plausible or not"
separately from "correct or not", and an exact `model_version` becomes eligible only when

    Wilson 95% lower bound of (plausible / reviewed)  >=  PLAUSIBLE_LB_THRESHOLD   (0.80)
    and reviewed >= MIN_REVIEWED                                                     (40)

Pre-registered (committed before any generator had been reviewed). At n = 40 this needs
37 or more plausible (92%). Operating characteristic, P(pass) at n = 40 / 100: true rate
0.95 -> 0.86 / 1.00; 0.90 -> 0.42 / 0.80; 0.85 -> 0.13 / 0.25. It deliberately prefers
excluding a good generator to admitting a bad one.

Eligibility is per exact model_version (tag@digest): re-pulled weights are a new
generator with no reviews. Being reviewed is not being eligible. The review queue samples
any generator; only eligible ones' human-labelled answers count in the evaluation set.
"""

from __future__ import annotations

import math

PLAUSIBLE_LB_THRESHOLD = 0.80
MIN_REVIEWED = 40
Z_95 = 1.959963984540054


def wilson_lower_bound(successes: int, n: int, z: float = Z_95) -> float:
    """Lower end of the Wilson score interval for a binomial proportion. Unlike the
    normal approximation it stays inside [0, 1] and behaves at small n and near p = 1,
    exactly where a "is this generator ~always plausible?" question lives."""
    if n <= 0:
        return 0.0
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def is_eval_eligible(plausible: int, reviewed: int) -> bool:
    """True when a generator's measured plausibility clears the pre-registered gate."""
    return reviewed >= MIN_REVIEWED and (
        wilson_lower_bound(plausible, reviewed) >= PLAUSIBLE_LB_THRESHOLD
    )


def min_plausible_to_pass(reviewed: int) -> int | None:
    """Smallest plausible count that passes at this many reviews (None if none can)."""
    if reviewed < MIN_REVIEWED:
        return None
    return next((k for k in range(reviewed + 1) if is_eval_eligible(k, reviewed)), None)
