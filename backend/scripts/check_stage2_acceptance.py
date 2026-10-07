"""Stage 2 acceptance criteria (DECISIONS.md D-034), checked against stored rows.

Run in CI after the Ollama smoke test. Exits non-zero unless ALL hold:
  1. the Ollama path works end to end: >= 1 stored answer parsed and mapped to chunks;
  2. attribution holds a real digest: every Ollama row's model_version is
     "<tag>@<64-hex digest>" and every attribution column is set;
  3. the parse-failure path ran on real model output: >= 1 stored failure with its
     reason and non-empty raw output.
(4, a green CI run on the tip, is this script exiting 0 inside that run.)

No quality number is required or printed as one. A smoke-test model's parse-failure rate
describes its JSON compliance, not grounded generation.
"""

import re
import sys

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from judge_check.db import get_engine
from judge_check.models import GeneratedAnswer, LLMCallColumns, PromptTemplateRow

REAL_DIGEST = re.compile(r"^[\w.\-]+:[\w.\-]+@(sha256:)?[0-9a-f]{64}$")


def main() -> int:
    s = sessionmaker(bind=get_engine())()
    rows = s.scalars(select(GeneratedAnswer).where(GeneratedAnswer.provider == "ollama")).all()
    problems = []
    if not rows:
        problems.append("no Ollama generations stored")
    bad_versions = sorted({r.model_version for r in rows if not REAL_DIGEST.match(r.model_version)})
    if bad_versions:
        problems.append(f"model_version without a real digest: {bad_versions}")
    for r in rows:
        missing = [c for c in LLMCallColumns.REQUIRED if getattr(r, c) in (None, "")]
        if missing:
            problems.append(f"row {r.id} missing attribution {missing}")
        if s.get(PromptTemplateRow, r.prompt_sha256) is None:
            problems.append(f"row {r.id} references an unstored prompt {r.prompt_sha256[:12]}")
    ok = [r for r in rows if r.parse_error is None and r.cited_chunk_ids]
    failed = [r for r in rows if r.parse_error and r.output_text]
    if not ok:
        problems.append("no generation parsed and mapped to chunks")
    if not failed:
        problems.append("parse-failure path never exercised on real output")

    print(f"Ollama rows: {len(rows)}; versions: {sorted({r.model_version for r in rows})}")
    print(f"  parsed and cited: {len(ok)}   stored parse failures: {len(failed)}")
    for reason in sorted({r.parse_error for r in failed}):
        print(f"    failure seen: {reason}")
    print("  SMOKE TEST: these counts describe the smoke model's output formatting,")
    print("  not grounded-generation quality (D-034).")
    if problems:
        print("\nSTAGE 2 ACCEPTANCE: FAIL")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nSTAGE 2 ACCEPTANCE: PASS (criteria 1-3; criterion 4 is this CI run being green)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
