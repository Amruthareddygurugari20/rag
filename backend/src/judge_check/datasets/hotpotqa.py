"""Build the HotpotQA demo subset from the official dev (distractor) file.

This module uses only the standard library so it can run as a plain script on the raw,
untrusted download (``python -I hotpotqa.py ...``) without installing anything.

It is fully deterministic: the same source file always yields byte-identical output. No
model is involved anywhere, because the subset is ground truth (DECISIONS.md D-000, D-013).

Output (the generic judge-check JSONL format, see judge_check.ingest.formats):
  corpus.jsonl     one Wikipedia intro paragraph per line, with sentence character spans
  questions.jsonl  question, reference answer, gold evidence as character spans
  MANIFEST.json    source checksum, seed, and how many source questions each filter rejected
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

SOURCE_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
# Pinned from the first download (see .github/workflows/build-demo-data.yml). While it is
# None the build still runs, records the digest in MANIFEST.json, and warns.
SOURCE_SHA256: str | None = None
SEED = "judge-check-hotpotqa-v1"
N_QUESTIONS = 200

# Rejection reasons, in the order they are checked. Each source question is counted under
# the first reason that applies.
YES_NO = "yes_no_answer"
BAD_SUPPORTING_FACT = "supporting_fact_unresolvable"
NOT_TWO_GOLD = "gold_not_exactly_two_paragraphs"
ANSWER_NOT_IN_EVIDENCE = "answer_not_in_gold_sentences"
TITLE_CONFLICT = "title_text_conflict"


def normalize_answer(s: str) -> str:
    """Normalisation from the official HotpotQA/SQuAD evaluation script."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains_answer(answer: str, evidence: str) -> bool:
    """Whole-word containment after normalisation ("art" does not match "party")."""
    a = normalize_answer(answer)
    return bool(a) and f" {a} " in f" {normalize_answer(evidence)} "


def build_paragraph(sentences: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join HotpotQA sentences into one text and return each sentence's character span.

    Spans stay index-aligned with HotpotQA's sentence ids (supporting facts refer to them).
    Spans exclude surrounding whitespace, and a whitespace-only sentence gets an empty span.
    """
    text = ""
    spans: list[tuple[int, int]] = []
    for sent in sentences:
        if text and sent and not sent[0].isspace() and not text[-1].isspace():
            text += " "
        start = len(text)
        text += sent
        stripped = sent.strip()
        if stripped:
            lead = len(sent) - len(sent.lstrip())
            spans.append((start + lead, start + lead + len(stripped)))
        else:
            spans.append((start, start))
    return text, spans


def check_example(ex: dict) -> tuple[str | None, dict]:
    """Return (rejection_reason or None, parsed paragraphs {title: (text, spans)})."""
    paragraphs = {title: build_paragraph(sents) for title, sents in ex["context"]}
    if normalize_answer(ex["answer"]) in {"yes", "no"}:
        return YES_NO, paragraphs
    if len(paragraphs) != len(ex["context"]):  # duplicate title inside one example
        return TITLE_CONFLICT, paragraphs
    for title, sent_id in ex["supporting_facts"]:
        if title not in paragraphs or not 0 <= sent_id < len(paragraphs[title][1]):
            return BAD_SUPPORTING_FACT, paragraphs
        start, end = paragraphs[title][1][sent_id]
        if start == end:
            return BAD_SUPPORTING_FACT, paragraphs
    if len({title for title, _ in ex["supporting_facts"]}) != 2:
        return NOT_TWO_GOLD, paragraphs
    evidence = " ".join(
        paragraphs[t][0][slice(*paragraphs[t][1][i])] for t, i in ex["supporting_facts"]
    )
    if not contains_answer(ex["answer"], evidence):
        return ANSWER_NOT_IN_EVIDENCE, paragraphs
    return None, paragraphs


def sample_key(qid: str) -> str:
    return hashlib.sha256(f"{SEED}:{qid}".encode()).hexdigest()


def build_subset(examples: list[dict], n: int = N_QUESTIONS) -> tuple[list, list, dict]:
    rejected: Counter[str] = Counter()
    eligible = []
    for ex in examples:
        reason, paragraphs = check_example(ex)
        if reason:
            rejected[reason] += 1
        else:
            eligible.append((ex, paragraphs))

    # Deterministic pseudo-random order: hash of a fixed seed and the question id.
    eligible.sort(key=lambda item: sample_key(item[0]["_id"]))

    corpus: dict[str, tuple[str, list[tuple[int, int]]]] = {}
    questions = []
    for ex, paragraphs in eligible:
        if len(questions) == n:
            break
        # The pooled corpus is keyed by title, so the same title must mean the same text.
        if any(t in corpus and corpus[t][0] != p[0] for t, p in paragraphs.items()):
            rejected[TITLE_CONFLICT] += 1
            continue
        corpus.update(paragraphs)
        questions.append(
            {
                "id": ex["_id"],
                "question": ex["question"],
                "reference_answer": ex["answer"],
                "gold_evidence": [
                    {
                        "document_id": t,
                        "start": paragraphs[t][1][i][0],
                        "end": paragraphs[t][1][i][1],
                    }
                    for t, i in ex["supporting_facts"]
                ],
                "metadata": {
                    "source": "hotpotqa_dev_distractor_v1",
                    "type": ex["type"],
                    "level": ex["level"],
                    "supporting_facts": ex["supporting_facts"],
                    "distractor_titles": [
                        t for t in paragraphs if t not in {sf[0] for sf in ex["supporting_facts"]}
                    ],
                },
            }
        )
    if len(questions) < n:
        raise ValueError(f"only {len(questions)} eligible questions, wanted {n}")

    documents = [
        {
            "id": title,
            "title": title,
            "text": text,
            # Only non-empty spans: these are what sentence-based chunkers split on.
            "sentence_spans": [list(s) for s in spans if s[0] != s[1]],
        }
        for title, (text, spans) in sorted(corpus.items())
    ]
    stats = {
        "source_questions": len(examples),
        "rejected": dict(sorted(rejected.items())),
        "eligible_before_conflicts": len(eligible),
        "selected_questions": len(questions),
        "selected_by_type": dict(sorted(Counter(q["metadata"]["type"] for q in questions).items())),
        "documents": len(documents),
    }
    return documents, questions, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, required=True, help="hotpot_dev_distractor_v1.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("-n", type=int, default=N_QUESTIONS)
    args = p.parse_args(argv)

    raw = args.input.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if SOURCE_SHA256 is None:
        print(f"WARNING: source checksum not pinned yet; got {digest}", file=sys.stderr)
    elif digest != SOURCE_SHA256:
        print(f"sha256 mismatch: got {digest}, expected {SOURCE_SHA256}", file=sys.stderr)
        return 1

    documents, questions, stats = build_subset(json.loads(raw), args.n)
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "corpus.jsonl", documents)
    write_jsonl(args.out / "questions.jsonl", questions)
    manifest = {
        "source_url": SOURCE_URL,
        "source_sha256": digest,
        "seed": SEED,
        "builder": "backend/src/judge_check/datasets/hotpotqa.py",
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "license": "CC BY-SA 4.0",
        "stats": stats,
    }
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
