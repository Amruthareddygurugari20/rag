"""Build the HotpotQA demo subset from the dev (distractor) set, in its original JSON layout.

This module uses only the standard library so it can run as a plain script on the raw,
untrusted download (``python -I hotpotqa.py ...``) without installing anything.

It is fully deterministic: the same source file and seed always yield byte-identical output
(the sample is seeded, see `seeded_permutation`). No
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
import random
import re
import string
import sys
from collections import Counter
from pathlib import Path

# Original distribution. Unreachable when the subset was built (2026-10-07), so the build uses
# the Hugging Face copy, converted back to this layout (D-013). Provenance goes into MANIFEST.
ORIGINAL_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
# Seed for the random sample. Recorded in MANIFEST.json. Changing it draws a different sample.
SEED = 20261007
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


def seeded_permutation(items: list, seed: int) -> list:
    """A uniformly random permutation, reproducible across Python versions.

    Fisher-Yates driven only by ``random.Random(seed).random()``. Python guarantees that
    sequence for a given seed across versions; it does *not* guarantee ``shuffle`` or
    ``sample``, whose internals have changed before. The first n items of a uniformly random
    permutation are a simple random sample of size n, without replacement.
    """
    rng = random.Random(seed)
    out = list(items)
    for i in range(len(out) - 1, 0, -1):
        j = int(rng.random() * (i + 1))
        out[i], out[j] = out[j], out[i]
    return out


def build_subset(
    examples: list[dict], n: int = N_QUESTIONS, seed: int = SEED
) -> tuple[list, list, dict]:
    rejected: Counter[str] = Counter()
    eligible = []
    for ex in examples:
        reason, paragraphs = check_example(ex)
        if reason:
            rejected[reason] += 1
        else:
            eligible.append((ex, paragraphs))

    # Simple random sample over every question that passed the filters: fix a canonical
    # order (by id, so the source file's order can't matter), permute it with the seed, and
    # take questions in that order. The one departure from a plain random sample is that a
    # question whose paragraphs contradict an already-selected one is skipped (counted).
    eligible.sort(key=lambda item: item[0]["_id"])
    population_by_type = Counter(ex["type"] for ex, _ in eligible)
    eligible = seeded_permutation(eligible, seed)

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
        "sampling": {
            "method": "simple random sample without replacement over all questions passing "
            "the filters (seeded Fisher-Yates permutation of id-sorted eligible questions, "
            "first n taken; title conflicts skipped)",
            "seed": seed,
        },
        "eligible_population": len(eligible),
        "eligible_by_type": dict(sorted(population_by_type.items())),
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
    p.add_argument("--seed", type=int, default=SEED, help="random-sample seed")
    p.add_argument("--source-url", required=True, help="where the source file came from")
    p.add_argument("--source-revision", default="", help="e.g. a Hugging Face commit sha")
    p.add_argument("--source-sha256", required=True, help="checksum of the downloaded file")
    args = p.parse_args(argv)

    raw = args.input.read_bytes()
    documents, questions, stats = build_subset(json.loads(raw), args.n, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "corpus.jsonl", documents)
    write_jsonl(args.out / "questions.jsonl", questions)
    manifest = {
        "source_url": args.source_url,
        "source_revision": args.source_revision,
        "source_sha256": args.source_sha256,
        "original_url": ORIGINAL_URL,
        "builder_input_sha256": hashlib.sha256(raw).hexdigest(),
        "sample_seed": args.seed,
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
