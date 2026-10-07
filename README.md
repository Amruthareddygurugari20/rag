# judge-check

**How far can you trust the LLM judge in your RAG evaluation?**

Most RAG and agent evaluations score answers with an LLM judge, and few report how reliable
that judge is. One audit of a standard benchmark found its judge accepted **62.81%** of
deliberately wrong answers that were merely on the right topic. With a judge that loose, a
2-point gap between two systems can be pure noise.

judge-check runs your judge against answers whose correctness is known *by construction*:
correct answers, plus wrong ones made by deterministically mutating the correct ones. It
then measures the judge's false-accept and false-reject rates and turns them into a single
sentence:

> *Differences smaller than X points are not interpretable with this judge at N = Y.*

> **Status: stage 1 of 7.** Ingestion and retrieval work: configurable chunking, local
> embeddings, pgvector dense search, BM25 in SQL, and hybrid fusion. Generation, variants,
> judging, adjudication, metrics and the UI arrive stage by stage (see [Roadmap](#roadmap)).
> Every design choice is explained in [DECISIONS.md](DECISIONS.md). Concepts are explained
> in [docs/learn/](docs/learn/).

## Quickstart (about 10 minutes, mostly downloads)

Requirements: Docker, [uv](https://docs.astral.sh/uv/), Python ≥ 3.11. No GPU, no API keys.
[Ollama](https://ollama.com) will be needed from stage 2.

```bash
cp .env.example .env
make up                                   # Postgres 17 + pgvector on localhost:5433
make migrate                              # create tables
cd backend
uv run judge-check ingest ../data/hotpotqa_demo --name hotpotqa_demo
#   loads 1,986 paragraphs + 200 questions, chunks them, embeds 4,672 chunks on CPU
#   (first run downloads BAAI/bge-small-en-v1.5, ~130 MB)

uv run judge-check eval-retrieval --corpus hotpotqa_demo -k 5     # dense vs BM25 vs hybrid
uv run judge-check search "What instrument of war was only used by the President of the United States who was born in Lamar, Missouri?" --corpus hotpotqa_demo --compare
uv run python scripts/show_similarities.py                       # raw embedding numbers
```

`make test` runs the test suite. Tests that need Postgres or the model skip if those are
unavailable, unless `JUDGE_CHECK_REQUIRE_DB=1` / `JUDGE_CHECK_REQUIRE_MODEL=1` are set, as
they are in CI.

## Bring your own eval set

Two JSONL files in one directory:

```jsonc
// corpus.jsonl: one document per line
{"id": "doc-1", "title": "Ann Arbor", "text": "...", "sentence_spans": [[0, 57], [58, 120]]}
// questions.jsonl: one question per line
{"id": "q1", "question": "...", "reference_answer": "...",
 "gold_evidence": [{"document_id": "doc-1", "start": 0, "end": 57}], "metadata": {}}
```

- `sentence_spans` is optional. Without it, a rule-based sentence splitter is used.
- Gold evidence is given as character spans, not chunk ids, so it stays valid under any
  chunking (see [D-014](DECISIONS.md)).

Then: `uv run judge-check ingest path/to/dir --name my_eval`.

Chunking is configurable with JSON:

```bash
--chunking '{"strategy": "sentence_window", "sentences_per_chunk": 3, "overlap": 1}'
--chunking '{"strategy": "fixed_words", "words_per_chunk": 128, "overlap": 32}'
--chunking '{"strategy": "document"}'
```

## Retrieval in one paragraph

Chunks are embedded with `BAAI/bge-small-en-v1.5`. Queries get BGE's query instruction;
passages don't. Vectors are normalised to unit length, so pgvector's inner product equals
cosine similarity. BM25 runs in SQL over term-statistics tables and is tested to match
`rank_bm25` on every score of the demo corpus. Hybrid search fuses the two ranked lists
with Reciprocal Rank Fusion. Details: [docs/learn/01-retrieval.md](docs/learn/01-retrieval.md).

## Demo data

`data/hotpotqa_demo/` holds 200 questions from the HotpotQA dev set (distractor setting),
**CC BY-SA 4.0**, © the HotpotQA authors (Yang et al., EMNLP 2018). It is built
deterministically by `backend/src/judge_check/datasets/hotpotqa.py`. The data card covers
attribution, the changes made, the selection filters and known label noise:
[data/hotpotqa_demo/README.md](data/hotpotqa_demo/README.md).

## Licence and citation

Two licences, with a hard boundary:

| What | Licence |
|---|---|
| judge-check code and docs: everything **except** `data/hotpotqa_demo/` | **MIT**, see [LICENSE](LICENSE) |
| `data/hotpotqa_demo/`, adapted from HotpotQA | **CC BY-SA 4.0**, see its [data card](data/hotpotqa_demo/README.md) |

The MIT licence does not apply to the demo data. If you redistribute or modify the data,
ShareAlike applies to it, and HotpotQA must be attributed. Using judge-check on your own
data puts no licence obligations on that data.

To cite judge-check, use [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this
repository" button). If you report numbers on the demo corpus, also cite HotpotQA (Yang et
al., EMNLP 2018), which is listed in the file's `references`.

## Repository layout

```
backend/
  src/judge_check/
    ingest/          JSONL format, chunking, ingestion pipeline
    retrieval/       dense, BM25 (SQL), RRF fusion, gold mapping, metrics
    datasets/        HotpotQA subset builder, demo loader
    embeddings.py    BGE embedder (query instruction, normalisation)
    models.py        ORM models
    cli.py           judge-check command
  migrations/        Alembic, one migration per schema change
  scripts/           show_similarities.py, HF→JSON converter
  tests/
data/hotpotqa_demo/  built-in demo corpus (CC BY-SA 4.0)
docs/learn/          concept notes
DECISIONS.md         every design choice and why
LICENSE              MIT (code); data/hotpotqa_demo/ is CC BY-SA 4.0
CITATION.cff         how to cite judge-check
```

## Roadmap

- [x] 0. Scaffold, Compose, decision log
- [x] 1. Ingestion and retrieval (configurable chunking, local embeddings, pgvector, BM25, hybrid)
- [ ] 2. Grounded answer generation with citations (Ollama)
- [ ] 3. Variant generator: nine types, deterministic mutations first
- [ ] 4. Judge harness: pluggable judges, fixed-prompt and prompt-variation arms
- [ ] 5. Human adjudication queue
- [ ] 6. Metrics, report, minimum-detectable-difference simulation
- [ ] 7. Web UI and live four-judge comparison
