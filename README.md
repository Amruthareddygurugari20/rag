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

> **Status: stage 0 of 7: scaffold only.** The quickstart below sets up the database and
> API skeleton. Ingestion, generation, variants, judging, adjudication, metrics and the UI
> arrive stage by stage (see [Roadmap](#roadmap)). Design rationale lives in
> [DECISIONS.md](DECISIONS.md).

## Quickstart

Requirements: Docker, [uv](https://docs.astral.sh/uv/), Python ≥ 3.11.
[Ollama](https://ollama.com) will be needed from stage 2.

```bash
cp .env.example .env          # defaults work as-is
make up                       # Postgres 17 + pgvector on localhost:5433
make migrate                  # enables pgvector in the judge_check database
make api                      # http://localhost:8000
curl localhost:8000/health    # {"status":"ok","database":"ok","pgvector":"0.8.x"}
make test
```

No `make`? Every target is a one-liner in the [Makefile](Makefile).

## How it works (planned)

1. **Ingest** your documents. Chunk, embed locally, store in pgvector, retrieve with hybrid
   dense + BM25 search.
2. **Generate** grounded answers with chunk-id citations.
3. **Construct variants** of each answer in nine types: four that a good judge should
   accept, five that it should reject. Wrong answers are made by code, never labelled by a
   model.
4. **Judge** every variant with each judge under an identical prompt, plus a separate arm
   that varies the prompt with the model fixed.
5. **Adjudicate** a sample by hand, which anchors the error rates to human judgement.
6. **Report** FAR/FRR with bootstrap CIs per judge and variant type, Cohen's κ against
   humans, cost and latency, and the minimum detectable accuracy difference (Monte Carlo).

## Repository layout

```
backend/                 Python package (FastAPI, SQLAlchemy, Alembic)
  src/judge_check/       application code
  migrations/            Alembic migrations, one per schema change
  tests/
docker/postgres/         database init (creates the test DB)
docker-compose.yml       Postgres+pgvector; Ollama behind --profile ollama
DECISIONS.md             every design choice and why
```

## Configuration

All settings are environment variables prefixed `JUDGE_CHECK_`. See
[.env.example](.env.example).

## Roadmap

- [x] 0. Scaffold, Compose, decision log
- [ ] 1. Ingestion and retrieval (configurable chunking, local embeddings, pgvector, BM25, hybrid)
- [ ] 2. Grounded answer generation with citations (Ollama)
- [ ] 3. Variant generator: nine types, deterministic mutations first
- [ ] 4. Judge harness: pluggable judges, fixed-prompt and prompt-variation arms
- [ ] 5. Human adjudication queue
- [ ] 6. Metrics, report, minimum-detectable-difference simulation
- [ ] 7. Web UI and live four-judge comparison
