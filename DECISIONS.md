# Design decisions

Every design choice in judge-check is logged here with the reason for it. Each entry records
what was decided, what else was considered, why this option won, and what would make us
revisit it. Entries are never deleted: if a decision is reversed, a new entry supersedes it
and the old one is marked as superseded.

---

## D-000: Ground truth is constructed, never inferred by a model

**Stage:** foundational · **Status:** binding

**Decision.** Each `answer_variant.is_correct` label comes from how the variant was
*constructed*. No model decides it. A wrong answer is made by deterministically mutating a
known-correct one in code (swap a number, a date, a named entity), so we know for certain
that it is wrong. An LLM may *phrase* a variant (hedges, plausible unsupported claims), but
the label is fixed before the LLM is called and the LLM's output never changes it.

**Why.** judge-check measures how often an LLM judge is wrong. If an LLM also produced the
reference labels, the measured error rate would be judge-vs-model agreement, not
judge-vs-truth, and the tool would be circular. Human adjudication (stage 5) is the
independent check that the constructed labels hold up.

**Consequence.** Every variant stores `how_generated`, and any code path that writes
`is_correct` must be deterministic code. The mutators get exhaustive unit tests.

---

## D-001: Monorepo with `backend/` and `frontend/`

**Stage:** 0

**Decision.** One repository. Python lives in `backend/` with a `src/` layout
(`backend/src/judge_check`). The React + TypeScript app goes in `frontend/`, created in
stage 7 and not before.

**Alternatives.** Two repos (needless overhead for a single tool); Python at the repo root
(gets messy once `package.json` arrives).

**Why `src/` layout.** Tests import the *installed* package, not whatever happens to be in
the working directory, so a missing file in the package can't hide behind the current
directory.

---

## D-002: `uv` for Python dependency management

**Stage:** 0

**Decision.** `backend/pyproject.toml` + `uv.lock`. Dev tools sit in a `dev` dependency
group.

**Alternatives.** pip + requirements.txt (no lockfile semantics); Poetry (slower and
heavier, no real advantage here).

**Why.** The lockfile makes installs reproducible, which matters for a measurement tool:
someone re-running an evaluation should get the same library versions. uv is also fast
enough that "ten minutes to first result" is realistic.

---

## D-003: One database, Postgres 17 + pgvector

**Stage:** 0

**Decision.** Postgres with the pgvector extension (`pgvector/pgvector:pg17` image) holds
everything: documents, chunk embeddings, variants, judgements, adjudications and reports.

**Alternatives.** A dedicated vector store (Qdrant, Chroma, Weaviate) next to a relational DB.

**Why.**
- The data is relational at heart. A judgement points to a variant, which points to a
  question, which points to gold chunks. Keeping vectors in the same database means one
  transaction, one backup and joins across all of it, e.g. "false accepts where the gold
  chunk ranked below 5".
- At our scale (thousands to low millions of chunks) pgvector's HNSW index is more than
  fast enough. A dedicated vector DB pays off at a scale and QPS this tool never reaches.
- BM25 / full-text search (stage 1) can also live in Postgres, so hybrid retrieval needs no
  second system.

**Revisit if.** Corpora reach tens of millions of chunks, which is out of scope for an
evaluation tool.

---

## D-004: Postgres published on host port 5433

**Stage:** 0

**Decision.** Compose maps the container's 5432 to host port **5433**.

**Why.** Researchers often already run Postgres locally on 5432. A port clash on first run
is the kind of friction that breaks a ten-minute quickstart.

---

## D-005: Compose runs infrastructure, the Python app runs on the host

**Stage:** 0 · **Revisit:** stage 7

**Decision.** `docker compose up` starts Postgres (plus Ollama, optionally). The API runs on
the host via `uv run`.

**Alternatives.** Containerise the API too.

**Why.** During development we want `--reload`, a debugger, and the sentence-transformers
model cache in the user's normal home directory instead of a container volume. A
containerised API that runs the whole stack with one command will be added in stage 7 once
there is a UI worth shipping.

---

## D-006: Ollama is optional in Compose, behind a profile

**Stage:** 0

**Decision.** The `ollama` service only starts with `--profile ollama`. By default the app
talks to `http://localhost:11434`, which works the same for native Ollama and the container.

**Why.** On macOS, Docker cannot use the Apple GPU (Metal), so containerised Ollama runs on
CPU and is several times slower than a native install. Linux users with no Ollama installed
get a one-flag option.

---

## D-007: Synchronous SQLAlchemy 2.0 over psycopg 3

**Stage:** 0

**Decision.** Sync engine and sessions. FastAPI runs sync endpoints in its threadpool.

**Alternatives.** async SQLAlchemy + asyncpg.

**Why.** The expensive work here is CPU-bound (embeddings) or a slow remote call (LLM
judging). Async DB access speeds up neither. Judge calls will be parallelised explicitly in
stage 4, where the concurrency limit is a property of the provider, not of the web server.
Sync code is also easier to read, test and step through. psycopg 3 is used because the
`pgvector` Python package supports it and it is the maintained driver.

---

## D-008: Alembic migrations, schema grows one stage at a time

**Stage:** 0

**Decision.** All schema changes are Alembic migrations, and each table arrives with the
stage that first uses it. Migration `0001` only runs `CREATE EXTENSION vector`. The
extension is enabled in a migration, not in a Docker init script, so it is also enabled on
a Postgres the user brings themselves.

**Why.** A `vector(n)` column fixes its dimension when the table is created, and the
dimension depends on the embedding model, which is chosen in stage 1. Creating every table
up front would mean guessing.

---

## D-009: Separate test database, and DB tests may skip locally but never in CI

**Stage:** 0

**Decision.** Tests use `judge_check_test` (created by `docker/postgres/init.sql`, or by the
CI service container), migrated to `head` at the start of each session. Tests that need
Postgres are marked `db`. If the database is unreachable they **skip** locally but **fail**
when `JUDGE_CHECK_REQUIRE_DB=1`, which CI sets.

**Why.** Pure-logic tests (above all the mutators) should run anywhere with no setup. But a
silently skipped test would let a broken database path pass CI.

---

## D-010: Configuration via `JUDGE_CHECK_*` environment variables

**Stage:** 0

**Decision.** `pydantic-settings`, prefix `JUDGE_CHECK_`, optional `.env`. A setting is added
only when code reads it.

**Why.** It is typed and validated at startup, and the prefix avoids picking up an
unrelated `DATABASE_URL` from the user's shell. A test checks this.

---

## D-011: Local and free by default. Azure OpenAI is the only optional cloud provider

**Stage:** 0 · **Status:** binding

**Decision.** Defaults: sentence-transformers on CPU for embeddings, Ollama for generation
and judging. Azure OpenAI can be plugged in behind the same provider interface (stage 2/4).
There is no AWS, GCP or Kubernetes in the code or the docs.

**Why.** Researchers should be able to audit their judge without an account or a budget.
Any paid provider is opt-in.

---

## D-012: CI tests on Python 3.11

**Stage:** 0

**Decision.** GitHub Actions runs lint, format check and tests on Python 3.11 against a
pgvector service container.

**Why.** 3.11 is the minimum we promise. Testing on the minimum catches use of newer syntax
or stdlib features. Newer interpreters are covered by local development (currently 3.13).
