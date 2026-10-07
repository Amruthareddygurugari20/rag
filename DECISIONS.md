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

---

# Stage 1: ingestion and retrieval

## D-013: Built-in demo corpus is a 200-question subset of HotpotQA (dev, distractor)

**Stage:** 1

**Decision.** `data/hotpotqa_demo/` holds 200 questions from the HotpotQA dev set
(distractor setting) and all paragraphs in their contexts, pooled into one corpus (1,995
paragraphs). It is built by a deterministic, stdlib-only script
(`judge_check/datasets/hotpotqa.py`) from the **Hugging Face copy** of the dataset, not
from the official file. See "Provenance" below for exactly what that does and doesn't
establish.

**Why an external dataset and not hand-written questions.**
- The reference answers and supporting facts aren't ours, so nobody can say we wrote labels
  that suit our tool.
- Every HotpotQA question needs two facts from two paragraphs. The `partially_correct`
  variant (stage 3: one required fact right, the other wrong or missing) comes straight
  from that structure, with no need to invent it.
- The paragraphs are Wikipedia intros, dense with numbers, dates and named entities, which
  is what the deterministic mutators need.

**Licence.** HotpotQA is CC BY-SA 4.0 (checked on 2026-10-07 in two places: the README
of github.com/hotpotqa/hotpot, "The HotpotQA dataset is distribued under the CC BY-SA 4.0
license", and the source of hotpotqa.github.io). The subset is an adaptation, so it is
distributed under CC BY-SA 4.0 with attribution to Yang et al., EMNLP 2018, and a list of
changes. Both are in `data/hotpotqa_demo/README.md`. ShareAlike applies to the data files,
not to judge-check's code, which is MIT (D-026).

**Provenance: what was verified and what was not.** The official file
(`curtis.ml.cmu.edu/.../hotpot_dev_distractor_v1.json`) could not be downloaded: the host
refused connections from CI runners and from the development sandbox on 2026-10-07. We used
`hotpotqa/hotpot_qa` on Hugging Face (config `distractor`, split `validation`), pinned to
commit `1908d6afbbead072334abe2965f91bd2709910ab`, and converted it back to the original
layout with a field-for-field, unit-tested converter.
- *Verified:* the exact mirror file we used (its SHA-256 is in `MANIFEST.json`, together
  with the commit), and that it has 7,405 rows, the dev-set size the HotpotQA authors
  report. **That is a count match, not a content match.**
- *Not verified:* that the mirror's content is identical to the official file. We never
  had the official file, so we don't know its checksum. The checksum we pinned is the
  *mirror's*. We also haven't verified that the conversion to parquet preserved text
  exactly (whitespace, Unicode, sentence boundaries).
- *How to close the gap:* when the official host is reachable, build from the official
  JSON with the same seed and compare outputs byte for byte.

The filter outcomes (1 unresolvable supporting fact, 73 answers not found in the gold
sentences) look like an intact dataset. That is evidence against gross corruption, not
proof of fidelity.

**Selection filters.** Reject yes/no answers; supporting facts that don't resolve; gold
not exactly two paragraphs; answer not present (as whole words, after normalisation) in the
gold sentences; same title with different text. Counts per filter are in `MANIFEST.json`.

**Sampling: a seeded simple random sample.** Of the 6,873 questions that pass the filters
(the eligible population), 200 are drawn uniformly at random without replacement.
1. Sort by id, so the source file's order can't matter.
2. Permute with a seeded Fisher-Yates shuffle driven only by `random.Random(seed).random()`.
   Python guarantees that sequence across versions; it doesn't guarantee `shuffle` or
   `sample`.
3. Take the first 200, skipping title conflicts (none occurred in this draw).

The seed (20261007) is in `MANIFEST.json`.

*Revision history:* the first version took the first 200 eligible questions ordered by
`sha256(seed:id)`. A cryptographic hash of the id is unrelated to content, so that was
also, in effect, a random permutation, and not biased by the sort key. It was replaced
because an explicit seeded sample is easier to audit and to describe in a paper, and needs
no argument about hash functions.

*This draw's composition:* 180 bridge and 20 comparison questions, against a population
split of 85.1% / 14.9% (expected ≈ 29.8 comparison). P(≤ 20) = 0.027 exactly
(hypergeometric), about 0.05 two-sided. That's a chance under-representation of about two
standard deviations. The seed was **not** changed after seeing it: re-drawing until the
split looks nice makes the sample depend on the outcome. Report per-type results
separately. The sample represents the *filtered* population, not HotpotQA dev as a whole:
yes/no questions, mostly comparison type, are excluded.

**Known label noise** (detail and citations in the data card):
- Many questions have single-hop shortcuts (Min et al. 2019; Jiang & Bansal 2019; Trivedi
  et al. 2020).
- The reference answer is only one surface form of the correct answer.
- Supporting-fact annotations are imperfect.

Our filters remove what can be detected mechanically, and their counts are a measured
*lower bound* on that noise. What survives is what human adjudication (stage 5) is for.

**Why the build runs in CI.** The development sandbox can't reach the download host, while
GitHub Actions can. The `refresh-generated` workflow downloads, verifies, builds and
commits. Anyone can re-run the script locally and get identical bytes.

---

## D-014: Gold evidence is stored as character spans; gold chunks are derived

**Stage:** 1 · **Deviates from the original data model** (`question.gold_chunk_ids`)

**Decision.** `question_evidence(question_id, document_id, char_start, char_end)`. Gold
chunk ids for a given chunking are computed by interval overlap. For half-open intervals,
`chunk.start < ev.end AND ev.start < chunk.end` (`retrieval/evaluate.py::gold_chunk_ids`).

**Why.** Chunk ids only exist after chunking, and chunking is configurable. Storing
`gold_chunk_ids` would tie the ground truth to one chunking and silently invalidate it when
the chunking changed. Spans belong to the source document and stay valid under every
chunking. The cost is a join, which is cheap.

**Subtlety to know.** Under coarse chunking a gold chunk contains the evidence plus other
text. Under a chunking that cuts through a gold sentence, *both* pieces count as gold.
Recall numbers are therefore only comparable within one chunking.

---

## D-015: `chunk_set` and `embedding_run` are first-class; one dimension-less vector column

**Stage:** 1 · **Deviates from the original data model** (`chunk.embedding`)

**Decision.**
- `chunk_set(corpus, chunking config)` owns its chunks.
- `embedding_run(chunk_set, model_name, dimension, normalized, query_instruction)` owns its
  vectors in `chunk_embedding(embedding_run_id, chunk_id, embedding vector)`.
- The column is `vector`, with no fixed dimension.

**Why.** Comparing chunkings or embedding models is a core use. With `chunk.embedding` you
could store only one model's vectors at a time, and switching would overwrite the old
vectors. A dimension-less column lets one table hold 384- and 768-dimensional runs side by
side. Ingestion checks every vector against `embedding_run.dimension`, and queries always
filter by `embedding_run_id`, so vectors of different models are never compared.

**Trade-off.** pgvector can only build an ANN index on a fixed-dimension expression, so a
future index would be a partial expression index per run:
`CREATE INDEX ... USING hnsw ((embedding::vector(384)) vector_ip_ops) WHERE embedding_run_id = 7`.
Not needed yet (D-019).

---

## D-016: CPU-only PyTorch on Linux/Windows via uv's index pinning

**Stage:** 1

**Decision.** In `pyproject.toml`, `[tool.uv.sources]` sends `torch` to
`https://download.pytorch.org/whl/cpu` on every platform except macOS. macOS uses PyPI,
whose wheels are already CPU/Metal-only.

**Why.** The PyPI torch wheel for Linux bundles CUDA libraries, several GB of downloads a
CPU-only user never uses. That is incompatible with "free, local, ten minutes". The lock
file is generated where that index is reachable (the `refresh-generated` workflow).

---

## D-017: Embedding model `BAAI/bge-small-en-v1.5`, asymmetric query encoding

**Stage:** 1

**Decision.** The default embedder is bge-small-en-v1.5: 384 dimensions, about 33M
parameters, fast on CPU, MIT licence. Queries are embedded as
`"Represent this sentence for searching relevant passages: " + query`. Passages are
embedded as they are. The prefix is applied in exactly one function,
`SentenceTransformerEmbedder.embed_queries`, and can be switched off
(`--no-query-instruction`).

**Why asymmetric.** A question and the passage that answers it are different kinds of
text. "Where was the founder of X born?" shares few words, and little form, with "Y was
born in Bergen". BGE was fine-tuned contrastively with the instruction in front of queries,
which teaches the model to map instruction-prefixed text into the region where its
*answers* live, not where similar *questions* live. Prefixing passages too, or leaving the
query bare, means using the model differently from how it was trained.

**Measured, not assumed.** `tests/test_embeddings_model.py::test_query_instruction_ablation_on_demo_corpus`
reports recall@5, complete@5 and MRR on the demo corpus with and without the prefix, and
CI prints the numbers. The BGE v1.5 model card itself says the instruction matters less for
v1.5 than for v1.0, so the size of the effect is an empirical question we answer on our own
data.

---

## D-018: Normalise on write; search with pgvector's inner-product operator `<#>`

**Stage:** 1

**Decision.** Embeddings are L2-normalised at encode time (`normalize_embeddings=True`) and
checked (`check_unit_norm`, |‖v‖ − 1| < 1e-3) before they are stored. A model producing
non-unit vectors is refused, not silently mis-scored. Dense search orders by
`embedding <#> query`.

**The operators.**

| operator | computes | ranking |
|---|---|---|
| `<->` | Euclidean distance ‖a−b‖ | ascending |
| `<=>` | cosine distance 1 − a·b/(‖a‖‖b‖) | ascending |
| `<#>` | **negative** inner product −(a·b) | ascending |

**Why `<#>`.** For unit vectors a·b = cos(a, b). Also ‖a−b‖² = 2 − 2·cos, so on unit
vectors all three operators give *the same ranking*. `<#>` does the least work per row (one
dot product, no norms), and its index opclass is `vector_ip_ops`. It is negated so that,
like the others, ascending means best first, which is why we report `-(a <#> b)` as the
similarity.

**The catch.** This is only correct because the vectors are normalised. That's why it is
enforced at write time rather than assumed.

---

## D-019: Exact (sequential-scan) vector search, no ANN index yet

**Stage:** 1

**Decision.** No HNSW/IVFFlat index. Every dense query scans the run's vectors.

**Why.** The demo corpus has 4,542 chunks. An exact scan over 384-d vectors takes
milliseconds. An ANN index trades recall for speed, and in a tool that *measures* retrieval
and judging, approximate retrieval would add an error source nobody asked for. Add a
partial HNSW index per run (D-015) when corpora pass roughly 100k chunks.

---

## D-020: One tokenizer for all lexical work

**Stage:** 1

**Decision.** `tokenize(text) = re.findall(r"\w+", text.lower())`: Unicode word characters,
lowercased, with no stemming and no stopword removal. Both the in-memory reference and the
SQL BM25 call this one function.

**Why.** BM25 scores depend entirely on what counts as a term, and two implementations can
only be compared if they see identical tokens. Postgres's `to_tsvector` would stem and drop
stopwords with its own dictionary, which would make agreement with any Python reference
impossible to test. Stopwords need no special handling: IDF already makes them nearly
worthless.

---

## D-021: BM25 implemented in SQL over term-statistics tables

**Stage:** 1

**History.** Implemented first with `rank_bm25` in memory, which got hybrid fusion working
end to end (commit "Stage 1 phase A"). Then replaced by SQL. `rank_bm25` is still used as
the test oracle.

**Tables.**
- `bm25_doc(chunk_id, length)` holds |D|.
- `bm25_posting(term, chunk_id, tf)` holds f(t, D).
- `bm25_term(term, df, idf)` holds n(t) and IDF(t).
- `bm25_stats(n_docs, avgdl, avg_idf, epsilon)` holds N, avgdl and the floor.

Python tokenizes; SQL aggregates, computes IDF and scores.

### The formula

For query Q = q₁…qₙ (tokens, duplicates kept) and chunk D:

```
                 n          IDF(qᵢ) · f(qᵢ,D) · (k₁ + 1)
score(D,Q) =     Σ    ───────────────────────────────────────────
                i=1    f(qᵢ,D) + k₁ · (1 − b + b · |D| / avgdl)

IDF(t) = ln( (N − n(t) + 0.5) / (n(t) + 0.5) )
         if negative: ε · mean over all terms of IDF   (ε = 0.25)
```

### Deriving it at a whiteboard, piece by piece

1. **Start from tf · idf.** A term matters more the more often it appears in the document
   (tf) and the rarer it is in the collection (idf).
2. **IDF.** This is the Robertson–Spärck Jones weight with no relevance information: the
   log-odds that a term occurs in a document, ln((N − n)/n), with +0.5 smoothing so that
   n = 0 or n = N don't blow up. It is **negative when a term occurs in more than half the
   documents**. Implementations differ here:
   - Lucene uses ln(1 + (N−n+0.5)/(n+0.5)), which is always positive.
   - rank_bm25 replaces negatives with ε · mean IDF.
   We follow rank_bm25, so the test oracle applies directly.
3. **Saturate tf.** Raw tf is linear: 10 mentions would count 10× one mention. BM25 uses
   f·(k₁+1)/(f + k₁), which is 1 at f = 1 and approaches k₁+1 as f → ∞.
   **k₁ controls how fast term frequency saturates.**
   - k₁ = 0: the fraction becomes (f·1)/f = 1, so only *presence* matters.
   - large k₁: close to raw tf.
   The (k₁+1) in the numerator only rescales, so that one occurrence in an average-length
   document contributes exactly IDF. It doesn't change rankings.
4. **Normalise for length.** A long document mentions everything more often by chance. So
   k₁ in the denominator is multiplied by L = 1 − b + b·|D|/avgdl.
   **b controls how strongly length is normalised.**
   - b = 0: L = 1, length is ignored.
   - b = 1: tf is judged fully relative to |D|/avgdl.
   - An average-length document has L = 1 for any b.
5. **Sum over query terms.** A term repeated in the query is counted again. There is no
   separate query-tf saturation (the original paper's k₃ term), as in rank_bm25.

**Worked numbers** (N = 5, the term in 1 chunk, k₁ = 1.5, b = 0.75):
- IDF = ln(4.5/1.5) = ln 3 ≈ 1.099.
- f = 1 in an average-length chunk: 1.099 · 2.5/2.5 = 1.099.
- f = 3: 1.099 · 7.5/4.5 ≈ 1.83. Three times the mentions, only 1.67× the score.
- f = 1 in a chunk twice the average length: L = 1.75, so 1.099 · 2.5/(1 + 2.625) ≈ 0.76.

**Defaults: k₁ = 1.5, b = 0.75, ε = 0.25.**
- These are rank_bm25's defaults, which lets us test against it unchanged.
- Both k₁ and b are within the ranges usually recommended (k₁ ∈ [1.2, 2.0], b ≈ 0.75;
  Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009).
- Our chunks are short and of similar length (~2 sentences), so b has less effect here
  than on full documents.
- k₁ and b are query-time parameters (`bm25.search(..., k1=, b=)`); only ε is fixed when
  the index is built.

**Proof of correctness.** `tests/test_bm25_sql.py` checks that the SQL scores equal
`rank_bm25.BM25Okapi` scores (tolerance 1e-9) for every chunk and every one of 400 queries
(4,542 chunks × 400 queries = 1,816,800 scores)
on the demo corpus (each question and each reference answer), plus edge cases:
- negative IDF and the floor
- duplicate query terms
- unknown terms
- a zero-token chunk that still counts in N and avgdl
- an empty query

`bm25.explain()` returns the per-term breakdown of any score, and a test checks the
breakdown sums to the score.

---

## D-022: Hybrid = Reciprocal Rank Fusion, k = 60, 100 candidates per retriever

**Stage:** 1

**Decision.** `RRF(d) = Σᵣ 1/(60 + rankᵣ(d))` over the dense top 100 and the BM25 top 100.
Ties keep first-seen order (dense list first).

**Why ranks, not scores.** Cosine similarity (roughly 0.3–0.9 for this model) and BM25
(0 to ~30, unbounded, growing with query length) have unrelated scales. Mixing scores
needs a normalisation (min-max, z-score), which is a tuning choice in itself and sensitive
to outliers. RRF uses only ranks, needs no calibration, and rewards *agreement*: rank 2 in
both lists beats rank 1 in one list only (a unit test pins this).

k = 60 comes from Cormack, Clarke & Büttcher (SIGIR 2009). It flattens the head of each
list so that one retriever's top hit can't dominate. 100 candidates per retriever lets a
chunk ranked low by one retriever still win through the other.

---

## D-023: Default chunking: two-sentence windows, no overlap

**Stage:** 1

**Decision.** `SentenceWindow(sentences_per_chunk=2, overlap=0)`. When a document provides
`sentence_spans`, those are used (HotpotQA has them). Otherwise a rule-based splitter is
used, which knows abbreviations and initials and has known limits.

**Why.** The demo paragraphs average 4.05 sentences (median 4), so this gives 2.28
chunks per paragraph: 4,542 chunks averaging 39.4 tokens (median 38). Each question has 2
gold chunks (165 questions), 3 (29) or 4 (6). More than two happens when a question has more than two
supporting sentences and they fall into different windows.
- Whole paragraphs would make "cites a chunk that doesn't support the claim" (stage 3,
  variant 9) nearly impossible: there would be too few candidate chunks per document.
- Single sentences often lose their referent ("He was born in 1912.").
- No overlap means every character belongs to exactly one chunk, so citation ground truth
  is unambiguous.

Other strategies (`document`, `fixed_words`, any window size or overlap) are a JSON config
away (`--chunking`), and gold chunks are recomputed for each (D-014).

---

## D-024: Network-dependent steps run in GitHub Actions; required in CI, skippable locally

**Stage:** 1

**Decision.** Tests marked `model` need the real embedding model. Like `db` tests (D-009),
they skip when it can't be loaded, unless `JUDGE_CHECK_REQUIRE_MODEL=1`, which CI sets. CI
caches the Hugging Face model directory. A "retrieval demo" step at the end of CI ingests
the demo corpus, prints retrieval metrics with and without the query instruction, shows one
query under all three modes, and runs `scripts/show_similarities.py`. Its log is a readable
record of what the system does.

**Why.** The development sandbox can't reach Hugging Face or the PyTorch index, but CI can.
Making CI the place where model-dependent behaviour is *required* means a missing model
can't produce a green build.

---

## D-026: MIT for the code, CC BY-SA 4.0 for the demo data, and a CITATION.cff

**Stage:** 1

**Decision.** judge-check's code and docs are MIT (`LICENSE`). `data/hotpotqa_demo/` is
CC BY-SA 4.0, because it is adapted from HotpotQA and ShareAlike requires it. `LICENSE`
ends with an explicit carve-out, and the README has a table of what is under which licence.
`CITATION.cff` (validated with `cffconvert`) makes GitHub show "Cite this repository", and
lists HotpotQA under `references`.

**Why MIT.** The audience is researchers who will run this on their own evaluations and
may vendor parts of it. A permissive licence with no copyleft keeps that frictionless.

**Why the boundary is safe.** The demo data is a separate work that ships alongside the
code. Nothing in the code is derived from it. Using judge-check on your own data creates no
ShareAlike obligations, and the code's MIT licence doesn't relicense the data.

**Open.** The copyright holder is "the judge-check contributors", and the CITATION.cff
author is the GitHub handle. Both should become the author's full name before a release.

---

## D-025: Chunk length and BM25's length normalisation (b)

**Stage:** 1 · **Measured on the demo corpus (n = 200 questions), BM25 only**

Our default chunks average 39 tokens, far smaller than typical RAG chunks (often 200–500
tokens). This entry explains how that interacts with b, with measurements.

### 1. b normalises *relative* length, not absolute length

The length term is L = 1 − b + b·|D|/avgdl. A corpus of uniformly short chunks has
|D|/avgdl ≈ 1 everywhere, and then b does nothing whatever the absolute size. What matters
is the **spread** of |D|/avgdl, and our two chunkings have a similar spread:

| chunking | N | avgdl | p10 / p90 of \|D\|/avgdl | CV | max |
|---|---|---|---|---|---|
| 2-sentence windows (default) | 4,542 | 39.4 | 0.43 / 1.60 | 0.47 | 167 |
| whole paragraphs | 1,995 | 89.7 | 0.40 / 1.66 | 0.58 | 573 |

### 2. Absolute length still matters, through tf

In a 39-token chunk a query term rarely occurs twice, so most contributions are evaluated
at f = 1, where the term is (k₁+1)/(1 + k₁·L). **b still acts at f = 1**. With k₁ = 1.5 and
b = 0.75, a chunk at the 10th length percentile (L = 0.57) gets 2.5/1.86 ≈ 1.35× IDF per
matched term, and one at the 90th (L = 1.45) gets 2.5/3.18 ≈ 0.79× IDF, about a 1.7× gap.
What short chunks remove is mainly the *tf-saturation* side of BM25 (k₁ barely matters
when f is almost always 1), not the length side.

### 3. Measured: length bias exists in both, but matters more for paragraphs

At b = 0 (no normalisation), BM25's top 5 are skewed towards long chunks in both
chunkings: their mean length is **1.40× avgdl** for windows and **1.49×** for paragraphs.
At b = 0.75 the ratios are 1.05× and 0.90×. Retrieval quality (gold = chunks overlapping
the gold sentences; MRR of the first gold chunk; paired differences over the same 200
questions):

| chunking | MRR b=0 | b=0.25 | b=0.5 | b=0.75 | b=1.0 | Δ 0→0.75 | Δ 0.75→1.0 |
|---|---|---|---|---|---|---|---|
| windows | 0.679 | 0.711 | 0.737 | **0.742** | 0.719 | +0.063 (SE 0.019) | −0.023 (SE 0.011) |
| paragraphs | 0.638 | 0.720 | 0.757 | 0.791 | **0.814** | +0.153 (SE 0.024) | +0.023 (SE 0.012) |

(recall@5 for windows: 0.539 / 0.562 / 0.555 / 0.568 / 0.548; for paragraphs: 0.527 / 0.583
/ 0.618 / 0.642 / 0.645.)

- Normalising matters for both: z = 3.2 for windows, 6.3 for paragraphs.
- The effect is less than half as big for short chunks.
- The best b differs: windows peak near 0.75 and get *worse* at 1.0 (z ≈ −2.2), while
  paragraphs keep improving up to full normalisation.
- MRRs across the two chunkings are not comparable: the gold units differ.

### 4. Why (a hypothesis consistent with the data, not tested separately)

Robertson's distinction: a document can be long because of **verbosity** (same content,
more words, so tf is inflated and should be normalised away, b → 1) or because of
**scope** (more content, so genuinely more chances to be relevant, which shouldn't be
fully normalised).
- A Wikipedia intro paragraph is long mostly because of scope: more entities, dates and
  relations. Multi-entity HotpotQA questions then match long paragraphs incidentally.
  HotpotQA's distractor paragraphs were also *chosen* by bigram TF-IDF similarity to the
  question (Yang et al. 2018), so they are lexically close by construction. Strong length
  normalisation favours focused paragraphs, so b = 1 helps.
- A two-sentence window's scope is capped by construction. Its length varies mostly with
  sentence verbosity, and the answer is often *in* the long sentence. Full normalisation
  then starts penalising the right chunk, so b = 1 hurts.

### 5. Consequences

- **b, avgdl, N and df all belong to a chunk set.** Tune b per chunking. Don't carry over
  a b from document-level retrieval. BM25 scores are not comparable across chunk sets.
- **We keep b = 0.75.** It's the best of the five values on the default chunking, but the
  margin over 0.5 and 1.0 is about 2 paired SEs. It's also rank_bm25's default, which the
  oracle test relies on. We don't tune b further on these 200 questions: tuning on the same
  questions we report would overfit.
- **Don't transfer these numbers** to RAG systems with 200–500-token chunks. There, b
  sensitivity looks more like the paragraph row, or larger.
- **Noise level.** At n = 200 a single MRR has SE ≈ 0.025, and a paired difference between
  configurations ≈ 0.01–0.025. Differences below about 0.03 in the tables above shouldn't
  be read as real. This is the same reasoning judge-check exists to apply to LLM judges.

Reproduce: `judge-check eval-retrieval --corpus hotpotqa_demo --modes bm25 -k 5 --b 0.5`
(add `--chunking '{"strategy":"document"}'` for paragraphs, after ingesting with it).

---

## D-027: ANALYZE right after every bulk load

**Stage:** 1 · **Found by measurement**

**Decision.** `build_chunk_set` runs `ANALYZE chunk`. `bm25.build_index` runs
`ANALYZE bm25_doc, bm25_posting` *before* it aggregates them, and `ANALYZE bm25_term,
bm25_stats` after. `embed_chunk_set` runs `ANALYZE chunk_embedding`.

**Why.** Tables that have just been bulk-loaded have no planner statistics until
autovacuum gets to them, and inside an open transaction it never does. The planner then
assumes about 1 row. On the demo corpus (whole-paragraph chunking, ~120k postings), the
df/IDF aggregation got this plan:

```
no stats:       GroupAggregate ← Incremental Sort ← Nested Loop ← Index Only Scan (rows=1)   30.79 s
after ANALYZE:  HashAggregate ← Nested Loop(Aggregate(bm25_doc), Seq Scan bm25_posting)      0.11 s
```

That's 280× slower, for identical results. The test suite didn't catch it because the
test database had statistics left over from earlier runs. `ANALYZE` (unlike `VACUUM`) can
run inside a transaction, so ingestion stays atomic.
