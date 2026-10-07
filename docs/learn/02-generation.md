# Grounded generation, and keeping its output honest

Notes for stage 2. Stage 2 builds the machinery for calling LLMs. That machinery is shared
with stage 4 (judging), so most of what matters here is less about generating text and more
about what a later measurement can trust. Decisions are cited as D-0xx.

## 1. What "grounded" means here

The model sees only the retrieved chunks, labelled `C1`…`Ck`. It must answer from them and
cite the labels it used, or say the passages don't contain the answer. That makes an answer
**checkable**: every claim points at a passage you can open.

Why short labels and not chunk ids: models copy "C3" reliably and mangle "chunk 1849203". The
label→id mapping is exact and stored (`retrieved_chunk_ids` in prompt order,
`cited_chunk_ids` after mapping).

## 2. Strict parsing: failures are data

The output must be JSON `{answer, citations, answerable}`, and decoding is constrained by a
JSON schema (Ollama `format`, Azure `response_format`). These are **parse failures**:

- a citation to a label that wasn't shown (`C9` when only C1–C5 existed);
- bad JSON, or the wrong types;
- `answerable: true` with no citations: an answer that claims support and cites none.

A failure is stored with its reason and the raw output, and the answer is NULL. **Nothing
is repaired.** Dropping a bad citation and keeping the rest would make the model look better
grounded than it was. That's the same mistake as an LLM judge that accepts on-topic wrong
answers, made one level down.

## 3. A generated answer is not a label

A model wrote it, so its correctness is unknown (D-000). Calling it `correct` because it
reads well would make the measurement circular. It would also make it uninterpretable: a
judge "wrongly rejecting" it might be correctly rejecting a bad generation. Generated
answers enter the evaluation set only through **human adjudication** (stage 5), as
`label_source = "human"` (D-032).

Why include them at all? **Realism.** Variants built by construction are cleaner than real
RAG output. Measuring judges only on clean text might not transfer to the messy output they
face in production. So judge error rates are reported separately by label provenance. If a
judge's false-reject rate on constructed-correct answers differs from its rate on
human-labelled generated-correct answers, that gap is a **finding** about how
representative synthetic benchmarks are, a weakness others concede but don't measure.

Two conditions keep that comparison honest:

- **The generator must produce plausible output, and that's measured, not assumed**
  (D-032 A-revised). A 0.5B model's answers are trivially rejectable, which would make the
  "realistic" arm *easier* than the constructed one and the gap come out backwards. A list
  of allowed model names wouldn't generalise (a 1.5B model would slip through). So the gate
  is reviewer-measured plausibility: a generator counts only once the Wilson 95% lower bound
  of its plausible rate is ≥ 0.80 over ≥ 40 reviews. That threshold was fixed before any
  model was reviewed.
- **Parse failures are excluded and counted** (D-032 B). They're unusable, not wrong.
  Dropping them silently skews the pool towards questions the generator found easy, so the
  exclusion rate is reported per model version as a selection-bias statement.

## 4. Attribution: every number must name its model

A judge's error rate is a property of **one exact model version**. Ollama tags move (`ollama
pull` swaps the weights) and Azure upgrades deployments in place. So every stored LLM output
records (D-030):

| field | example |
|---|---|
| `model_requested` | `qwen2.5:7b` (what we asked for) |
| `model_reported` | `qwen2.5:7b` (what the provider said answered) |
| `model_version` | `qwen2.5:7b@sha256:…` (Ollama digest); `gpt-4o-2024-08-06+fp_…` (Azure) |
| `temperature`, `seed`, `max_tokens` | always explicit; the seed is always sent |
| `prompt_sha256` | the exact prompt file that was rendered |

They live on the **row**, not on a config table, because configs change and old rows stay.
They're enforced three times: the `Completion` object can't be built without them, the
database rejects NULL or empty values, and a test covers every table that stores LLM output,
failing if a new one is added without coverage.

## 5. One call path, versioned prompts

For the judge comparison to be valid, judges must differ **only** in the model (or, in the
prompt-variation arm, only in the prompt). So (D-031):

- prompts are TOML files with a version; a lock file pins each released file's hash, and
  editing a released prompt fails CI (change it = new version);
- `run_prompt` is the only function that calls a model. A test parses the source and fails
  on any other call site.

This is the same technique as the heuristic wall (D-033): when a rule matters to the
measurement, a test that reads the code enforces it, because documentation drifts.

## 6. What to say in an interview

*"Generation in a judge-evaluation tool is mostly about provenance. Every output names the
exact model weights, decoding parameters and prompt hash; malformed outputs are recorded,
never repaired; generated text is never a label, only realistic material for humans to
label; and the rules that protect the measurement (one call path, no heuristic labels,
immutable prompts) are enforced by tests that read the source, not by convention."*
