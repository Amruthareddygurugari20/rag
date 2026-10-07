# Retrieval, from first principles

Notes for stage 1: what each part of retrieval does and why, at the level you would need
to explain it in an interview. Run `backend/scripts/show_similarities.py` while you read.
Its output has the real numbers. Design decisions are cited as D-0xx (see `DECISIONS.md`).

## 1. What an embedding actually is

An embedding model is a function `f(text) → ℝ³⁸⁴` for bge-small-en-v1.5. The 384 numbers
have no individual meaning: dimension 17 is not "about sports". What matters is geometry.
The model was trained contrastively. Given (query, relevant passage) pairs, training pulls
each pair's vectors together and pushes the query away from the other passages in the
batch (in-batch negatives). After enough pairs, texts that answer each other end up
pointing in similar *directions*.

Mechanically: text → tokens → a BERT-style transformer → one vector per token → pooled
into one vector (BGE uses the [CLS] token's vector) → L2-normalised. Different texts of
any length give vectors of the same size, so any two can be compared.

Three facts to remember:
- **It's lossy.** A paragraph becomes 384 floats. Exact facts (a specific number, a rare
  name) are what gets lost first. That is why we also use BM25.
- **It's not calibrated.** A cosine of 0.62 doesn't mean "62% relevant". Unrelated texts
  often score 0.3 to 0.5 with this model, because all its vectors share a common
  direction. Use similarities for *ranking*, never as absolute thresholds. The script's
  passage-by-passage matrix shows this.
- **It's model-specific.** Vectors from two models (or two versions of one model) live in
  different spaces and must never be compared. That's why `embedding_run` exists (D-015).

## 2. Why cosine similarity is the comparison

The training objective rewards *direction*: it uses cosine (or dot product on normalised
vectors), scaled by a temperature, inside a softmax. So direction is what carries meaning,
and the measure has to match the objective.

    cos(a, b) = a·b / (‖a‖ ‖b‖)       ∈ [−1, 1]

Length (‖a‖) carries little meaning and is noisy, for example by text length or token
frequency. Cosine discards it. Raw dot product would let a "loud" vector win against every
query (see section 4 of the script: scale a vector ×3 and its dot products triple while
its cosines don't move).

**Why we then use the inner-product operator.** If every vector is normalised to length 1
at write time, then ‖a‖ = ‖b‖ = 1 and cos(a, b) = a·b. A dot product is cheaper than a
cosine (no norms). Euclidean distance gives the same ranking too, since
‖a − b‖² = 2 − 2 cos(a, b) for unit vectors. So we normalise once, at write time, check
it, and search with pgvector's `<#>` (negative inner product) (D-018). If a model ever
produced non-unit vectors, inner product would silently compute the wrong thing, which is
why ingestion refuses them.

## 3. Asymmetric encoding: the query instruction

BGE embeds queries as `"Represent this sentence for searching relevant passages: " + q`
and passages bare. Why treat them differently?

A question and its answer aren't paraphrases. "Where was the founder of Alpha Corp born?"
and "Ann Lee was born in Bergen." share few words and have a different form. Symmetric
similarity ("how alike are these two texts?") would rank *other questions* about Alpha
Corp above the answer. The instruction signals "this is a query, put it where its answers
are", and the model was fine-tuned with exactly that prefix on queries. If you leave it
off, you use the model differently from how it was trained. The size of the effect is
measured on our data, not assumed: CI prints the ablation from
`test_query_instruction_ablation_on_demo_corpus` (D-017).

What to say in an interview: *asymmetric models encode the two sides of retrieval
differently, because a question and its answer are different kinds of text. With BGE the
asymmetry is a query-side prefix. Other models do it with separate query and passage
prefixes (E5: `query:` / `passage:`) or even separate encoders (DPR).*

## 4. BM25: the lexical half

Dense retrieval misses exact matches: rare entities, numbers, codes. BM25 is the opposite.
It only sees exact token overlap, weighted by rarity, saturated in frequency and
normalised by length. The full formula and derivation are in D-021. In one line:
*tf·idf, with tf saturating (k₁) and adjusted for document length (b).*

Our BM25 runs in SQL over term-statistics tables, and a test shows it agrees with
`rank_bm25` to 1e-9 on every (query, chunk) pair of the demo corpus.

## 5. Hybrid: Reciprocal Rank Fusion

The two retrievers fail differently: dense fails on rare exact strings, BM25 fails on
paraphrase. Combining them is cheap insurance. Their scores aren't comparable (cosine ≈
0.3 to 0.9, BM25 ≈ 0 to 30 and unbounded), so RRF fuses *ranks*:
`Σ 1/(60 + rank)`. Agreement between the retrievers is rewarded. Run
`judge-check search "..." --compare` to see each list and how fusion reorders them (D-022).

## 6. Chunking and gold labels

A chunk is a span `[start, end)` of a document. Gold evidence is also a span, so whether
a chunk is "gold" depends on the chunking, and is computed by interval overlap (D-014).
Consequence: retrieval metrics are only comparable within one chunking. Finer chunks make
"complete@k" harder (more gold chunks to find) but make citations more precise.

## 7. Metrics we report

- **recall@k**: the mean fraction of a question's gold chunks found in the top k.
- **complete@k**: the fraction of questions with *all* gold chunks in the top k. This is
  what multi-hop needs. If you miss one of the two facts, a faithful answer is impossible.
- **MRR**: the mean of 1/rank of the first gold chunk.

These numbers matter for judge-check because a grounded answer (stage 2) can only be as
good as what was retrieved, and a judge (stage 4) shown the retrieved chunks is judging
against that evidence.
