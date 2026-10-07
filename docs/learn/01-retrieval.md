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
  often score 0.3 to 0.6 with this model (the demo printout has unrelated pairs at 0.35–0.55), because all its vectors share a common
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
off, you use the model differently from how it was trained.

**What we measured: nothing.** On the demo corpus, with the prefix vs without, paired over
the same 200 questions: recall@5 identical (0.745 both), complete@5 0.515 vs 0.520, MRR
+0.009 (SE 0.005, z 1.6, not significant). BAAI's own documentation predicts roughly this.
For v1.5, "No instruction only has a slight degradation in retrieval performance compared
with using instruction", and "The best method to decide whether to add instructions for
queries is choosing the setting that achieves better performance on your task." HotpotQA
questions are full, well-formed sentences, not the short keyword queries the instruction is
recommended for. We keep the prefix (it's how the model was trained, and it costs nothing),
but the lesson is the measured null, not the theory (D-017).

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

## 5. Hybrid: Reciprocal Rank Fusion, and when it hurts

The two retrievers fail differently: dense fails on rare exact strings, BM25 fails on
paraphrase. That's the usual case for combining them. Their scores aren't comparable
(cosine ≈ 0.3 to 0.9, BM25 ≈ 0 to 60 and unbounded), so RRF fuses *ranks*:
`Σ 1/(60 + rank)`, which rewards agreement between the retrievers. Run
`judge-check search "..." --compare` to see each list and how fusion reorders them (D-022).

**On our corpus it hurts.** Equal-weight hybrid is worse than dense alone: MRR −0.051
(z −2.5), complete@5 −0.075 (z −2.3). The reason is the corpus, not fusion. HotpotQA's
distractor paragraphs were *chosen by TF-IDF similarity to the question*, so the negatives
are adversarial for a lexical retriever, BM25 is much weaker than dense here (MRR −0.117,
z −4.2), and equal weighting pulls its mistakes into the fused list. Weighted fusion
(dense-weighted RRF, normalised linear combination) was tested with cross-fitting. See
D-028 for the pre-registered prediction and the result. The fusion default is
**unresolved** until we test a corpus with naturally sampled negatives. What to say in an
interview: *"hybrid helps" is an empirical claim about a corpus, and this corpus was built
to break the lexical half.*

## 6. Chunking and gold labels

A chunk is a span `[start, end)` of a document. Gold evidence is also a span, so whether
a chunk is "gold" depends on the chunking, and is computed by interval overlap (D-014).
Consequence: retrieval metrics are only comparable within one chunking. Finer chunks make
"complete@k" harder (more gold chunks to find) but make citations more precise.

## 7. Chunk length and BM25's b

`b` normalises by *relative* length (|D|/avgdl), so short chunks don't switch it off. What
short chunks remove is mostly tf saturation, because a term rarely appears twice in 40
tokens. Measured: normalising length (b 0 → 0.75) raises MRR by 0.063 (z 3.2) on our
two-sentence chunks and by 0.153 (z 6.3) on whole paragraphs. Short chunks peak near 0.75;
paragraphs keep improving up to b = 1. The likely reason is verbosity vs scope (D-025).

## 8. Metrics we report

- **recall@k**: the mean fraction of a question's gold chunks found in the top k.
- **complete@k**: the fraction of questions with *all* gold chunks in the top k. This is
  what multi-hop needs. If you miss one of the two facts, a faithful answer is impossible.
- **MRR**: the mean of 1/rank of the first gold chunk.

These numbers matter for judge-check because a grounded answer (stage 2) can only be as
good as what was retrieved, and a judge (stage 4) shown the retrieved chunks is judging
against that evidence.

## 9. How to compare two retrieval configurations without fooling yourself

This is the part of stage 1 that carries over to the rest of judge-check. Every rule below
was needed at least once while building it.

1. **Pair your comparisons.** Two configurations run on the same questions. Compute the
   per-question difference and its SE (`paired_difference`). The shared question difficulty
   cancels, so the SE of a difference (~0.01-0.02 here) is much smaller than either MRR's
   own SE (~0.025). Comparing two independent confidence intervals throws that away.
2. **Know your noise floor before you read a table.** At n = 200, a single MRR has SE about
   0.025. "b barely matters" was an eyeball claim from a table; paired SEs showed b = 0 vs
   0.75 is a z = 3.2 effect. The correction is in D-025.
3. **Don't pick the best row and report it.** Choosing a fusion weight on the same
   questions you report on is tuning on the test set. Cross-fit: choose on one half, score
   on the other. One split is noisy at n = 100, so repeat over random partitions and look at
   the spread.
4. **Watch tie-breaks.** When several settings tie, "take the first" silently prefers
   whatever is listed first. Ties now resolve towards the untuned default, so a move away
   from it has to be earned.
5. **Write the prediction down before the number arrives.** D-028's prediction was
   committed before the sweep output was read. Then nobody, including you, can say the
   story was fitted to the data.
6. **Know what your sample can support.** 20 comparison questions give a ±0.19 interval;
   the smallest detectable bridge-vs-comparison gap is about 0.29. That's a *minimum
   detectable difference*, the same quantity judge-check reports for LLM judges in stage 6.
   This subset supports pooled claims only.
