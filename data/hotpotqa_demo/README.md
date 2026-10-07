# HotpotQA demo subset (200 questions)

The built-in demo corpus for judge-check: 200 questions from the HotpotQA dev set
(distractor setting), together with every paragraph that appears in their contexts.

## Attribution and licence

This subset is derived from **HotpotQA**:

> Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan
> Salakhutdinov, Christopher D. Manning. *HotpotQA: A Dataset for Diverse, Explainable
> Multi-hop Question Answering.* EMNLP 2018. https://hotpotqa.github.io/

HotpotQA is distributed under the **Creative Commons Attribution-ShareAlike 4.0
International** licence (CC BY-SA 4.0): https://creativecommons.org/licenses/by-sa/4.0/
(as stated on hotpotqa.github.io and in the README of github.com/hotpotqa/hotpot, checked
2026-10-07). The paragraph text comes from English Wikipedia, which is also CC BY-SA.

Because of the ShareAlike term, **the files in this directory are distributed under CC BY-SA
4.0**, whatever licence the rest of the repository uses. If you redistribute or adapt them,
keep this attribution and licence.

### Source file, and what was and wasn't verified

The original distribution is `hotpot_dev_distractor_v1.json` from
`http://curtis.ml.cmu.edu/datasets/hotpot/`. On 2026-10-07 that host did not accept
connections, neither from GitHub's CI runners (four connection timeouts) nor from the
development environment. So the subset is built from the **Hugging Face copy**:
`hotpotqa/hotpot_qa`, config `distractor`, split `validation`, at the repository commit
recorded in `MANIFEST.json`. `backend/scripts/hf_hotpotqa_to_json.py` converts that parquet
back to the original JSON layout, a field-for-field mapping documented and unit-tested in
the repo. The builder runs on the converted file.

**Verified:**
- The parquet file we used is pinned: its SHA-256 (`source_sha256`) and the Hugging Face
  commit (`source_revision`) are in `MANIFEST.json`. Anyone can fetch the identical file.
- It has 7,405 rows, which matches the dev-set size reported by the HotpotQA authors.
  **That is a count match, not a content match.**
- Every row has the expected fields and types, and the filter outcomes look like an intact
  dataset: 1 unresolvable supporting fact, 73 answers not found in the gold sentences. That
  is evidence *against* gross corruption, not proof of fidelity.

**Not verified:**
- That the Hugging Face file's content is identical to the official
  `hotpot_dev_distractor_v1.json`. We have never downloaded the official file, so we don't
  know its checksum, and the checksum in `MANIFEST.json` is the *mirror's*.
- That the Hugging Face conversion preserved text exactly (whitespace, Unicode
  normalisation, sentence boundaries, row order).

**How to close the gap** when the original host is reachable: run the builder directly on
the official JSON with the same seed (`--seed`, see `MANIFEST.json`) and compare the outputs
to these files. Byte-identical `corpus.jsonl` and `questions.jsonl` would show the subset
is unaffected by the mirror. Comparing the whole converted JSON with the official file
would verify the mirror itself.

Licence and attribution are the same either way: the Hugging Face copy is the HotpotQA
dataset, redistributed under the same CC BY-SA 4.0 licence.

### Changes made to the original

- Selected 200 of the 7,405 dev questions: deterministic filters, then a seeded simple
  random sample (described below).
- Joined each paragraph's sentence list into a single string, inserting a space only between
  sentences that had no whitespace between them, and recorded each sentence's character
  span.
- Pooled all context paragraphs of the selected questions into one corpus keyed by
  Wikipedia title.
- Converted supporting facts `(title, sentence index)` into character spans
  (`gold_evidence`).
- Text is otherwise unmodified.

## Files

| File | Generated? | Content |
|---|---|---|
| `corpus.jsonl` | yes | `{id, title, text, sentence_spans}`, one paragraph per line |
| `questions.jsonl` | yes | `{id, question, reference_answer, gold_evidence[{document_id,start,end}], metadata}` |
| `MANIFEST.json` | yes | source URL, revision and SHA-256, seed, builder checksum, filter counts |
| `README.md` | no | this data card |

Generated files are produced by `backend/src/judge_check/datasets/hotpotqa.py` through the
`refresh-generated` GitHub workflow. Do not edit them by hand.

## How the 200 were chosen

Each source question is checked against these filters in order. Rejection counts for each
filter are in `MANIFEST.json` → `stats.rejected`.

1. **`yes_no_answer`**: answer is "yes" or "no". A yes/no answer can't be cited to a span,
   and flipping it is too easy to be an informative wrong answer.
2. **`supporting_fact_unresolvable`**: a supporting fact names a title that isn't in the
   context, or a sentence index that is out of range or points to an empty sentence.
3. **`gold_not_exactly_two_paragraphs`**: the supporting facts don't come from exactly two
   paragraphs. Two paragraphs is HotpotQA's own design, and judge-check's `partially_correct`
   variant needs those two facts.
4. **`answer_not_in_gold_sentences`**: the normalised answer does not occur, as whole
   words, in the gold sentences. Without this the citations could not be ground truth.
5. **`title_text_conflict`**: two paragraphs with the same title but different text, either
   inside one question or across selected questions. The pooled corpus is keyed by title.

**Sampling.** The 200 are a **simple random sample without replacement** from *all*
questions that pass filters 1-4 (the "eligible population", whose size is in
`MANIFEST.json` → `stats.eligible_population`). The procedure:

1. Sort the eligible questions by id, so the source file's row order can't matter.
2. Permute them with a seeded Fisher-Yates shuffle driven by
   `random.Random(seed).random()`. Python guarantees that sequence for a given seed across
   versions; it does not guarantee `random.shuffle` or `random.sample`.
3. Take questions in permuted order until 200 are selected. A question whose paragraphs
   conflict with an already-selected one is skipped (filter 5).

The seed is `MANIFEST.json` → `sample_seed` (also `stats.sampling.seed`). The first n of a
uniformly random permutation is a simple random sample. The only departure is the
conflict skip, and how many times it fired is in `stats.rejected`.

**Representativeness.** `stats.eligible_by_type` (the population) sits next to
`stats.selected_by_type` (the sample), so the bridge/comparison split of the 200 can be
compared with the population it was drawn from. The sample represents the *eligible*
population, not HotpotQA dev as a whole: the filters remove all yes/no questions, which
are mostly comparison questions. Retrieval and judge numbers on this corpus should be
reported as being on "HotpotQA dev, filtered as described, n = 200".

**This draw (seed 20261007): bridge/comparison.**

|  | eligible population | sample |
|---|---|---|
| bridge | 5,850 (85.1%) | 180 (90.0%) |
| comparison | 1,023 (14.9%) | 20 (10.0%) |

Under simple random sampling the expected number of comparison questions is 29.8. Drawing
20 or fewer has probability 0.027 (exact hypergeometric), about 0.05 two-sided. So this
draw under-represents comparison questions by roughly two standard deviations. It is
chance, not a bug, and the seed was fixed before the draw was seen. We deliberately did
**not** re-draw with another seed: choosing a seed after seeing its outcome would make the
sample depend on what we wanted to see. Per-type numbers should be reported separately,
and comparison-question results at n = 20 carry wide intervals.

## Known label noise in HotpotQA

These are documented problems with the source data. They matter here because judge-check
treats these labels as ground truth.

- **Shortcuts.** Many questions can be answered from one paragraph, without the intended
  two-hop reasoning (Min et al., *Compositional Questions Do Not Necessitate Multi-hop
  Reasoning*, ACL 2019; Jiang & Bansal, *Avoiding Reasoning Shortcuts*, ACL 2019; Trivedi et
  al., *Is Multihop QA in DiRe Condition?*, EMNLP 2020). For us this weakens the
  `partially_correct` construction for those questions. It does not make labels wrong.
- **Reference answers are one surface form.** A correct answer can be phrased differently
  (aliases, a longer or shorter span), and the reference shows only one. That is exactly why
  judge-check never scores by string match against the reference.
- **Supporting facts are imperfect.** Annotated sentences are sometimes incomplete or
  unnecessary. Filters 2 and 4 remove the cases that can be detected mechanically. Their
  counts in `MANIFEST.json` are a measured **lower bound** on this kind of noise in the dev
  set, not an estimate of all of it.

Noise that survives the filters is what stage 5 (human adjudication) exists to catch. If
reviewers disagree with a constructed label, the disagreement is recorded, not hidden.
