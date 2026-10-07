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

### Changes made to the original

- Selected 200 of the 7,405 dev questions using the deterministic filters and seeded order
  described below.
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
| `MANIFEST.json` | yes | source URL and SHA-256, seed, builder checksum, filter counts |
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

The eligible questions are ordered by `sha256("judge-check-hotpotqa-v1:" + question_id)` and
taken from the top. The order of the source file doesn't matter, and rebuilding gives
identical bytes.

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
