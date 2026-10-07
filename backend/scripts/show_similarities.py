"""Print raw embedding numbers and cosine similarities for a handful of demo documents.

    cd backend && uv run python scripts/show_similarities.py

No database needed. Read the output next to docs/learn/01-retrieval.md.
"""

import numpy as np

from judge_check.datasets.demo import load_demo
from judge_check.embeddings import DEFAULT_MODEL, get_embedder

N_QUESTIONS = 3
np.set_printoptions(precision=4, suppress=True, linewidth=120)


def short(s: str, n: int = 60) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> None:
    documents, questions = load_demo()
    docs = {d.id: d for d in documents}
    emb = get_embedder(DEFAULT_MODEL)

    # The gold paragraphs of the first few questions: each question has exactly two.
    qs = questions[:N_QUESTIONS]
    titles: list[str] = []
    for q in qs:
        for ev in q.gold_evidence:
            if ev.document_id not in titles:
                titles.append(ev.document_id)
    passages = [docs[t].text for t in titles]
    P = emb.embed_passages(passages)

    print(f"model: {emb.spec.name}   dimension: {emb.spec.dimension}\n")
    print("1) An embedding is just a vector of floats. First passage:")
    print(f"   text:      {short(passages[0], 90)}")
    print(f"   first 8:   {P[0][:8]}")
    print(f"   min / max: {P[0].min():.4f} / {P[0].max():.4f}")
    print(f"   L2 norm:   {np.linalg.norm(P[0]):.6f}   (we normalise every vector to length 1)\n")

    print("2) Passage x passage cosine similarity (unit vectors, so this is just P @ P.T):")
    S = P @ P.T
    for i, t in enumerate(titles):
        print(f"   {i}  {short(t, 34):<34} " + " ".join(f"{x:6.3f}" for x in S[i]))
    print("   Diagonal is 1.0. Note how high unrelated pairs still are: these vectors share a")
    print("   common direction, so absolute values mean little; only the *ranking* matters.\n")

    print("3) Query x passage similarity, with and without the BGE query instruction.")
    print("   * marks the two gold paragraphs for that question.")
    for label, use in [("WITH instruction", True), ("WITHOUT instruction", False)]:
        Q = emb.embed_queries([q.question for q in qs], use_instruction=use)
        print(f"\n   {label}")
        for qi, q in enumerate(qs):
            gold = {ev.document_id for ev in q.gold_evidence}
            sims = Q[qi] @ P.T
            cells = " ".join(
                f"{s:6.3f}{'*' if titles[j] in gold else ' '}" for j, s in enumerate(sims)
            )
            print(f"   q{qi} {short(q.question, 50):<50} {cells}")
    print()

    print("4) Why normalise? Cosine ignores length; the raw dot product does not.")
    a, b = P[0], P[1]
    for scale in (1.0, 3.0):
        dot = float((scale * a) @ b)
        cos = dot / (np.linalg.norm(scale * a) * np.linalg.norm(b))
        print(f"   a scaled x{scale:.0f}:  dot = {dot:.4f}   cosine = {cos:.4f}")
    print("   With unit vectors dot == cosine, so pgvector can use the cheaper inner product.")


if __name__ == "__main__":
    main()
