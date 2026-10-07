"""Convert the Hugging Face HotpotQA parquet (hotpotqa/hotpot_qa, distractor/validation)
into the original hotpot_dev_distractor_v1.json layout, so the stdlib builder can read it.

    uv run --no-project --with pyarrow -- python -I hf_hotpotqa_to_json.py IN.parquet OUT.json

Field mapping (HF -> original):
    id                                    -> _id
    supporting_facts{title[], sent_id[]}  -> supporting_facts [[title, sent_id], ...]
    context{title[], sentences[][]}       -> context [[title, [sentence, ...]], ...]
    question, answer, type, level         -> unchanged
Row order is preserved. Output is deterministic (sorted keys, fixed separators).
"""

import json
import sys


def convert_row(row: dict) -> dict:
    sf, ctx = row["supporting_facts"], row["context"]
    return {
        "_id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row["type"],
        "level": row["level"],
        "supporting_facts": [list(p) for p in zip(sf["title"], sf["sent_id"], strict=True)],
        "context": [[t, list(s)] for t, s in zip(ctx["title"], ctx["sentences"], strict=True)],
    }


def main() -> None:
    import pyarrow.parquet as pq  # only needed here; keeps convert_row testable without it

    src, dst = sys.argv[1], sys.argv[2]
    rows = pq.read_table(src).to_pylist()
    with open(dst, "w", encoding="utf-8") as f:
        json.dump([convert_row(r) for r in rows], f, ensure_ascii=False, sort_keys=True)
    print(f"converted {len(rows)} rows -> {dst}")


if __name__ == "__main__":
    main()
