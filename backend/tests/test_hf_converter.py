"""The Hugging Face -> original-layout converter (scripts/hf_hotpotqa_to_json.py)."""

import importlib.util
from pathlib import Path

from judge_check.datasets import hotpotqa as hp

_path = Path(__file__).resolve().parents[1] / "scripts" / "hf_hotpotqa_to_json.py"
_spec = importlib.util.spec_from_file_location("hf_hotpotqa_to_json", _path)
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)

HF_ROW = {
    "id": "5a8b57f25542995d1e6f1371",
    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
    "answer": "yes",
    "type": "comparison",
    "level": "hard",
    "supporting_facts": {"title": ["Scott Derrickson", "Ed Wood"], "sent_id": [0, 0]},
    "context": {
        "title": ["Ed Wood", "Scott Derrickson"],
        "sentences": [
            ["Edward Davis Wood Jr. was an American filmmaker.", " He was born in 1924."],
            ["Scott Derrickson is an American director."],
        ],
    },
}


def test_convert_row_maps_every_field() -> None:
    assert conv.convert_row(HF_ROW) == {
        "_id": "5a8b57f25542995d1e6f1371",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "yes",
        "type": "comparison",
        "level": "hard",
        "supporting_facts": [["Scott Derrickson", 0], ["Ed Wood", 0]],
        "context": [
            [
                "Ed Wood",
                ["Edward Davis Wood Jr. was an American filmmaker.", " He was born in 1924."],
            ],
            ["Scott Derrickson", ["Scott Derrickson is an American director."]],
        ],
    }


def test_converted_row_is_accepted_by_the_builder() -> None:
    row = conv.convert_row(HF_ROW)
    # Sentence text and order survive unchanged (leading space included).
    assert row["context"][0][1][1] == " He was born in 1924."
    assert hp.check_example(row)[0] == hp.YES_NO  # parsed fine, then filtered as yes/no
