"""Pinned GSM8K train/test materialization and answer-free policy prompts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

DATASET_ID = "openai/gsm8k"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
INSTRUCTION = (
    "Solve the following math problem. Write your reasoning between <think> and "
    "</think>. Then give only the final numerical answer on a separate last line "
    "in the form #### answer.\n\n"
)


def read_jsonl(path: str | Path) -> list[dict]:
    # str.splitlines() also splits Unicode U+0085/U+2028 inside valid JSON strings.
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prepare_gsm8k(directory: str | Path) -> dict:
    from huggingface_hub import hf_hub_download
    from pyarrow import parquet

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": DATASET_ID, "revision": DATASET_REVISION, "splits": {}}
    question_sets = {}
    for split, expected in (("train", 7473), ("test", 1319)):
        filename = f"main/{split}-00000-of-00001.parquet"
        path = Path(
            hf_hub_download(
                DATASET_ID, filename, repo_type="dataset", revision=DATASET_REVISION
            )
        )
        rows = parquet.read_table(path).to_pylist()
        if len(rows) != expected:
            raise ValueError(f"Unexpected {split} size: {len(rows)}")
        records = []
        for index, row in enumerate(rows):
            solution = row["answer"]
            if solution.count("####") != 1:
                raise ValueError(f"Ambiguous reference answer: {split}/{index}")
            answer = solution.split("####", 1)[1].strip()
            if not re.fullmatch(r"[-+]?[\d,]+(?:\.\d+)?", answer):
                raise ValueError(f"Non-numeric GSM8K answer: {answer}")
            records.append(
                {
                    "id": f"{split}/{index}",
                    "question": row["question"],
                    "answer": answer,
                }
            )
        output = directory / f"{split}.jsonl"
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
        )
        question_sets[split] = {
            " ".join(row["question"].casefold().split()) for row in records
        }
        manifest["splits"][split] = {
            "rows": len(records),
            "source": filename,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "jsonl_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    overlap = question_sets["train"] & question_sets["test"]
    manifest["normalized_question_overlap"] = len(overlap)
    if overlap:
        raise ValueError("GSM8K train/test questions overlap")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prompt_tokens(tokenizer, question: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": INSTRUCTION + question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    # Both tags are generated actions and earn the format reward themselves.
    if text.endswith("<think>\n"):
        text = text[: -len("<think>\n")]
    elif text.endswith("<think>"):
        text = text[: -len("<think>")]
    return tokenizer.encode(text, add_special_tokens=False)
