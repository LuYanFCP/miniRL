"""Complete GSM8K test evaluation through the rollout inference engine."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from minirl.data.gsm8k import prompt_tokens, read_jsonl
from minirl.inference.vllm import VLLMClient
from minirl.rewards.gsm8k import score_gsm8k


def evaluate_gsm8k(
    client,
    tokenizer,
    rows: list[dict],
    output_dir: str | Path,
    *,
    step: int,
    max_tokens: int = 4096,
    batch_size: int = 128,
    seed: int = 44,
) -> dict:
    if len(rows) != 1319 or any(not row["id"].startswith("test/") for row in rows):
        raise ValueError("Full GSM8K evaluation requires all 1319 test questions")
    if len({row["id"] for row in rows}) != 1319:
        raise ValueError("Evaluation IDs must be unique")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / f"step_{step:08d}.jsonl"
    summary_path = sample_path.with_suffix(".json")
    sums = {
        key: 0.0
        for key in (
            "correct",
            "think_format",
            "answer_format",
            "answer_parsed",
            "reward",
            "correctness_reward",
            "think_reward",
            "answer_reward",
            "generated_tokens",
            "truncated",
            "eos",
            "grader_errors",
        )
    }
    started = time.monotonic()
    count = 0
    with sample_path.open("w") as stream:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            choices = client.generate(
                [prompt_tokens(tokenizer, row["question"]) for row in chunk],
                n=1,
                max_tokens=max_tokens,
                temperature=0.0,
                seed=seed,
            )
            for row, choice in zip(chunk, choices, strict=True):
                scores = score_gsm8k(choice["text"], row["answer"])
                scores.update(
                    {
                        "generated_tokens": len(choice["token_ids"]),
                        "truncated": int(choice["finish_reason"] == "length"),
                        "eos": int(choice["token_ids"][-1] == tokenizer.eos_token_id),
                        "grader_errors": int(scores["grader_error"] is not None),
                    }
                )
                record = {
                    **row,
                    "step": step,
                    "text": choice["text"],
                    "token_ids": choice["token_ids"],
                    "scores": scores,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                for key in sums:
                    sums[key] += scores[key]
                count += 1
            stream.flush()
            print(
                f"GSM8K eval step={step} {count}/{len(rows)} correct={sums['correct']:.0f}",
                flush=True,
            )
    seconds = time.monotonic() - started
    metrics = {f"eval/{key}": value / count for key, value in sums.items()}
    metrics.update(
        {
            "eval/examples": count,
            "eval/seconds": seconds,
            "eval/tokens_per_second": sums["generated_tokens"] / seconds,
        }
    )
    summary = {
        "step": step,
        "split": "test",
        "rows": count,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": seed,
        "metrics": metrics,
        "counts": sums,
        "completed_at": time.time(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if sums["grader_errors"]:
        raise RuntimeError("GSM8K evaluation encountered verifier errors")
    return summary


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18700")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = read_jsonl(args.test_data)
    result = evaluate_gsm8k(
        VLLMClient(args.url),
        tokenizer,
        rows,
        args.output_dir,
        step=args.step,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
