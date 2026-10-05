"""Full-parameter GSM8K GRPO with DDP training and a dedicated vLLM engine."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from minirl.algorithm import GRPO, GRPOConfig
from minirl.common import (
    CSVSink,
    DataCollector,
    JSONLSink,
    LoggingSink,
    TensorBoardSink,
)
from minirl.common.mfu import MFU6NEstimator
from minirl.config import RLConfig
from minirl.data.gsm8k import DATASET_REVISION, prompt_tokens, read_jsonl
from minirl.inference.vllm import VLLMClient
from minirl.models.causal_lm import ModelConfig, load_policy, load_tokenizer
from minirl.models.policy_logprobs import QwenPolicyLogprobs
from minirl.recipes.gsm8k_eval import evaluate_gsm8k
from minirl.rewards.gsm8k import score_gsm8k
from minirl.trainer import RLTrainer


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def share(value, rank: int):
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def archive_uncommitted_metrics(root: Path) -> int:
    """Preserve failed-attempt records separately before resuming a checkpoint."""
    state = torch.load(
        root / "trainer_state.pt", map_location="cpu", weights_only=True, mmap=True
    )
    step, iteration = state["global_step"], state["loop"]["iteration"]
    archive = root / "interrupted_attempts" / str(time.time_ns())
    archive.mkdir(parents=True)
    for name in ("metrics.jsonl", "metrics.csv"):
        path = root / name
        if not path.exists():
            continue
        shutil.copyfile(path, archive / name)
        if path.suffix == ".jsonl":
            valid = []
            for line in path.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    break
                if record["step"] <= step:
                    valid.append(line)
            path.write_text("\n".join(valid) + "\n")
        else:
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = [
                    r
                    for r in reader
                    if r.get("step", "").isdigit() and int(r["step"]) <= step
                ]
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    for directory, boundary in (("rollouts", iteration - 1), ("eval", step)):
        for path in (root / directory).glob("*"):
            if (
                path.stem.rsplit("_", 1)[-1].isdigit()
                and int(path.stem.rsplit("_", 1)[-1]) > boundary
            ):
                target = archive / directory / path.name
                target.parent.mkdir(exist_ok=True)
                path.rename(target)
    return step


class GSM8KRollout:
    def __init__(self, rows, tokenizer, client, args, rank):
        self.rows, self.tokenizer, self.client = rows, tokenizer, client
        self.args, self.rank = args, rank
        self.order = list(range(len(rows)))
        random.Random(args.seed).shuffle(self.order)
        self.cursor = 0
        self.iteration = 0
        self.policy_step = 0
        self.last_metrics = {}

    def state_dict(self):
        return {
            "cursor": self.cursor,
            "iteration": self.iteration,
            "policy_step": self.policy_step,
            "order": self.order,
        }

    def load_state_dict(self, state):
        if state["order"] != self.order or not 0 <= state["cursor"] <= len(self.rows):
            raise ValueError("Checkpoint GSM8K order/cursor differs")
        self.cursor, self.iteration = state["cursor"], state["iteration"]
        self.policy_step = state["policy_step"]

    def __call__(self, _):
        payload = None
        if self.rank == 0:
            started = time.monotonic()
            rows = [
                self.rows[i]
                for i in self.order[
                    self.cursor : self.cursor + self.args.prompts_per_rollout
                ]
            ]
            if not rows:
                raise RuntimeError("No remaining training questions")
            prompts = [prompt_tokens(self.tokenizer, row["question"]) for row in rows]
            choices = self.client.generate(
                prompts,
                n=self.args.group_size,
                max_tokens=self.args.max_new_tokens,
                temperature=1.0,
                seed=self.args.seed + self.iteration,
                logprobs=True,
            )
            records = []
            for index, choice in enumerate(choices):
                group = index // self.args.group_size
                row = rows[group]
                scores = score_gsm8k(choice["text"], row["answer"])
                if scores["grader_error"] is not None:
                    raise RuntimeError(
                        f"Verifier failed on {row['id']}: {scores['grader_error']}"
                    )
                records.append(
                    {
                        **row,
                        "group": group,
                        "sample": index % self.args.group_size,
                        "policy_step": self.policy_step,
                        "iteration": self.iteration,
                        "prompt_ids": prompts[group],
                        "response_ids": choice["token_ids"],
                        "old_logprobs": choice["logprobs"]["token_logprobs"],
                        "text": choice["text"],
                        "finish_reason": choice["finish_reason"],
                        "scores": scores,
                    }
                )
            width = max(len(r["prompt_ids"]) + len(r["response_ids"]) for r in records)
            ids = torch.full(
                (len(records), width), self.tokenizer.pad_token_id, dtype=torch.long
            )
            attention = torch.zeros_like(ids, dtype=torch.bool)
            response = torch.zeros_like(attention)
            old = torch.zeros_like(ids, dtype=torch.float32)
            for index, record in enumerate(records):
                prefix, completion = record["prompt_ids"], record["response_ids"]
                end = len(prefix) + len(completion)
                ids[index, :end] = torch.tensor(prefix + completion)
                attention[index, :end] = True
                response[index, len(prefix) : end] = True
                old[index, len(prefix) : end] = torch.tensor(record["old_logprobs"])
            batch = {
                "input_ids": ids,
                "attention_mask": attention,
                "response_mask": response,
                "old_logprobs": old,
                "rewards": torch.tensor([r["scores"]["reward"] for r in records]),
                "group_ids": torch.tensor([r["group"] for r in records]),
            }
            path = (
                Path(self.args.output_dir)
                / "rollouts"
                / f"iteration_{self.iteration:06d}.jsonl"
            )
            path.parent.mkdir(exist_ok=True)
            path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
            )
            count = len(records)
            metrics = {
                key: sum(r["scores"][key] for r in records) / count
                for key in (
                    "correct",
                    "think_format",
                    "answer_format",
                    "answer_parsed",
                    "correctness_reward",
                    "think_reward",
                    "answer_reward",
                )
            }
            seconds = time.monotonic() - started
            metrics.update(
                {
                    "seconds": seconds,
                    "samples": count,
                    "prompts": len(rows),
                    "truncated": sum(r["finish_reason"] == "length" for r in records)
                    / count,
                    "eos": sum(
                        r["response_ids"][-1] == self.tokenizer.eos_token_id
                        for r in records
                    )
                    / count,
                    "tokens_per_second": sum(len(r["response_ids"]) for r in records)
                    / seconds,
                }
            )
            payload = (batch, metrics, len(rows))
        batch, self.last_metrics, consumed = share(payload, self.rank)
        self.cursor += consumed
        self.iteration += 1
        return batch


class GSM8KGRPO(GRPO):
    def __init__(self, config, rollout):
        super().__init__(config)
        self.rollout = rollout

    def rollout_metrics(self):
        return {**super().rollout_metrics(), **self.rollout.last_metrics}


class GSM8KTrainer(RLTrainer):
    def __init__(
        self, config, algorithm, actor, rollout, collector, args, tokenizer, test_rows
    ):
        super().__init__(
            config, algorithm, {"actor": actor}, rollout, collector=collector
        )
        self.args, self.tokenizer, self.test_rows = args, tokenizer, test_rows
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.policy = actor.module.policy
        self.root = Path(args.output_dir)
        self.last_eval = 0
        self.best_accuracy = -1.0
        self._checkpoint_rngs = None
        self._perf = {}
        self.mfu = MFU6NEstimator(
            sum(p.numel() for p in self.policy.parameters()), 835.5, self.world
        )

    def _minibatches(self, batch, size):
        count = batch["input_ids"].shape[0]
        if count % self.world or size % self.world:
            raise ValueError(
                "Rollout and minibatch sizes must divide evenly across DDP ranks"
            )
        order = (
            torch.randperm(count, device=self.device)
            if self.rank == 0
            else torch.empty(count, device=self.device, dtype=torch.long)
        )
        dist.broadcast(order, src=0)
        for start in range(0, count, size):
            selection = order[start : start + size].chunk(self.world)[self.rank]
            yield {key: value[selection] for key, value in batch.items()}

    def _update_minibatch(self, batch):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.monotonic()
        losses = super()._update_minibatch(batch)
        torch.cuda.synchronize(self.device)
        seconds = torch.tensor(time.monotonic() - started, device=self.device)
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        tokens = batch["attention_mask"].sum()
        dist.all_reduce(tokens)
        peak = torch.tensor(
            torch.cuda.max_memory_allocated(self.device), device=self.device
        )
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        for loss in losses.values():
            dist.all_reduce(loss)
            loss.div_(self.world)
        self._perf = {
            "train/step_seconds": float(seconds),
            "train/tokens": int(tokens),
            "train/tokens_per_second": float(tokens / seconds),
            "train/peak_gpu_memory_gb": float(peak) / 1e9,
            **self.mfu.estimate(tokens=int(tokens), seconds=float(seconds)),
        }
        return losses

    def log(self, metrics):
        if self.rank == 0:
            super().log(
                {
                    **metrics,
                    **self._perf,
                    "train/questions_seen": self.rollout_fn.cursor,
                }
            )
            self.status("training")

    def status(self, phase, **extra):
        if self.rank == 0:
            atomic_json(
                self.root / "training_status.json",
                {
                    "phase": phase,
                    "step": self.global_step,
                    "iteration": self.iteration,
                    "questions_seen": self.rollout_fn.cursor,
                    "questions_total": 7473,
                    "iterations_total": self.config.num_iterations,
                    "last_eval_step": self.last_eval,
                    "pid": os.getpid(),
                    "updated_at": time.time(),
                    **extra,
                },
            )

    def _after_iteration(self):
        if self.rank == 0:
            self.status("syncing_weights")
            self.rollout_fn.client.sync_weights()
        dist.barrier()
        self.rollout_fn.policy_step = self.global_step
        final = self.iteration == self.config.num_iterations
        if self.global_step >= self.last_eval + self.args.eval_every_steps or final:
            self.evaluate()
        if self.iteration == 1:
            self.save_checkpoint()

    def evaluate(self):
        result = None
        if self.rank == 0:
            self.status("evaluating")
            result = evaluate_gsm8k(
                self.rollout_fn.client,
                self.tokenizer,
                self.test_rows,
                self.root / "eval",
                step=self.global_step,
                max_tokens=self.args.max_new_tokens,
                seed=self.args.seed,
            )
            self.collector.record(result["metrics"], step=self.global_step)
            if result["metrics"]["eval/correct"] > self.best_accuracy:
                self.policy.save_pretrained(self.root / "best", safe_serialization=True)
                self.tokenizer.save_pretrained(self.root / "best")
                atomic_json(self.root / "best" / "evaluation.json", result)
        result = share(result, self.rank)
        self.best_accuracy = max(self.best_accuracy, result["metrics"]["eval/correct"])
        self.last_eval = self.global_step

    def _loop_state_dict(self):
        return {
            **super()._loop_state_dict(),
            "world_size": self.world,
            "last_eval": self.last_eval,
            "best_accuracy": self.best_accuracy,
            "recipe": vars(self.args) | {"resume": False},
        }

    def _load_loop_state_dict(self, state):
        if state["world_size"] != self.world or state["recipe"] != vars(self.args) | {
            "resume": False
        }:
            raise ValueError("Checkpoint distributed GSM8K recipe differs")
        super()._load_loop_state_dict(state)
        self.last_eval, self.best_accuracy = state["last_eval"], state["best_accuracy"]

    def save_checkpoint(self):
        local = super()._rng_state()
        states = [None] * self.world
        dist.all_gather_object(states, local)
        self._checkpoint_rngs = states
        if self.rank == 0:
            self.status("checkpointing")
            super().save_checkpoint()
        dist.barrier()
        self._checkpoint_rngs = None

    def _rng_state(self):
        if self._checkpoint_rngs is None:
            raise RuntimeError("Distributed RNG states have not been gathered")
        return {"ranks": self._checkpoint_rngs}

    def _restore_rng_state(self, state):
        super()._restore_rng_state(state["ranks"][self.rank])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18700")
    parser.add_argument("--weight-port", type=int, default=18702)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--prompts-per-rollout", type=int, default=16)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--eval-every-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.group_size,
            args.prompts_per_rollout,
            args.minibatch_size,
            args.microbatch_size,
            args.max_new_tokens,
            args.eval_every_steps,
        )
        < 1
    ):
        raise ValueError("GSM8K loop and generation sizes must be positive")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(hours=2))
    root = Path(args.output_dir)
    resume_step = share(
        archive_uncommitted_metrics(root) if rank == 0 and args.resume else None, rank
    )
    manifest = json.loads((Path(args.data_dir) / "manifest.json").read_text())
    if (
        manifest["revision"] != DATASET_REVISION
        or manifest["normalized_question_overlap"] != 0
    ):
        raise ValueError("Unexpected GSM8K data manifest")
    datasets = {}
    for split, count in (("train", 7473), ("test", 1319)):
        data_path = Path(args.data_dir) / f"{split}.jsonl"
        if (
            hashlib.sha256(data_path.read_bytes()).hexdigest()
            != manifest["splits"][split]["jsonl_sha256"]
        ):
            raise ValueError("GSM8K materialized data changed")
        datasets[split] = read_jsonl(data_path)
        if len(datasets[split]) != count:
            raise ValueError("GSM8K split is incomplete")
    baseline = json.loads((root / "eval/step_00000000.json").read_text())
    if baseline["rows"] != 1319 or baseline["max_tokens"] != args.max_new_tokens:
        raise ValueError(
            "Complete, matching step-0 evaluation is required before training"
        )
    config = ModelConfig(
        name_or_path=args.model,
        local_files_only=True,
        dtype="bfloat16",
        gradient_checkpointing=True,
    )
    tokenizer = load_tokenizer(config)
    policy = load_policy(config, tokenizer).to(local_rank)
    actor = DistributedDataParallel(
        QwenPolicyLogprobs(policy).to(local_rank),
        device_ids=[local_rank],
        gradient_as_bucket_view=True,
    )
    client = VLLMClient(args.url) if rank == 0 else None
    rollout = GSM8KRollout(datasets["train"], tokenizer, client, args, rank)
    algorithm = GSM8KGRPO(GRPOConfig(group_size=args.group_size, kl_coef=0), rollout)
    train_config = RLConfig(
        output_dir=str(root),
        device=f"cuda:{local_rank}",
        bf16=True,
        num_iterations=math.ceil(len(datasets["train"]) / args.prompts_per_rollout),
        update_epochs=1,
        minibatch_size=args.minibatch_size,
        microbatch_size=args.microbatch_size,
        learning_rate=args.learning_rate,
        weight_decay=0,
        max_grad_norm=1,
        log_steps=1,
        save_steps=25,
        seed=args.seed,
    )
    if rank == 0:
        atomic_json(
            root / "recipe_config.json",
            {
                "args": vars(args),
                "trainer": asdict(train_config),
                "algorithm": asdict(algorithm.config),
                "dataset": manifest,
                "world_size": dist.get_world_size(),
                "reward": {"correct": 1.0, "think": 0.3, "answer": 0.2},
            },
        )
    sinks = (
        [
            LoggingSink(root / "train.log"),
            JSONLSink(root / "metrics.jsonl"),
            CSVSink(root / "metrics.csv"),
            TensorBoardSink(
                root / "tensorboard",
                purge_step=resume_step + 1 if args.resume else None,
            ),
        ]
        if rank == 0
        else []
    )
    with DataCollector(sinks, strict=True) as collector:
        trainer = GSM8KTrainer(
            train_config,
            algorithm,
            actor,
            rollout,
            collector,
            args,
            tokenizer,
            datasets["test"],
        )
        if args.resume:
            trainer.load_checkpoint()
        else:
            if trainer._ckpt_path.exists():
                raise ValueError("Run already has a checkpoint; use --resume")
            trainer.best_accuracy = baseline["metrics"]["eval/correct"]
            if rank == 0:
                collector.record(baseline["metrics"], step=0)
            trainer.save_checkpoint()
        if rank == 0:
            client.connect_weights(policy, port=args.weight_port)
            client.sync_weights()
        dist.barrier()
        try:
            trainer.train()
            if rollout.cursor != len(datasets["train"]):
                raise RuntimeError("Training did not visit every GSM8K train question")
            if rank == 0:
                trainer.status("exporting")
                if not all(torch.isfinite(p).all() for p in policy.parameters()):
                    raise RuntimeError("Final policy contains non-finite parameters")
                policy.save_pretrained(root / "final", safe_serialization=True)
                tokenizer.save_pretrained(root / "final")
                trainer.status("complete", finished_at=time.time(), exit_code=0)
        except BaseException as exc:
            trainer.status("failed", error=f"{type(exc).__name__}: {exc}")
            raise
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
