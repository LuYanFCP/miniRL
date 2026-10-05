"""Tokenize JSONL conversations and collate masked causal-LM training batches."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from minirl.algorithm import TensorBatch


def limo_messages(
    record: object, system_prompt: str | None = None
) -> list[dict[str, str]]:
    """Keep the full LIMO solution as reasoning and the answer as a separate final response."""
    if not isinstance(record, dict):
        raise TypeError("LIMO records must be objects")
    for field in ("question", "solution", "answer"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"LIMO {field} must be a non-empty string")
    answer = record["answer"].strip()
    final = (
        answer
        if answer.startswith(r"\boxed{") and answer.endswith("}")
        else rf"\boxed{{{answer}}}"
    )
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.extend(
        [
            {"role": "user", "content": record["question"].strip()},
            {
                "role": "assistant",
                "reasoning_content": record["solution"].strip(),
                "content": final,
            },
        ]
    )
    return messages


@dataclass
class DataStats:
    rows_seen: int = 0
    examples: int = 0
    skipped_too_long: int = 0
    longest_sequence: int = 0
    supervised_tokens: int = 0
    input_tokens: int = 0
    sequence_length_mean: float = 0.0
    sequence_length_p50: float = 0.0
    sequence_length_p95: float = 0.0


def _messages(record: object) -> list[dict[str, str]]:
    if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
        raise TypeError("Each record must contain a messages list")
    messages = record["messages"]
    if len(messages) < 2:
        raise ValueError(
            "A conversation must contain a user message and an assistant reply"
        )
    expected = "user"
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError("Each message must be an object with role and content")
        role, content = message.get("role"), message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Message content must be a non-empty string")
        if index == 0 and role == "system":
            continue
        if role != expected:
            raise ValueError(
                "Messages must alternate user/assistant after an optional system message"
            )
        expected = "assistant" if role == "user" else "user"
    if messages[-1]["role"] != "assistant":
        raise ValueError("The last message must be an assistant reply")
    result = []
    for message in messages:
        cleaned = {"role": message["role"], "content": message["content"]}
        if "reasoning_content" in message:
            if not isinstance(message["reasoning_content"], str):
                raise ValueError("reasoning_content must be a string")
            cleaned["reasoning_content"] = message["reasoning_content"]
        result.append(cleaned)
    return result


class ChatSFTDataset(Dataset[TensorBatch]):
    """Pre-tokenize conversations; by default supervise only the final assistant turn."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int,
        train_on_prompt: bool = False,
        max_samples: int | None = None,
        format: Literal["limo", "messages"] = "messages",
        overlength: Literal["skip", "error"] = "skip",
        system_prompt: str | None = None,
    ) -> None:
        if max_length < 2 or (max_samples is not None and max_samples < 1):
            raise ValueError("max_length must be >= 2 and max_samples must be positive")
        if not train_on_prompt and not tokenizer.is_fast:
            raise ValueError(
                "Assistant-only masking requires a fast tokenizer with token offsets"
            )
        if format not in ("limo", "messages") or overlength not in ("skip", "error"):
            raise ValueError("Unsupported data format or overlength policy")
        self.samples: list[TensorBatch] = []
        self.records: list[dict[str, Any]] = []
        self.prompts: list[list[dict[str, str]]] = []
        self.prompt_keys: list[str] = []
        self.stats = DataStats()
        fingerprint = hashlib.sha256()
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                if max_samples is not None and self.stats.rows_seen >= max_samples:
                    break
                self.stats.rows_seen += 1
                try:
                    record = json.loads(line)
                    messages = (
                        limo_messages(record, system_prompt)
                        if format == "limo"
                        else _messages(record)
                    )
                    full_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    if format == "limo" and record["solution"].strip() not in full_text:
                        raise ValueError(
                            "The chat template discarded LIMO reasoning_content"
                        )
                    prefix_length = 0
                    if not train_on_prompt:
                        prefix = tokenizer.apply_chat_template(
                            messages[:-1], tokenize=False, add_generation_prompt=True
                        )
                        if not full_text.startswith(prefix):
                            raise ValueError(
                                "The chat template's generation prefix differs from the training text; "
                                "use a compatible template or train_on_prompt=true"
                            )
                        prefix_length = len(prefix)
                    encoded = tokenizer(
                        full_text,
                        add_special_tokens=False,
                        truncation=False,
                        return_offsets_mapping=not train_on_prompt,
                    )
                    length = len(encoded["input_ids"])
                    self.stats.longest_sequence = max(
                        self.stats.longest_sequence, length
                    )
                    if length > max_length:
                        if overlength == "error":
                            raise ValueError(
                                f"Sequence length {length} exceeds max_length={max_length}"
                            )
                        self.stats.skipped_too_long += 1
                        continue
                    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
                    labels = input_ids.clone()
                    if not train_on_prompt:
                        # Mask boundary-straddling tokens too; prompt tokens must not leak.
                        for index, (start, end) in enumerate(encoded["offset_mapping"]):
                            if start < prefix_length or end <= start:
                                labels[index] = -100
                    if not (labels[1:] != -100).any():
                        raise ValueError("No supervised tokens remain in this example")
                    self.samples.append(
                        {
                            "input_ids": input_ids,
                            "attention_mask": torch.ones_like(input_ids),
                            "labels": labels,
                        }
                    )
                    self.stats.examples += 1
                    self.records.append(record)
                    self.prompts.append(messages[:-1])
                    # Group duplicate prompts even when their reference answers differ.
                    normalized = [
                        {key: " ".join(value.split()) for key, value in message.items()}
                        for message in messages[:-1]
                    ]
                    self.prompt_keys.append(
                        hashlib.sha256(
                            json.dumps(normalized, sort_keys=True).encode()
                        ).hexdigest()
                    )
                    self.stats.supervised_tokens += int((labels[1:] != -100).sum())
                    fingerprint.update(len(input_ids).to_bytes(8, "little"))
                    fingerprint.update(input_ids.numpy().tobytes())
                    fingerprint.update(labels.numpy().tobytes())
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
        if not self.samples:
            raise ValueError(
                f"{path}: no training examples retained; {self.stats.skipped_too_long} exceeded max_length={max_length}"
            )
        self.fingerprint = fingerprint.hexdigest()
        self._length_stats()

    def _length_stats(self) -> None:
        lengths = [len(sample["input_ids"]) for sample in self.samples]
        self.stats.input_tokens = sum(lengths)
        self.stats.sequence_length_mean = float(np.mean(lengths))
        self.stats.sequence_length_p50 = float(np.quantile(lengths, 0.5))
        self.stats.sequence_length_p95 = float(np.quantile(lengths, 0.95))

    def select(self, indices: list[int]) -> ChatSFTDataset:
        """Build a deterministic view without tokenizing again or sharing mutable stats."""
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Dataset selection requires non-empty, unique indices")
        selected = copy.copy(self)
        selected.samples = [self.samples[i] for i in indices]
        selected.records = [self.records[i] for i in indices]
        selected.prompts = [self.prompts[i] for i in indices]
        selected.prompt_keys = [self.prompt_keys[i] for i in indices]
        selected.stats = DataStats(
            rows_seen=len(indices),
            examples=len(indices),
            longest_sequence=max(
                len(sample["input_ids"]) for sample in selected.samples
            ),
            supervised_tokens=sum(
                int((sample["labels"][1:] != -100).sum()) for sample in selected.samples
            ),
        )
        selected._length_stats()
        digest = hashlib.sha256()
        for sample in selected.samples:
            digest.update(len(sample["input_ids"]).to_bytes(8, "little"))
            digest.update(sample["input_ids"].numpy().tobytes())
            digest.update(sample["labels"].numpy().tobytes())
        selected.fingerprint = digest.hexdigest()
        return selected

    def split(
        self, validation_size: int, seed: int
    ) -> tuple[ChatSFTDataset, ChatSFTDataset]:
        """Hold out entire prompt groups, so repeated questions cannot leak into training."""
        if not 0 < validation_size < len(self):
            raise ValueError(
                "holdout_size must be positive and smaller than the retained dataset"
            )
        groups: dict[str, list[int]] = {}
        for index, key in enumerate(self.prompt_keys):
            groups.setdefault(key, []).append(index)
        keys = sorted(groups)
        random.Random(seed).shuffle(keys)
        validation: list[int] = []
        for key in keys:
            if len(validation) >= validation_size:
                break
            validation.extend(groups[key])
        heldout = set(validation)
        training = [i for i in range(len(self)) if i not in heldout]
        if not training:
            raise ValueError(
                "Prompt groups leave no training examples; reduce holdout_size or deduplicate data"
            )
        return self.select(training), self.select(validation)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TensorBatch:
        return self.samples[index]


class SFTCollator:
    """Right-pad batches without masking real EOS tokens when EOS doubles as padding."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[TensorBatch]) -> TensorBatch:
        return {
            name: pad_sequence(
                [sample[name] for sample in samples],
                batch_first=True,
                padding_value=padding,
            )
            for name, padding in (
                ("input_ids", self.pad_token_id),
                ("attention_mask", 0),
                ("labels", -100),
            )
        }
