"""Load a Transformers text policy, including Qwen3.5's text-only checkpoint view."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass
class ModelConfig:
    name_or_path: str
    revision: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    gradient_checkpointing: bool = True
    attn_implementation: Literal["sdpa", "eager"] = "sdpa"
    dtype: Literal["float32", "bfloat16"] = "float32"
    eos_token: str | None = "<|im_end|>"
    loss_chunk_size: int | None = None

    def __post_init__(self) -> None:
        if not self.name_or_path.strip():
            raise ValueError("model.name_or_path must not be empty")
        if self.loss_chunk_size is not None and (
            type(self.loss_chunk_size) is not int or self.loss_chunk_size < 1
        ):
            raise ValueError("model.loss_chunk_size must be positive or null")


def load_tokenizer(config: ModelConfig) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        config.name_or_path,
        revision=config.revision,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
        trust_remote_code=False,
        use_fast=True,
    )
    if not tokenizer.chat_template:
        raise ValueError("The SFT recipe requires a tokenizer with a chat template")
    if config.eos_token is not None:
        if config.eos_token not in tokenizer.get_vocab():
            raise ValueError(
                f"EOS token {config.eos_token!r} is not in the existing vocabulary"
            )
        tokenizer.eos_token = config.eos_token
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer needs an EOS or padding token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_policy(
    config: ModelConfig, tokenizer: PreTrainedTokenizerBase
) -> PreTrainedModel:
    # Transformers 5.18 extracts Qwen3.5 text_config and maps language_model weights.
    model, info = AutoModelForCausalLM.from_pretrained(
        config.name_or_path,
        revision=config.revision,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
        trust_remote_code=False,
        dtype=getattr(torch, config.dtype),
        attn_implementation=config.attn_implementation,
        output_loading_info=True,
    )
    if (
        info.get("missing_keys")
        or info.get("mismatched_keys")
        or info.get("error_msgs")
    ):
        raise ValueError(
            f"Policy checkpoint did not fully initialize the text model: {info}"
        )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if config.loss_chunk_size is not None:
        from .chunked_loss import enable_chunked_loss

        enable_chunked_loss(model, config.loss_chunk_size)
    return model
