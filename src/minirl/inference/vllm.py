"""Token-preserving vLLM HTTP inference and synchronous NCCL weight updates."""

from __future__ import annotations

import math

import httpx


class VLLMClient:
    def __init__(self, base_url: str, model: str = "minirl-gsm8k") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session = httpx.Client()
        self.weight_engine = None

    def generate(
        self,
        prompts: list[list[int]],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        seed: int,
        logprobs: bool = False,
    ) -> list[dict]:
        payload = {
            "model": self.model,
            "prompt": prompts,
            "n": n,
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "max_tokens": max_tokens,
            "seed": seed,
            "return_token_ids": True,
            "skip_special_tokens": False,
        }
        if logprobs:
            payload["logprobs"] = 0
        response = self.session.post(
            self.base_url + "/v1/completions", json=payload, timeout=3600
        )
        response.raise_for_status()
        choices = sorted(response.json()["choices"], key=lambda c: c["index"])
        if [c["index"] for c in choices] != list(range(len(prompts) * n)):
            raise ValueError("vLLM returned incomplete or misindexed completions")
        for choice in choices:
            ids = choice["token_ids"]
            if not ids or choice["finish_reason"] not in ("stop", "length"):
                raise ValueError("vLLM returned an empty or failed completion")
            if logprobs:
                values = choice["logprobs"]["token_logprobs"]
                if len(values) != len(ids) or not all(
                    v is not None and math.isfinite(v) for v in values
                ):
                    raise ValueError(
                        "Sampled token IDs and behavior logprobs must align"
                    )
        return choices

    def connect_weights(self, model, *, port: int) -> None:
        from vllm.distributed.weight_transfer import (
            HTTPVLLMWeightSyncClient,
            ModuleSource,
            WeightTransferTrainerFactory,
        )
        from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo

        response = self.session.get(self.base_url + "/get_world_size", timeout=30)
        response.raise_for_status()
        self.weight_engine = WeightTransferTrainerFactory.trainer_init(
            init_info=NCCLTrainerInitInfo(
                master_address="127.0.0.1",
                master_port=port,
                world_size=response.json()["world_size"] + 1,
                rank=0,
                packed=True,
            ),
            client=HTTPVLLMWeightSyncClient(self.base_url),
            source=ModuleSource(model),
        )

    def sync_weights(self) -> None:
        if self.weight_engine is None:
            raise RuntimeError("Weight-transfer engine has not been initialized")
        self.session.post(self.base_url + "/pause", timeout=60).raise_for_status()
        # On a failed transfer leave serving paused; partial weights are unusable.
        self.weight_engine.send_weights()
        self.session.post(
            self.base_url + "/reset_prefix_cache", timeout=60
        ).raise_for_status()
        self.session.post(self.base_url + "/resume", timeout=60).raise_for_status()
