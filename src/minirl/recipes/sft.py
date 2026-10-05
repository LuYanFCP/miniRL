"""Build, validate, train, and export a YAML-configured SFT policy for RL initialization."""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerBase, set_seed

from minirl.common import (
    DataCollector,
    LoggingSink,
    MetricSink,
    OpenTelemetrySink,
    TensorBoardSink,
    dump_yaml_config,
    load_yaml_config,
)
from minirl.common.plugins import (
    PluginConfig,
    load_plugin,
    plugin_fingerprint,
    resolve_entrypoint,
)
from minirl.config import SFTConfig
from minirl.data.sft import ChatSFTDataset, SFTCollator
from minirl.models.causal_lm import ModelConfig, load_policy, load_tokenizer
from minirl.recipes.evaluation import SFTEvaluator
from minirl.trainer import SFTTrainer

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    repo_id: str | None = None
    filename: str = "limo-v2.jsonl"
    revision: str | None = None
    local_path: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    format: Literal["limo", "messages"] = "limo"
    max_length: int = 8192
    max_samples: int | None = None
    overlength: Literal["skip", "error"] = "skip"
    train_on_prompt: bool = False
    system_prompt: str | None = None

    def __post_init__(self) -> None:
        if bool(self.repo_id) == bool(self.local_path):
            raise ValueError("Set exactly one of data.repo_id and data.local_path")
        if self.max_length < 2:
            raise ValueError("data.max_length must be >= 2")
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError("data.max_samples must be positive")
        if self.format == "messages" and self.system_prompt is not None:
            raise ValueError(
                "For messages data, put the system message in the data itself"
            )


@dataclass
class MetricsConfig:
    console: bool = True
    log_file: bool = True
    tensorboard: bool = False
    opentelemetry: bool = False
    otlp_endpoint: str | None = None
    strict: bool = False

    def __post_init__(self) -> None:
        if self.otlp_endpoint is not None and not self.opentelemetry:
            raise ValueError(
                "metrics.otlp_endpoint requires metrics.opentelemetry=true"
            )


@dataclass
class EvaluationConfig:
    holdout_size: int = 0
    data: DataConfig | None = None
    seed: int = 42
    batch_size: int = 1
    every_steps: int = 0
    at_start: bool = True
    plugins: list[PluginConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        if (self.data is None) == (self.holdout_size == 0):
            raise ValueError(
                "Evaluation requires exactly one of data and a positive holdout_size"
            )
        for name in ("holdout_size", "seed", "every_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"evaluation.{name} must be a non-negative integer")
        if (
            not 0 <= self.seed < 2**32
            or type(self.batch_size) is not int
            or self.batch_size < 1
        ):
            raise ValueError("evaluation requires batch_size > 0 and 0 <= seed < 2**32")
        names = [plugin.name for plugin in self.plugins]
        if len(names) != len(set(names)):
            raise ValueError("Evaluation plugin names must be unique")


@dataclass
class SFTRecipeConfig:
    model: ModelConfig
    data: DataConfig
    trainer: SFTConfig = field(default_factory=SFTConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    evaluation: EvaluationConfig | None = None


def load_sft_config(path: str | Path) -> SFTRecipeConfig:
    """Resolve filesystem paths relative to the YAML file, leaving Hub IDs intact."""
    path = Path(path).expanduser().resolve()
    config = load_yaml_config(path, SFTRecipeConfig)

    def local(value: str) -> str:
        expanded = Path(value).expanduser()
        return str((path.parent / expanded).resolve())

    config.trainer.output_dir = local(config.trainer.output_dir)
    for section, name in (
        (config.data, "local_path"),
        (config.data, "cache_dir"),
        (config.model, "cache_dir"),
    ):
        value = getattr(section, name)
        if value is not None:
            setattr(section, name, local(value))
    name = config.model.name_or_path
    if name.startswith((".", "/", "~")) or (path.parent / name).is_dir():
        config.model.name_or_path = local(name)
    if config.evaluation is not None:
        if config.evaluation.data is not None:
            for name in ("local_path", "cache_dir"):
                value = getattr(config.evaluation.data, name)
                if value is not None:
                    setattr(config.evaluation.data, name, local(value))
        for plugin in config.evaluation.plugins:
            try:
                plugin.entrypoint = resolve_entrypoint(plugin.entrypoint, path.parent)
            except ValueError as error:
                from minirl.common.config import ConfigError

                raise ConfigError(
                    f"{path}: evaluation plugin {plugin.name}: {error}"
                ) from error
    return config


def prepare_data(
    config: SFTRecipeConfig,
) -> tuple[PreTrainedTokenizerBase, ChatSFTDataset]:
    tokenizer = load_tokenizer(config.model)
    return tokenizer, _prepare_dataset(config.data, tokenizer)


def _prepare_dataset(
    data: DataConfig, tokenizer: PreTrainedTokenizerBase
) -> ChatSFTDataset:
    path = data.local_path
    if path is None:
        path = hf_hub_download(
            repo_id=data.repo_id,
            filename=data.filename,
            repo_type="dataset",
            revision=data.revision,
            cache_dir=data.cache_dir,
            local_files_only=data.local_files_only,
        )
    dataset = ChatSFTDataset(
        path,
        tokenizer,
        max_length=data.max_length,
        max_samples=data.max_samples,
        format=data.format,
        overlength=data.overlength,
        train_on_prompt=data.train_on_prompt,
        system_prompt=data.system_prompt,
    )
    logger.info("Prepared SFT data: %s", asdict(dataset.stats))
    return dataset


def prepare_sft_data(
    config: SFTRecipeConfig,
) -> tuple[PreTrainedTokenizerBase, ChatSFTDataset, ChatSFTDataset | None, dict]:
    tokenizer, source = prepare_data(config)
    training, validation = source, None
    statistics = asdict(source.stats)
    if config.evaluation is not None:
        evaluation = config.evaluation
        if evaluation.data is not None:
            validation = _prepare_dataset(evaluation.data, tokenizer)
            if set(training.prompt_keys) & set(validation.prompt_keys):
                raise ValueError(
                    "Training and validation data contain overlapping prompts"
                )
        else:
            training, validation = source.split(
                evaluation.holdout_size, evaluation.seed
            )
        statistics["train"] = asdict(training.stats)
        statistics["validation"] = asdict(validation.stats)
        logger.info(
            "SFT split: %d training, %d validation examples",
            len(training),
            len(validation),
        )
    return tokenizer, training, validation, statistics


def _collector(
    config: MetricsConfig,
    output: Path,
    *,
    run_id: str | None = None,
    purge_step: int | None = None,
) -> DataCollector:
    # Close any already-created sinks if a later optional backend fails to initialize.
    with ExitStack() as cleanup:
        sinks: list[MetricSink] = []

        def add(sink: MetricSink) -> None:
            cleanup.callback(sink.close)
            sinks.append(sink)

        if config.console:
            add(LoggingSink())
        if config.log_file:
            add(LoggingSink(output / "train.log"))
        if config.tensorboard:
            add(TensorBoardSink(output / "tensorboard", purge_step=purge_step))
        if config.opentelemetry:
            add(OpenTelemetrySink(endpoint=config.otlp_endpoint))
        collector = DataCollector(sinks, strict=config.strict, run_id=run_id)
        cleanup.pop_all()
    return collector


def run_sft(config: SFTRecipeConfig, *, resume: bool = False) -> Path:
    output = Path(config.trainer.output_dir)
    checkpoint = output / "trainer_state.pt"
    if resume and not checkpoint.is_file():
        raise FileNotFoundError(f"No checkpoint to resume: {checkpoint}")
    if not resume and checkpoint.exists():
        raise FileExistsError(
            f"{checkpoint} already exists; use --resume or a new output_dir"
        )
    plugins = (
        [(plugin, load_plugin(plugin)) for plugin in config.evaluation.plugins]
        if config.evaluation
        else []
    )
    set_seed(config.trainer.seed)
    tokenizer, dataset, validation, statistics = prepare_sft_data(config)
    manifest = {
        "model": asdict(config.model),
        "data": asdict(config.data),
        "dataset_fingerprint": dataset.fingerprint,
        "evaluation": asdict(config.evaluation) if config.evaluation else None,
        "evaluation_fingerprint": validation.fingerprint if validation else None,
        "evaluation_records_fingerprint": hashlib.sha256(
            json.dumps(validation.records, sort_keys=True).encode()
        ).hexdigest()
        if validation
        else None,
        "plugin_fingerprints": {
            plugin.name: plugin_fingerprint(plugin) for plugin, _ in plugins
        },
    }
    manifest_path = output / "recipe_state.json"
    if resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError(
                "Recipe model, data, evaluation/plugin configuration, or dataset changed since checkpoint"
            )
    output.mkdir(parents=True, exist_ok=True)
    model = load_policy(config.model, tokenizer)
    evaluator = (
        SFTEvaluator(
            model,
            tokenizer,
            validation,
            output,
            batch_size=config.evaluation.batch_size,
            plugins=plugins,
            seed=config.evaluation.seed,
        )
        if validation is not None and config.evaluation is not None
        else None
    )
    trainer = SFTTrainer(
        config.trainer,
        model,
        dataset,
        SFTCollator(tokenizer.pad_token_id),
        collector=DataCollector([]),
        evaluator=evaluator,
        eval_steps=config.evaluation.every_steps if config.evaluation else 0,
        eval_at_start=config.evaluation.at_start if config.evaluation else False,
    )
    if resume:
        trainer.load_checkpoint()
    with _collector(
        config.metrics,
        output,
        run_id=trainer.collector.run_id,
        purge_step=trainer.global_step + 1 if resume else 0,
    ) as collector:
        trainer.collector = collector
        (output / "recipe.yaml").write_text(dump_yaml_config(config), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (output / "data_stats.json").write_text(
            json.dumps(statistics, indent=2), encoding="utf-8"
        )
        source_stats = {
            f"data/source/{name}": value
            for name, value in statistics.items()
            if isinstance(value, (int, float))
        }
        source_stats["data/source/filtered_fraction"] = (
            statistics["skipped_too_long"] / statistics["rows_seen"]
        )
        for prefix, values in (
            ("train", asdict(dataset.stats)),
            ("validation", asdict(validation.stats) if validation else {}),
        ):
            source_stats.update(
                {f"data/{prefix}/{name}": value for name, value in values.items()}
            )
        collector.record(source_stats, step=trainer.global_step)
        trainer.train()
        destination = output / "final"
        model.config.use_cache = True
        model.save_pretrained(destination, safe_serialization=True)
        tokenizer.save_pretrained(destination)
    logger.info("SFT policy ready for RL initialization: %s", destination)
    return destination
