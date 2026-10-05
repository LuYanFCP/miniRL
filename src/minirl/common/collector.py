"""Collect scalar training metrics and fan them out to independent sinks."""

from __future__ import annotations

import csv
import json
import logging
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Protocol, Self
from uuid import uuid4

import torch

if TYPE_CHECKING:
    from opentelemetry.metrics import Gauge
    from opentelemetry.sdk.metrics import MeterProvider

type Scalar = int | float
type MetricValue = Scalar | torch.Tensor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricRecord:
    """A snapshot containing only CPU scalars, never tensors or autograd graphs."""

    metrics: Mapping[str, Scalar]
    step: int
    timestamp: float
    run_id: str


class MetricSink(Protocol):
    def write(self, record: MetricRecord) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class DataCollector:
    """Normalize metrics once and send them to all sinks without retaining history.

    Defaults to console logging. Pass an empty iterable to disable output.
    Use as a context manager, or call ``close()`` when the run is finished.
    Sink failures are logged and isolated unless ``strict=True``; invalid metric
    values always raise. Call from the training thread (and one distributed rank).
    """

    def __init__(
        self,
        sinks: Iterable[MetricSink] | None = None,
        *,
        run_id: str | None = None,
        strict: bool = False,
    ) -> None:
        self.sinks = tuple(sinks) if sinks is not None else (LoggingSink(),)
        self.run_id = run_id if run_id is not None else uuid4().hex
        self.strict = strict
        self._closed = False

    def record(self, metrics: Mapping[str, MetricValue], *, step: int) -> MetricRecord:
        if self._closed:
            raise RuntimeError("DataCollector is closed")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("step must be a non-negative integer")

        values: dict[str, Scalar] = {}
        for name, value in metrics.items():
            scalar: object = value
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must be non-empty strings")
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(
                        f"Metric {name!r} must be scalar; got shape {tuple(value.shape)}"
                    )
                scalar = value.detach().item()
            if not isinstance(scalar, Real):
                raise TypeError(
                    f"Metric {name!r} must be a real number or scalar tensor"
                )
            values[name] = (
                int(scalar) if isinstance(scalar, Integral) else float(scalar)
            )

        record = MetricRecord(MappingProxyType(values), step, time.time(), self.run_id)
        self._dispatch(lambda sink: sink.write(record))
        return record

    def _dispatch(self, action: Callable[[MetricSink], None]) -> None:
        failures: list[Exception] = []
        for sink in self.sinks:
            try:
                action(sink)
            except Exception as exc:
                if self.strict:
                    failures.append(exc)
                else:
                    logger.warning(
                        "Metric sink %s failed", type(sink).__name__, exc_info=True
                    )
        if failures:
            raise ExceptionGroup("Metric sink failures", failures)

    def flush(self) -> None:
        if not self._closed:
            self._dispatch(lambda sink: sink.flush())

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._dispatch(lambda sink: sink.close())

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("DataCollector is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_value is None:
            self.close()
        else:
            try:
                self.close()
            except BaseException as cleanup_error:
                exc_value.add_note(
                    f"DataCollector.close also failed: {cleanup_error!r}"
                )
                logger.warning(
                    "Collector close failed while handling another error", exc_info=True
                )


class LoggingSink:
    """Write to an application logger, or an owned console/file handler.

    Does not configure the root logger or close handlers owned by the caller.
    """

    def __init__(
        self,
        log_file: str | Path | None = None,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        if logger is not None and log_file is not None:
            raise ValueError("Pass either logger or log_file")
        self._owns_logger = logger is None
        # Owned sinks need separate logger instances to avoid sharing handlers.
        self.logger = (
            logger
            if logger is not None
            else logging.Logger("minirl.training", logging.INFO)  # noqa: LOG001
        )
        if self._owns_logger:
            handler: logging.Handler
            if log_file is None:
                handler = logging.StreamHandler()
            else:
                path = Path(log_file)
                path.parent.mkdir(parents=True, exist_ok=True)
                handler = logging.FileHandler(path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
            self.logger.addHandler(handler)

    def write(self, record: MetricRecord) -> None:
        fields = [f"run_id={record.run_id}", f"step={record.step}"]
        fields.extend(f"{name}={value}" for name, value in record.metrics.items())
        self.logger.info(
            " | ".join(fields),
            extra={
                "training_metrics": dict(record.metrics),
                "training_step": record.step,
                "training_timestamp": record.timestamp,
                "training_run_id": record.run_id,
            },
        )

    def flush(self) -> None:
        for handler in self.logger.handlers:
            handler.flush()

    def close(self) -> None:
        self.flush()
        if self._owns_logger:
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)


class JSONLSink:
    """Append complete metric records and flush each write for live monitoring."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a", encoding="utf-8")

    def write(self, record: MetricRecord) -> None:
        line = json.dumps(
            {
                "run_id": record.run_id,
                "step": record.step,
                "timestamp": record.timestamp,
                "metrics": dict(record.metrics),
            },
            ensure_ascii=False,
            allow_nan=False,
        )
        self.stream.write(line + "\n")
        self.flush()

    def flush(self) -> None:
        if not self.stream.closed:
            self.stream.flush()

    def close(self) -> None:
        self.stream.close()


class CSVSink:
    """Append one row per scalar, allowing training and evaluation keys to differ.

    Replayed steps on resume remain in the raw history, identified by run/step/time.
    """

    columns = ("run_id", "step", "timestamp", "metric", "value")

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        has_header = path.exists() and path.stat().st_size > 0
        if has_header:
            with path.open(encoding="utf-8", newline="") as existing:
                if next(csv.reader(existing), None) != list(self.columns):
                    raise ValueError(f"Incompatible metrics CSV header: {path}")
        self.stream = path.open("a", encoding="utf-8", newline="")
        self.writer = csv.writer(self.stream)
        if not has_header:
            self.writer.writerow(self.columns)
            self.flush()

    def write(self, record: MetricRecord) -> None:
        self.writer.writerows(
            (record.run_id, record.step, record.timestamp, name, value)
            for name, value in record.metrics.items()
        )
        self.flush()

    def flush(self) -> None:
        if not self.stream.closed:
            self.stream.flush()

    def close(self) -> None:
        self.stream.close()


class TensorBoardSink:
    """Write per-step scalar events; requires ``minirl[tensorboard]``.

    Use a separate log directory for each run. ``purge_step`` can hide stale
    events when resuming a checkpoint in an existing directory.
    """

    def __init__(self, log_dir: str | Path, *, purge_step: int | None = None) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "TensorBoardSink requires: pip install 'minirl[tensorboard]'"
            ) from exc
        self.writer = SummaryWriter(log_dir=str(log_dir), purge_step=purge_step)

    def write(self, record: MetricRecord) -> None:
        for name, value in record.metrics.items():
            self.writer.add_scalar(
                name, value, global_step=record.step, walltime=record.timestamp
            )

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


class OpenTelemetrySink:
    """Export current metric values via OTLP/HTTP or a supplied SDK provider.

    Requires ``minirl[opentelemetry]``. ``endpoint`` is the full metrics URL
    (including ``/v1/metrics``); None uses the standard OTEL environment variables.
    A supplied provider is flushed but never shut down. No global provider is set.
    Metrics use stable run attributes; step is a gauge, not a time-series label.
    Periodic gauge export keeps the latest value between export intervals, while
    TensorBoard/logging retain each call to ``record``.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        service_name: str = "minirl",
        export_interval_millis: float = 10_000,
        meter_provider: MeterProvider | None = None,
    ) -> None:
        if meter_provider is not None and endpoint is not None:
            raise ValueError("Pass either meter_provider or endpoint")
        if export_interval_millis <= 0:
            raise ValueError("export_interval_millis must be positive")
        try:
            from opentelemetry.sdk.metrics import MeterProvider
        except ImportError as exc:
            raise ImportError(
                "OpenTelemetrySink requires: pip install 'minirl[opentelemetry]'"
            ) from exc

        self._owns_provider = meter_provider is None
        if meter_provider is None:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import Resource

            reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=endpoint),
                export_interval_millis=export_interval_millis,
            )
            meter_provider = MeterProvider(
                metric_readers=[reader],
                resource=Resource.create({"service.name": service_name}),
            )
        self.provider = meter_provider
        self.meter = self.provider.get_meter("minirl.training")
        self._gauges: dict[str, Gauge] = {}
        self._step_gauge = self.meter.create_gauge("minirl.global_step")
        self._closed = False

    def write(self, record: MetricRecord) -> None:
        # Validate before writing so a malformed name cannot partially export a record.
        for name in record.metrics:
            if (
                not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_./-]{0,254}", name)
                or name.lower() == "minirl.global_step"
            ):
                raise ValueError(
                    f"Invalid or reserved OpenTelemetry metric name: {name!r}"
                )
        attributes = {"run.id": record.run_id}
        for name, value in record.metrics.items():
            if name not in self._gauges:
                self._gauges[name] = self.meter.create_gauge(name)
            self._gauges[name].set(value, attributes=attributes)
        self._step_gauge.set(record.step, attributes=attributes)

    def flush(self) -> None:
        if not self._closed and not self.provider.force_flush():
            raise TimeoutError("OpenTelemetry metrics flush timed out")

    def close(self) -> None:
        if not self._closed:
            try:
                self.flush()
            finally:
                self._closed = True
                if self._owns_provider:
                    self.provider.shutdown()
