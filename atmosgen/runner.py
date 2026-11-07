"""Run directories, seeding and resource accounting (plan §2.1)."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def set_global_seed(seed: int) -> dict[str, Any]:
    """Seed ``random``/``numpy`` (and torch when importable) deterministically."""

    random.seed(seed)
    np.random.seed(seed % (2**32))
    seeded = {"python_random": True, "numpy": True, "torch": False}
    try:  # optional: torch is not a Phase 1 dependency
        import torch  # type: ignore

        torch.manual_seed(seed)
        seeded["torch"] = True
    except Exception:
        pass
    return seeded


def make_run_id(kind: str, model: str, *, seed: int, timestamp: datetime | None = None) -> str:
    """``<mode>_<model>_<YYYYmmdd-HHMMSS>_<seed>`` as specified in DATA_SCHEMA §4."""

    moment = timestamp or datetime.now()
    safe_model = model.replace("/", "_").replace(" ", "_")
    return f"{kind}_{safe_model}_{moment:%Y%m%d-%H%M%S}_{seed}"


def create_run_dir(
    output_dir: str | Path,
    *,
    kind: str,
    model: str,
    seed: int,
    timestamp: datetime | None = None,
    exist_ok: bool = False,
) -> Path:
    """Create (and return) ``<output_dir>/<run_id>``."""

    run_dir = Path(output_dir) / make_run_id(kind, model, seed=seed, timestamp=timestamp)
    if run_dir.exists() and not exist_ok:
        raise FileExistsError(f"run directory already exists: {run_dir}")
    for sub in ("figures",):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def snapshot_config(config_path: str | Path, run_dir: str | Path) -> Path:
    """Copy the raw YAML into the run directory (provenance, plan §10.4)."""

    source = Path(config_path)
    target = Path(run_dir) / "config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def _sanitise_json(value: Any) -> Any:
    """Replace NaN/Inf with null so the output is RFC-8259 valid JSON."""

    if isinstance(value, Mapping):
        return {key: _sanitise_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitise_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _sanitise_json(value.tolist())
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write indented UTF-8 JSON, creating parent directories.

    Non-finite floats (NaN/Inf) are written as ``null``: bare ``NaN`` is not
    valid JSON and breaks strict parsers (Node/Go/Rust/serde).
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sanitised = _sanitise_json(payload)
    target.write_text(
        json.dumps(
            sanitised,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return target


def _json_default(value: Any) -> Any:  # pragma: no cover - defensive serialisation
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serialisable")


@dataclass
class ResourceTracker:
    """Measure wall time and peak RSS of one stage.

    Peak RSS is sampled by a background thread every ``sample_interval_s``
    seconds. ``ru_maxrss`` (process lifetime maximum, monotonic) is reported
    alongside as ``process_max_rss_mb`` so the two can be cross-checked:
    sampling can miss a very short allocation spike, while ``ru_maxrss`` may
    include earlier stages of the same process.
    """

    label: str = ""
    started_at: float = 0.0
    elapsed_seconds: float = 0.0
    peak_rss_mb: float = 0.0
    process_max_rss_mb: float = 0.0
    peak_rss_source: str = "sampled"
    sample_interval_s: float = 0.05
    _process: Any = field(default=None, repr=False)
    _stop_event: Any = field(default=None, repr=False)
    _thread: Any = field(default=None, repr=False)
    _max_rss_bytes: int = 0

    def start(self) -> "ResourceTracker":
        self.started_at = time.perf_counter()
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
            self._max_rss_bytes = int(self._process.memory_info().rss)
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        except Exception:
            self._process = None
            self.peak_rss_source = "unavailable"
        return self

    def _sample_loop(self) -> None:
        while self._stop_event is not None and not self._stop_event.is_set():
            try:
                rss = int(self._process.memory_info().rss)
            except Exception:  # pragma: no cover - process exiting
                return
            if rss > self._max_rss_bytes:
                self._max_rss_bytes = rss
            self._stop_event.wait(self.sample_interval_s)

    def stop(self) -> dict[str, Any]:
        self.elapsed_seconds = round(time.perf_counter() - self.started_at, 3)
        if self._thread is not None and self._stop_event is not None:
            self._stop_event.set()
            self._thread.join(timeout=2.0)
        self.peak_rss_mb = round(self._max_rss_bytes / (1024**2), 1)
        try:
            import resource

            raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # macOS reports bytes, Linux reports kilobytes
            self.process_max_rss_mb = (
                round(raw / (1024**2), 1) if sys.platform == "darwin" else round(raw / 1024, 1)
            )
        except Exception:  # pragma: no cover - platform specific
            self.process_max_rss_mb = 0.0
        return self.as_dict()

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_rss_mb": self.peak_rss_mb,
            "peak_rss_source": self.peak_rss_source,
            "process_max_rss_mb": self.process_max_rss_mb,
        }


def environment_snapshot() -> dict[str, Any]:
    """Environment fingerprint recorded next to every run (Test E support)."""

    snapshot: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }
    try:
        import numpy
        import pydantic
        import xarray

        snapshot["numpy"] = numpy.__version__
        snapshot["xarray"] = xarray.__version__
        snapshot["pydantic"] = pydantic.__version__
    except Exception:  # pragma: no cover
        pass
    try:
        import sdv

        snapshot["sdv"] = getattr(sdv, "__version__", "unknown")
    except Exception:
        snapshot["sdv"] = "not installed"
    try:
        import sdmetrics

        snapshot["sdmetrics"] = getattr(sdmetrics, "__version__", "unknown")
    except Exception:
        snapshot["sdmetrics"] = "not installed"
    return snapshot


def write_environment_snapshot(run_dir: str | Path) -> Path:
    """Persist :func:`environment_snapshot` as ``<run_dir>/environment.json``."""

    return write_json(Path(run_dir) / "environment.json", environment_snapshot())


__all__ = [
    "ResourceTracker",
    "create_run_dir",
    "environment_snapshot",
    "make_run_id",
    "set_global_seed",
    "snapshot_config",
    "write_environment_snapshot",
    "write_json",
]
