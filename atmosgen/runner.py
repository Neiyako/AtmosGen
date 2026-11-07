"""Run directories, seeding and resource accounting (plan §2.1)."""

from __future__ import annotations

import json
import os
import platform
import random
import shutil
import sys
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
    (run_dir).mkdir(parents=True, exist_ok=True)
    return run_dir


def snapshot_config(config_path: str | Path, run_dir: str | Path) -> Path:
    """Copy the raw YAML into the run directory (provenance, plan §10.4)."""

    source = Path(config_path)
    target = Path(run_dir) / "config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write indented UTF-8 JSON, creating parent directories."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
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
    if isinstance(value, (datetime,)):
        return value.isoformat()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serialisable")


@dataclass
class ResourceTracker:
    """Measure wall time and peak RSS of one stage."""

    label: str = ""
    started_at: float = 0.0
    elapsed_seconds: float = 0.0
    peak_rss_mb: float = 0.0
    _process: Any = field(default=None, repr=False)

    def start(self) -> "ResourceTracker":
        self.started_at = time.perf_counter()
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
            self._process.cpu_percent(None)
        except Exception:
            self._process = None
        return self

    def stop(self) -> dict[str, float]:
        self.elapsed_seconds = round(time.perf_counter() - self.started_at, 3)
        if self._process is not None:
            try:
                self.peak_rss_mb = round(self._process.memory_info().rss / (1024**2), 1)
            except Exception:
                self.peak_rss_mb = 0.0
        return self.as_dict()

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_rss_mb": self.peak_rss_mb,
        }


def environment_snapshot() -> dict[str, Any]:
    """Environment fingerprint recorded in run metadata (Test E support)."""

    snapshot: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }
    try:
        import xarray
        import numpy
        import pydantic

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


__all__ = [
    "ResourceTracker",
    "create_run_dir",
    "environment_snapshot",
    "make_run_id",
    "set_global_seed",
    "snapshot_config",
    "write_json",
]
