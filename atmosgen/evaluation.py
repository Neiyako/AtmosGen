"""Quality evaluation orchestration (Phase 2).

`atmosgen evaluate --input data.nc --real era5_local.nc` produces
``quality_report.json`` (+ markdown) next to the dataset: statistical,
temporal and spatial modules from :mod:`atmosgen.validation`, with physical
validation arriving in Phase 3.

Missing ``--real`` is an explicit NOT_RUN, never a silent pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import xarray as xr

from .config import GenerationConfigFile
from .contract import CONTRACT_ATTR, ContractError, read_contract
from .data.io import open_source
from .runner import environment_snapshot, write_json


class EvaluationError(RuntimeError):
    """Raised when evaluation cannot produce a report."""


def _not_run(reason: str) -> dict[str, Any]:
    return {
        "status": "NOT_RUN",
        "metrics": {},
        "notes": [reason],
    }


def align_real_to_synthetic(real: xr.Dataset, synthetic: xr.Dataset) -> xr.Dataset:
    """Select the real dataset on the *exact* coordinates of ``synthetic``.

    The synthetic grid coordinates are real values (selection/thinning), so a
    nearest-with-tolerance match must always cover them; a failure means the
    two datasets genuinely disagree, which callers must see as an error.
    """

    aligned = real
    for coord in ("time", "level", "latitude", "longitude"):
        if coord not in synthetic.coords:
            continue
        if coord not in real.coords:
            raise EvaluationError(
                f"real 数据缺少 {coord} 坐标，无法对齐到合成数据网格"
            )
        targets = np.asarray(synthetic[coord].values)
        if targets.dtype.kind == "M":
            tolerance = np.timedelta64(1, "ns")
        elif coord == "level" and real[coord].dtype.kind in "iu":
            # int-typed level coordinates need a small tolerance for round-off
            tolerance = 0.51
        else:
            tolerance = 1e-6
        try:
            aligned = aligned.sel({coord: list(targets)}, method="nearest", tolerance=tolerance)
        except KeyError as exc:
            raise EvaluationError(
                f"real 数据在 {coord}=…{targets[:3]}… 处对齐失败（网格不一致）: {exc}"
            ) from exc
    mismatch = {
        name: (aligned.sizes[name], synthetic.sizes[name])
        for name in synthetic.sizes
        if aligned.sizes.get(name) != synthetic.sizes[name]
    }
    if mismatch:
        raise EvaluationError(
            f"real 对齐后尺寸与合成数据不一致: {mismatch}；无法做逐点对比"
        )
    return aligned.load()


def run_evaluation(
    synthetic_path: str | Path,
    *,
    real_path: str | Path | None = None,
    config: GenerationConfigFile | Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run all available validation modules and write the quality report.

    Returns the report document; caller gets ``report["report_paths"]`` with
    the written JSON/MD locations.
    """

    from .validation import report as report_mod
    from .validation.statistical import run_statistical_validation
    from .validation.spatial import run_spatial_validation
    from .validation.temporal import run_temporal_validation

    synth_path = Path(synthetic_path)
    synth = open_source(synth_path, chunks=None)
    try:
        synthetic = synth.load()
        contract = read_contract(synthetic)
    finally:
        synth.close()

    thresholds: dict[str, float] = {}
    if isinstance(config, GenerationConfigFile):
        thresholds = dict(config.validation.thresholds_overrides)

    real_dataset: xr.Dataset | None = None
    if real_path is not None:
        real_target = Path(real_path)
        if not real_target.exists():
            raise FileNotFoundError(f"--real={real_target} 不存在")
        real_handle = open_source(real_target, chunks=None)
        try:
            real_dataset = align_real_to_synthetic(real_handle, synthetic)
        except Exception:
            real_handle.close()
            raise
        real_handle.close()

    if real_dataset is None:
        reason = (
            "--real 未提供：statistical/temporal/spatial 指标需要真实参考数据；"
            "本次评价整体 NOT_RUN（不是静默通过）"
        )
        modules: dict[str, dict[str, Any]] = {
            "statistical": _not_run(reason),
            "temporal": _not_run(reason),
            "spatial": _not_run(reason),
        }
    else:
        modules = {
            "statistical": run_statistical_validation(real_dataset, synthetic, thresholds=thresholds),
            "temporal": run_temporal_validation(real_dataset, synthetic, thresholds=thresholds),
            "spatial": run_spatial_validation(real_dataset, synthetic, thresholds=thresholds),
        }

    overall = report_mod.aggregate_status([block["status"] for block in modules.values()])
    quality: dict[str, Any] = {
        "meta": {
            "input": str(synth_path),
            "real": str(real_path) if real_path else None,
            "contract": {
                key: contract.get(key)
                for key in (
                    "schema_version",
                    "generation_method",
                    "model_name",
                    "model_version",
                    "random_seed",
                )
            },
            "threshold_overrides": thresholds,
            "threshold_source": "module defaults" if not thresholds else "module defaults + overrides",
            "environment": environment_snapshot(),
        },
        "statistical": modules["statistical"],
        "temporal": modules["temporal"],
        "spatial": modules["spatial"],
        "overall_status": overall,
    }

    target_dir = Path(output_dir) if output_dir else synth_path.parent
    json_path = target_dir / "quality_report.json"
    md_path = target_dir / "quality_report.md"
    if json_path.exists() and not overwrite:
        raise FileExistsError(
            f"{json_path} 已存在；传 overwrite=True（CLI：--overwrite）重新生成"
        )
    write_json(json_path, quality)
    md_path.write_text(report_mod.render_markdown(quality), encoding="utf-8")
    quality["report_paths"] = {"json": str(json_path), "md": str(md_path)}
    return quality


def read_quality_report(path: str | Path) -> dict[str, Any]:
    """Load a quality report JSON (accepts the report file or its run directory)."""

    target = Path(path)
    if target.is_dir():
        target = target / "quality_report.json"
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError(f"quality report not found: {target}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"{target} 不是有效 JSON: {exc}") from exc
    if not isinstance(payload, dict) or "overall_status" not in payload:
        raise EvaluationError(
            f"{target} 缺少 overall_status；它似乎不是 AtmosGen 的 quality_report.json"
        )
    return payload


__all__ = ["EvaluationError", "align_real_to_synthetic", "read_quality_report", "run_evaluation"]
