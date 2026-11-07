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
    # Physical validation (plan §5.4): hard constraints plus soft indicators on
    # the synthetic dataset itself; needs no real reference, so it runs even
    # without --real. physical_parameters (u0/v0/kappa/BC) come from the
    # embedded contract so Mode 2 datasets get their mass-budget checks.
    try:
        from .validation.physical import run_physical_validation

        modules["physical"] = run_physical_validation(
            synthetic,
            thresholds=thresholds,
            physical_parameters=contract.get("physical_parameters"),
            real=real_dataset,
        )
    except Exception as exc:  # noqa: BLE001 - a broken validator must not crash evaluate
        modules["physical"] = _not_run(
            f"physical 验证未运行（模块异常）: {type(exc).__name__}: {exc}"
        )

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
        "physical": modules["physical"],
        "overall_status": overall,
        "coverage": {
            "implemented": ["statistical", "temporal", "spatial", "physical"],
            "not_implemented": [],
        },
    }

    target_dir = Path(output_dir) if output_dir else synth_path.parent
    json_path = target_dir / "quality_report.json"
    md_path = target_dir / "quality_report.md"
    if json_path.exists() and not overwrite:
        raise FileExistsError(
            f"{json_path} 已存在；传 overwrite=True（CLI：--overwrite）重新生成"
        )
    write_json(json_path, quality)
    markdown = report_mod.render_markdown(quality)
    figure_paths: list[str] = []
    if real_dataset is not None:
        figure_paths = _write_figures(real_dataset, synthetic, target_dir)
    if figure_paths:
        markdown += "\n## 图\n\n"
        for relative in figure_paths:
            markdown += f"![{relative}]({relative})\n\n"
    md_path.write_text(markdown, encoding="utf-8")
    quality["report_paths"] = {"json": str(json_path), "md": str(md_path)}
    quality["figures"] = figure_paths
    return quality


def _write_figures(real: xr.Dataset, synthetic: xr.Dataset, out_dir: Path) -> list[str]:
    """Write the plan §5.5 figures (distributions / ACF / spectrum / variogram).

    Figure generation is best-effort: a plotting failure must not invalidate a
    report whose numbers are already computed.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib is a declared dependency
        return []
    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    variables = [name for name in synthetic.data_vars if name in real.data_vars][:6]
    try:
        # 1) marginal distributions
        if variables:
            figure, axes = plt.subplots(1, len(variables), figsize=(3.2 * len(variables), 2.8))
            axes = np.atleast_1d(axes)
            for axis, name in zip(axes, variables):
                real_values = np.asarray(real[name].values, dtype=float).ravel()
                syn_values = np.asarray(synthetic[name].values, dtype=float).ravel()
                real_values = real_values[np.isfinite(real_values)]
                syn_values = syn_values[np.isfinite(syn_values)]
                axis.hist(real_values, bins=60, density=True, alpha=0.5, label="real")
                axis.hist(syn_values, bins=60, density=True, alpha=0.5, label="synthetic")
                axis.set_title(name)
                axis.legend(fontsize=6)
            figure.suptitle("Marginal distributions: real vs synthetic")
            figure.tight_layout()
            target = fig_dir / "distributions.png"
            figure.savefig(target, dpi=110)
            plt.close(figure)
            written.append(str(target.relative_to(Path(out_dir))))

        # time series for ACF / spectrum: first variable with a level axis
        series_name = next(
            (name for name in variables if "level" in synthetic[name].dims and "time" in synthetic[name].dims),
            variables[0] if variables else None,
        )
        if series_name is not None and synthetic.sizes.get("time", 1) > 3:
            sel = {}
            if "level" in synthetic[series_name].dims:
                sel["level"] = 0
            if "latitude" in synthetic[series_name].dims:
                sel["latitude"] = 0
            if "longitude" in synthetic[series_name].dims:
                sel["longitude"] = 0
            real_series = np.asarray(real[series_name].isel(**sel).values, dtype=float).ravel()
            syn_series = np.asarray(synthetic[series_name].isel(**sel).values, dtype=float).ravel()
            figure, axes = plt.subplots(1, 2, figsize=(9, 3))
            # ACF
            lags = range(0, min(10, real_series.size - 1))
            axes[0].plot(list(lags), [_acf(real_series, lag) for lag in lags], "o-", label="real")
            axes[0].plot(list(lags), [_acf(syn_series, lag) for lag in lags], "s-", label="synthetic")
            axes[0].set_title(f"ACF of {series_name}")
            axes[0].set_xlabel("lag")
            axes[0].legend(fontsize=7)
            # power spectrum
            real_freq, real_power = _spectrum(real_series)
            syn_freq, syn_power = _spectrum(syn_series)
            axes[1].loglog(real_freq[1:], real_power[1:], label="real")
            axes[1].loglog(syn_freq[1:], syn_power[1:], label="synthetic")
            axes[1].set_title(f"Power spectrum of {series_name}")
            axes[1].set_xlabel("frequency")
            axes[1].legend(fontsize=7)
            figure.tight_layout()
            target = fig_dir / "acf_spectrum.png"
            figure.savefig(target, dpi=110)
            plt.close(figure)
            written.append(str(target.relative_to(Path(out_dir))))

        # variogram of the first 2-D field
        field_name = next(
            (name for name in variables if "latitude" in synthetic[name].dims and "longitude" in synthetic[name].dims),
            None,
        )
        if field_name is not None:
            field = synthetic[field_name]
            if "level" in field.dims:
                field = field.isel(level=0)
            field = field.isel(time=0)
            real_field = real[field_name]
            if "level" in real_field.dims:
                real_field = real_field.isel(level=0)
            real_field = real_field.isel(time=0)
            figure, axis = plt.subplots(figsize=(5, 3.2))
            axis.plot(range(1, 6), [_variogram(np.asarray(field.values, dtype=float), lag) for lag in range(1, 6)], "o-", label="synthetic")
            axis.plot(range(1, 6), [_variogram(np.asarray(real_field.values, dtype=float), lag) for lag in range(1, 6)], "s-", label="real")
            axis.set_title(f"Semi-variogram of {field_name}")
            axis.set_xlabel("lag (grid cells)")
            axis.legend(fontsize=7)
            figure.tight_layout()
            target = fig_dir / "variogram.png"
            figure.savefig(target, dpi=110)
            plt.close(figure)
            written.append(str(target.relative_to(Path(out_dir))))
    except Exception:  # pragma: no cover - best effort only
        return written
    return written


def _acf(values: np.ndarray, lag: int) -> float:
    if lag == 0:
        return 1.0
    a = values[:-lag] - values.mean()
    b = values[lag:] - values.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator else 0.0


def _spectrum(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.abs(np.fft.rfft(values - values.mean())) ** 2
    frequencies = np.fft.rfftfreq(values.size)
    return frequencies, spectrum


def _variogram(field: np.ndarray, lag: int) -> float:
    differences = []
    if field.shape[0] > lag:
        differences.append(field[lag:, :] - field[:-lag, :])
    if field.shape[1] > lag:
        differences.append(field[:, lag:] - field[:, :-lag])
    if not differences:
        return float("nan")
    squared = np.concatenate([(difference**2).ravel() for difference in differences])
    squared = squared[np.isfinite(squared)]
    return float(0.5 * squared.mean()) if squared.size else float("nan")


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
