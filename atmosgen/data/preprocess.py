"""Real-data preprocessing: local ERA5 / CDS sample → canonical AtmosGen datasets.

Phase 1 scope (Test A, plan §7):

- read the local ERA5 archive (``/Volumes/Untitled``, **read-only**) with
  explicit time / level / spatial subsets;
- derive ``tcwv`` by vertical integration of ``q`` and mark it as derived;
- write a raw slice (write-once, ``data/raw/``) and a canonical processed
  dataset (``data/processed/``) that carries the full data contract;
- prove the source archive was not modified (size+mtime always, SHA-256 for
  files ≤ ``SHA256_SIZE_LIMIT``);
- emit a machine-readable Test A report plus a human-readable table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import xarray as xr

from ..config import IngestConfigFile, SubsetConfig, config_sha256
from ..contract import (
    VAR_SPECS,
    build_contract,
    known_variables,
    sha256_file,
    validate_contract,
    write_contract_file,
)
from .io import (
    DEFAULT_RAW_DIR,
    assert_writable_target,
    file_fingerprint,
    open_source,
    resolve_path,
    save_canonical,
    sidecar_contract_path,
)
from ..runner import (
    ResourceTracker,
    create_run_dir,
    make_run_id,
    set_global_seed,
    snapshot_config,
    write_environment_snapshot,
    write_json,
)

GRAVITY = 9.80665  # m s-2, WMO standard gravity
SHA256_SIZE_LIMIT = 64 * 1024 * 1024  # skip full-file digests above this size
DEFAULT_VARIABLES = ("t", "q", "u", "v")


class IngestError(RuntimeError):
    """Raised when an ingest run cannot produce a trustworthy dataset."""


# --------------------------------------------------------------------------- #
# Report dataclass
# --------------------------------------------------------------------------- #
@dataclass
class IngestReport:
    """Everything Test A needs to know about one ingest run."""

    kind: str
    status: str = "FAIL"
    run_dir: Path | None = None
    report_json: Path | None = None
    raw_path: Path | None = None
    processed_path: Path | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    coordinates: dict[str, Any] = field(default_factory=dict)
    time: dict[str, Any] = field(default_factory=dict)
    missing: dict[str, Any] = field(default_factory=dict)
    derived: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, str] = field(default_factory=dict)
    readonly: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    contract_sha256: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kind": self.kind,
            "status": self.status,
            "run_dir": _str(self.run_dir),
            "report_json": _str(self.report_json),
            "raw_path": _str(self.raw_path),
            "processed_path": _str(self.processed_path),
            "sources": self.sources,
            "variables": self.variables,
            "coordinates": self.coordinates,
            "time": self.time,
            "missing": self.missing,
            "derived": self.derived,
            "outputs": self.outputs,
            "checks": self.checks,
            "readonly": self.readonly,
            "resources": self.resources,
            "contract_sha256": self.contract_sha256,
            "notes": self.notes,
        }
        return payload

    def render_text(self) -> str:
        """Human-readable summary printed by ``atmosgen ingest``."""

        lines: list[str] = []
        lines.append(f"Ingest report ({self.kind}) — status: {self.status}")
        lines.append(f"run directory: {self.run_dir}")
        lines.append("")
        lines.append("sources (opened read-only):")
        for source in self.sources:
            lines.append(
                f"  - {source['path']}  {source['size_mb']:.1f} MB  "
                f"unchanged={source.get('unchanged')}  sha256={source.get('sha256') or 'skipped'}"
            )
        lines.append("")
        lines.append("variables (name / dims / units / long_name):")
        for name, info in self.variables.items():
            lines.append(
                f"  - {name:<5} {str(info['dims']):<42} {info['units']:<10} {info['long_name']}"
            )
        lines.append("")
        lines.append("coordinates:")
        for name, info in self.coordinates.items():
            lines.append(f"  - {name}: {info}")
        lines.append("")
        time_info = self.time
        lines.append(
            f"time: {time_info.get('start')} .. {time_info.get('end')}  "
            f"steps={time_info.get('steps')}  step_hours={time_info.get('step_hours')}"
        )
        lines.append("")
        lines.append("missing values (NaN count / ratio):")
        for name, info in self.missing.items():
            lines.append(f"  - {name}: {info['count']} / {info['total']} = {info['ratio_percent']:.2f}%")
        lines.append("")
        lines.append("value ranges (min / max / mean):")
        for name, info in self.variables.items():
            lines.append(
                f"  - {name}: {_fmt(info.get('min'))} / {_fmt(info.get('max'))} / {_fmt(info.get('mean'))}"
            )
        if self.derived:
            lines.append("")
            lines.append("derived variables:")
            for name, info in self.derived.items():
                lines.append(f"  - {name}: {info.get('formula')}  finite={info.get('finite')}  "
                             f"min={_fmt(info.get('min'))} max={_fmt(info.get('max'))}")
        lines.append("")
        lines.append("outputs:")
        for key, info in self.outputs.items():
            lines.append(
                f"  - {key}: {info['path']}  {info['size_mb']:.2f} MB  sha256={info['sha256']}"
            )
        lines.append("")
        lines.append("checks:")
        for key, state in self.checks.items():
            lines.append(f"  - {key}: {state}")
        if self.notes:
            lines.append("")
            lines.append("notes:")
            lines.extend(f"  - {note}" for note in self.notes)
        lines.append("")
        lines.append(f"resources: {self.resources}")
        return "\n".join(lines)


def _overall_status(checks: Mapping[str, str]) -> str:
    """Aggregate per-check states without ever upgrading a partial pass.

    ``FAIL`` dominates; any check that is not ``PASS`` (i.e. ``CONDITIONAL_PASS``
    or ``NOT_RUN``) downgrades the overall status to ``CONDITIONAL_PASS``;
    only all-``PASS`` becomes ``PASS``. A ``NOT_RUN`` is never reported as PASS.
    """

    states = set(checks.values())
    if "FAIL" in states:
        return "FAIL"
    if states != {"PASS"}:
        return "CONDITIONAL_PASS"
    return "PASS"


def _str(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if value == 0:
            return "0"
        magnitude = abs(value)
        if magnitude < 1e-3 or magnitude >= 1e5:
            return f"{value:.4e}"
        return f"{value:.{digits}g}"
    return str(value)


# --------------------------------------------------------------------------- #
# Selection helpers
# --------------------------------------------------------------------------- #
def _date_range(start: str, end: str) -> list[date]:
    begin = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    if begin > stop:
        raise IngestError(f"date_range start {begin} is after end {stop}")
    days: list[date] = []
    cursor = begin
    while cursor <= stop:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _pick_data_var(ds: xr.Dataset, canonical: str, mapped: str | None) -> str:
    candidates = [mapped, canonical]
    for name in candidates:
        if name and name in ds.variables:
            return name
    raise IngestError(
        f"source file has neither {mapped!r} nor {canonical!r}; available data variables: "
        f"{list(ds.data_vars)}"
    )


def _horizontal_slice(coord: xr.DataArray, bounds: tuple[float, float] | None) -> slice:
    values = np.asarray(coord.values, dtype=float)
    if bounds is None:
        return slice(None)
    low, high = float(min(bounds)), float(max(bounds))
    descending = values.size > 1 and values[0] > values[-1]
    return slice(high, low) if descending else slice(low, high)


def _select_horizontal(da: xr.DataArray, subset: SubsetConfig) -> xr.DataArray:
    if subset.latitude is None and subset.longitude is None:
        return da
    if "latitude" in da.coords and subset.latitude is not None:
        da = da.sel(latitude=_horizontal_slice(da["latitude"], subset.latitude))
    if "longitude" in da.coords and subset.longitude is not None:
        da = da.sel(longitude=_horizontal_slice(da["longitude"], subset.longitude))
    return da


def _select_times(da: xr.DataArray, subset: SubsetConfig) -> xr.DataArray:
    step = subset.time_step_hours
    if step is None:
        return da
    hours = np.asarray(da["time"].dt.hour.values, dtype=int)
    mask = hours % step == 0
    if not mask.any():
        raise IngestError(f"no time steps match time_step_hours={step}")
    return da.isel(time=np.where(mask)[0])


def _select_levels(
    da: xr.DataArray, levels: list[float] | None, *, keep_all: bool = False
) -> xr.DataArray:
    if keep_all or levels is None:
        return da
    available = [float(value) for value in da["level"].values]
    missing = [value for value in levels if value not in available]
    if missing:
        raise IngestError(f"requested levels {missing} not present; available levels: {available}")
    return da.sel(level=[float(value) for value in sorted(levels)])


# --------------------------------------------------------------------------- #
# tcwv derivation
# --------------------------------------------------------------------------- #
def derive_tcwv(q: xr.DataArray, *, gravity: float = GRAVITY) -> xr.DataArray:
    """``tcwv = (1/g) * ∫ q dp`` over the full available pressure column.

    The integral uses the trapezoidal rule on the native level axis; levels are
    sorted first so the result is independent of the source file's level order.
    The result is a 3-D field ``(time, latitude, longitude)`` in kg m-2 and must
    be labelled as *derived* in the contract.
    """

    if "level" not in q.dims:
        raise IngestError("derive_tcwv requires a 'level' dimension on q")
    q_sorted = q.sortby("level")
    levels_pa = np.asarray(q_sorted["level"].values, dtype=float) * 100.0  # hPa -> Pa
    if levels_pa.size < 2:
        raise IngestError("derive_tcwv requires at least two pressure levels")
    # NOTE: compute positionally in numpy. Building xarray DataArrays for the
    # layer thickness and then adding staggered slices would align on the
    # ``level`` coordinate, silently pairing q(700) with q(850) instead of
    # q(500) with q(700). Keeping the arithmetic positional avoids that class
    # of bug entirely (a unit test pins the analytic constant-q result).
    values = np.asarray(q_sorted.values, dtype="float64")
    thickness_pa = np.diff(levels_pa)
    if np.any(thickness_pa <= 0):
        raise IngestError("derive_tcwv requires strictly increasing (sorted) pressure levels")
    level_axis = q_sorted.get_axis_num("level")
    values = np.moveaxis(values, level_axis, -1)  # level last, other axis order unchanged
    column_values = (0.5 * (values[..., :-1] + values[..., 1:]) * thickness_pa).sum(axis=-1) / gravity
    out_dims = tuple(dim for dim in q_sorted.dims if dim != "level")
    coords = {dim: q_sorted.coords[dim] for dim in out_dims if dim in q_sorted.coords}
    column = xr.DataArray(column_values, dims=out_dims, coords=coords, name="tcwv")
    column.attrs.update(
        {
            "units": "kg m**-2",
            "long_name": "Total column water vapour (derived)",
            "standard_name": "atmosphere_mass_content_of_water_vapor",
            "atmosgen_derived": "tcwv = (1/g) * integral(q dp) over native pressure levels",
            "levels_used": f"{levels_pa.min()/100:.0f}..{levels_pa.max()/100:.0f} hPa ({levels_pa.size} levels)",
        }
    )
    return column


# --------------------------------------------------------------------------- #
# Statistics / description
# --------------------------------------------------------------------------- #
def dataset_statistics(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    """Per-variable min/max/mean/NaN accounting used by the Test A report."""

    stats: dict[str, dict[str, Any]] = {}
    for name in ds.data_vars:
        data = np.asarray(ds[name].values)
        finite_mask = np.isfinite(data)
        finite = data[finite_mask]
        total = int(data.size)
        nan_count = int(total - finite.size)
        stats[name] = {
            "dims": list(ds[name].dims),
            "units": str(ds[name].attrs.get("units", "")),
            "long_name": str(ds[name].attrs.get("long_name", "")),
            "size": total,
            "nan_count": nan_count,
            "nan_ratio_percent": round(100.0 * nan_count / total, 4) if total else 0.0,
            "min": float(finite.min()) if finite.size else float("nan"),
            "max": float(finite.max()) if finite.size else float("nan"),
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": float(finite.std()) if finite.size else float("nan"),
        }
    return stats


def _coordinates_block(ds: xr.Dataset) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for name in ds.coords:
        coord = ds.coords[name]
        values = np.asarray(coord.values)
        if values.dtype.kind == "M":
            block[name] = {
                "size": int(values.size),
                "start": str(np.datetime_as_string(values.min(), unit="s")),
                "end": str(np.datetime_as_string(values.max(), unit="s")),
                "step_hours": _step_hours(values),
            }
        else:
            numeric = values.astype(float) if values.dtype.kind in "fiu" else values
            entry: dict[str, Any] = {"size": int(values.size)}
            if values.dtype.kind in "fiu":
                entry.update(
                    {
                        "min": float(numeric.min()),
                        "max": float(numeric.max()),
                        "step": _step_value(numeric),
                        "ascending": bool(values.size < 2 or numeric[-1] > numeric[0]),
                    }
                )
            if coord.attrs.get("units"):
                entry["units"] = str(coord.attrs["units"])
            block[name] = entry
    return block


def _step_hours(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    deltas = np.diff(values.astype("datetime64[h]").astype("int64"))
    return float(np.median(deltas))


def _step_value(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.round(np.median(np.abs(np.diff(values))), 10))


# --------------------------------------------------------------------------- #
# Source file handling
# --------------------------------------------------------------------------- #
def _prepare_source_file(
    path: Path,
    *,
    canonical: str,
    mapped: str | None,
    subset: SubsetConfig,
    levels: list[float] | None,
    keep_all_levels: bool,
) -> xr.DataArray:
    ds = open_source(path, chunks=None, decode_times=True)
    try:
        var_name = _pick_data_var(ds, canonical, mapped)
        data = ds[var_name]
        available = ds[var_name].attrs.get("units")
        long_name = ds[var_name].attrs.get("long_name")
        # time first (cheapest), then horizontal, then level
        data = _select_times(data, subset)
        data = _select_horizontal(data, subset)
        data = _select_levels(data, levels, keep_all=keep_all_levels)
        data = data.load()  # materialise only the requested slice
    finally:
        ds.close()

    out = data.rename(canonical)
    out.attrs = {
        "units": str(available) if available else VAR_SPECS[canonical].units,
        "long_name": str(long_name) if long_name else VAR_SPECS[canonical].long_name,
    }
    if VAR_SPECS[canonical].standard_name:
        out.attrs["standard_name"] = VAR_SPECS[canonical].standard_name
    return out


def _check_time_consistency(arrays: Mapping[str, xr.DataArray]) -> dict[str, Any]:
    times = {name: np.asarray(da["time"].values) for name, da in arrays.items()}
    reference_name = next(iter(times))
    reference = times[reference_name]
    inconsistent = []
    for name, values in times.items():
        if values.shape != reference.shape or not np.array_equal(values, reference):
            inconsistent.append(name)
    if inconsistent:
        raise IngestError(
            "source files disagree on the time axis: "
            f"{inconsistent} differ from {reference_name}; refusing to merge silently"
        )
    return {
        "consistent": True,
        "reference": reference_name,
        "steps": int(reference.size),
        "start": str(np.datetime_as_string(reference.min(), unit="s")),
        "end": str(np.datetime_as_string(reference.max(), unit="s")),
    }


def _source_fingerprints(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    fingerprints: dict[str, dict[str, Any]] = {}
    for path in paths:
        size = path.stat().st_size
        entry = file_fingerprint(path, with_sha256=size <= SHA256_SIZE_LIMIT)
        if "sha256" not in entry:
            entry["sha256"] = None
            entry["sha256_note"] = (
                f"file larger than {SHA256_SIZE_LIMIT // (1024 * 1024)} MiB; "
                "size+mtime fingerprint used instead of a full-file digest"
            )
        fingerprints[str(path)] = entry
    return fingerprints


# --------------------------------------------------------------------------- #
# Ingest entry points
# --------------------------------------------------------------------------- #
def ingest(
    config: IngestConfigFile,
    *,
    config_path: str | Path | None = None,
    overwrite: bool = False,
) -> IngestReport:
    """Run an ingest configuration and return its report."""

    kind = config.source.kind
    if kind == "local_era5":
        return _ingest_local_era5(config, config_path=config_path, overwrite=overwrite)
    if kind == "local_netcdf":
        return _ingest_local_netcdf(config, config_path=config_path, overwrite=overwrite)
    raise IngestError(
        f"source.kind={kind!r} is not implemented in Phase 1; "
        "use local_era5 (local archive) or local_netcdf (existing sample); "
        "CDS download is a Phase 1 time-boxed follow-up"
    )


def _check_vertical_type(ds: xr.Dataset, declared: str) -> None:
    """Reject a pressure/model-level mismatch between config and source data."""

    if "level" not in ds.coords:
        return
    units = str(ds["level"].attrs.get("units", "")).lower()
    pressure_like = units in {"hpa", "mb", "millibars", "millibar", "pa", "hpaa"} or units.startswith("hpa")
    if declared == "pressure_level" and not pressure_like:
        raise IngestError(
            f"source.vertical=pressure_level 但数据 level 单位为 {units!r}；"
            "请确认是气压层还是模式层"
        )
    if declared == "model_level" and pressure_like:
        raise IngestError(
            f"source.vertical=model_level 但数据 level 单位为 {units!r}（看起来是气压层）"
        )


def _check_years(root: Path, years: list[int] | None, date_range: tuple[str, str]) -> None:
    """Verify ``source.years`` is consistent with the archive and the date range."""

    if not years:
        return
    for year in years:
        if not (root / str(year)).is_dir():
            raise IngestError(f"source.years 声明了 {year}，但 {root / str(year)} 不存在")
    start_year = date.fromisoformat(date_range[0]).year
    end_year = date.fromisoformat(date_range[1]).year
    missing = [year for year in range(start_year, end_year + 1) if year not in years]
    if missing:
        raise IngestError(
            f"source.date_range 覆盖年份 {missing} 不在 source.years={years} 中；"
            "删除 years（自动按 date_range 推导）或补齐年份"
        )


def _run_context(
    config: IngestConfigFile, *, config_path: str | Path | None
) -> tuple[Path, ResourceTracker, int]:
    seed = int(config.project.seed)
    set_global_seed(seed)
    output_dir = config.project.output_dir or Path("./output")
    run_dir = create_run_dir(
        output_dir, kind="ingest", model=config.source.kind, seed=seed
    )
    if config_path is not None:
        snapshot_config(config_path, run_dir)
    write_environment_snapshot(run_dir)
    tracker = ResourceTracker(label="ingest").start()
    return run_dir, tracker, seed


def _finalize_report(
    report: IngestReport,
    *,
    config: IngestConfigFile,
    config_path: str | Path | None,
    tracker: ResourceTracker,
    run_dir: Path,
) -> IngestReport:
    report.resources = tracker.stop()
    report.run_dir = run_dir
    report.notes.append("run 目录含 config.yaml 快照与 environment.json 环境快照（Test E 依据）")
    report.report_json = run_dir / ("test_a_report.json" if report.kind == "local_era5" else "ingest_report.json")
    write_json(report.report_json, report.to_dict())
    return report


def _ingest_local_era5(
    config: IngestConfigFile,
    *,
    config_path: str | Path | None,
    overwrite: bool,
) -> IngestReport:
    source = config.source
    subset = source.subset or SubsetConfig()
    if source.root is None or source.file_pattern is None or source.date_range is None:
        raise IngestError("local_era5 source requires root, file_pattern and date_range")

    root = resolve_path(source.root)
    if not root.is_dir():
        raise IngestError(f"ERA5 archive root is not a directory: {root}")
    days = _date_range(*source.date_range)
    _check_years(root, source.years, source.date_range)
    variables = [name for name in (source.variable_map or {}) if name in known_variables()]
    variables = variables or [name for name in DEFAULT_VARIABLES if name in known_variables()]
    level_targets = subset.levels
    derive_tcwv_flag = bool(source.derive and source.derive.tcwv)

    run_dir, tracker, seed = _run_context(config, config_path=config_path)
    report = IngestReport(kind="local_era5")

    # 1. resolve source files and fingerprint them before any read
    file_paths: list[tuple[Path, str, str | None]] = []
    for day in days:
        for canonical in variables:
            mapped = (source.variable_map or {}).get(canonical)
            segment = mapped or canonical
            path = root / f"{day.year}" / source.file_pattern.format(date=day, variable=segment)
            if not path.exists():
                raise IngestError(
                    f"missing source file: {path}; check source.root / file_pattern / date_range"
                )
            file_paths.append((path, canonical, mapped))
    before = _source_fingerprints(path for path, _, _ in file_paths)

    # 2. read per day, verify the time axis, concatenate
    per_day: list[xr.Dataset] = []
    tcwv_parts: list[xr.DataArray] = []
    for day in days:
        day_arrays: dict[str, xr.DataArray] = {}
        for canonical in variables:
            mapped = (source.variable_map or {}).get(canonical)
            path = root / f"{day.year}" / source.file_pattern.format(date=day, variable=mapped or canonical)
            keep_all = derive_tcwv_flag and canonical == "q"
            day_arrays[canonical] = _prepare_source_file(
                path,
                canonical=canonical,
                mapped=mapped,
                subset=subset,
                levels=level_targets,
                keep_all_levels=keep_all,
            )
        _check_time_consistency(day_arrays)
        if derive_tcwv_flag and "q" in day_arrays:
            q_all = day_arrays["q"]
            tcwv = derive_tcwv(q_all)
            tcwv_parts.append(tcwv)
            day_arrays["q"] = _select_levels(q_all, level_targets)
        per_day.append(xr.merge(list(day_arrays.values())))

    raw_ds = xr.concat(per_day, dim="time")
    if raw_ds.sizes.get("time", 0) == 0:
        raise IngestError("no time steps were selected; check subset.time_step_hours")

    processed_ds = raw_ds
    derived_info: dict[str, Any] = {}
    if tcwv_parts:
        tcwv_all = xr.concat(tcwv_parts, dim="time")
        processed_ds = xr.merge([raw_ds, tcwv_all])
        finite = np.asarray(tcwv_all.values)
        finite = finite[np.isfinite(finite)]
        derived_info["tcwv"] = {
            "formula": "tcwv = (1/g) * integral(q dp) over native pressure levels (trapezoidal)",
            "kind": "derived",
            "finite": bool(finite.size == np.asarray(tcwv_all.values).size),
            "min": float(finite.min()) if finite.size else float("nan"),
            "max": float(finite.max()) if finite.size else float("nan"),
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "negative_count": int((finite < 0).sum()) if finite.size else 0,
            "gravity": GRAVITY,
        }

    _check_vertical_type(processed_ds, source.vertical)

    # 3. contracts + writes
    source_meta: dict[str, Any] = {
        "kind": "real",
        "dataset": "local_era5_archive",
        "uri": str(root),
        "variable_map": {name: (source.variable_map or {}).get(name, name) for name in variables},
        "access_date": date.today().isoformat(),
        "subset": {
            "time": {
                "date_range": list(source.date_range),
                "step_hours": subset.time_step_hours,
            },
            "level": sorted(level_targets) if level_targets else None,
            "lat": list(subset.latitude) if subset.latitude else None,
            "lon": list(subset.longitude) if subset.longitude else None,
        },
        "readonly": True,
        "license": "Copernicus Climate Change Service (C3S) licence",
    }
    if derived_info:
        source_meta["derivation"] = {
            "tcwv": "tcwv = (1/g) * integral(q dp) over 1..1000 hPa (37 levels); derived, not ERA5 official tcwv"
        }

    generation_config = {
        "config_path": str(config_path) if config_path else None,
        "config_sha256": config_sha256(config_path) if config_path else None,
        "samples": None,
        "training": None,
    }
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "env_lock": "research/env/logs/atmosgen_pip_freeze.txt",
        "notes": "Phase 1 ingest: local ERA5 archive slice, read-only source",
    }

    raw_source_meta = dict(source_meta)
    raw_source_meta.pop("derivation", None)
    raw_contract = build_contract(
        raw_ds,
        generation_method="ingest",
        source=raw_source_meta,
        random_seed=seed,
        initial_conditions={"kind": "era5_slice", "details": {"description": "normalised slice of the local ERA5 archive"}},
        generation_config=generation_config,
        metadata=metadata,
        level_type=source.vertical,
    )
    if config.output.raw_path is not None:
        raw_target = assert_writable_target(config.output.raw_path, allow_raw=True)
        save_canonical(
            raw_ds,
            raw_target,
            contract=raw_contract,
            format="netcdf",
            allow_raw=True,
            overwrite=overwrite,
        )
        report.raw_path = raw_target

    processed_contract = build_contract(
        processed_ds,
        generation_method="ingest",
        source=source_meta,
        random_seed=seed,
        initial_conditions={"kind": "era5_slice", "details": {"description": "normalised slice of the local ERA5 archive"}},
        generation_config=generation_config,
        metadata=metadata,
        level_type=source.vertical,
    )
    processed_target = assert_writable_target(config.output.processed_path)
    save_canonical(
        processed_ds,
        processed_target,
        contract=processed_contract,
        format=config.output.format,
        overwrite=overwrite,
    )
    report.processed_path = processed_target

    # 4. file checksums + contract sidecar with the final file digest
    processed_sha = sha256_file(processed_target)
    processed_contract.setdefault("source", {})["sha256"] = processed_sha
    validate_contract(processed_contract, processed_ds)
    write_contract_file(sidecar_contract_path(processed_target), processed_contract)
    report.contract_sha256 = processed_sha

    # 5. read back to prove the file is usable
    readback = open_source(processed_target, chunks=None).load()
    try:
        readback_problems = _readback_problems(processed_ds, readback)
    finally:
        readback.close()

    # 6. read-only assertion: fingerprints must be unchanged
    after = _source_fingerprints(path for path, _, _ in file_paths)
    unchanged: list[bool] = []
    for path, _canonical, _mapped in file_paths:
        key = str(path)
        prev, now = before[key], after[key]
        same = prev["size_bytes"] == now["size_bytes"] and prev["mtime_ns"] == now["mtime_ns"]
        if prev.get("sha256") is not None:
            same = same and prev["sha256"] == now["sha256"]
        unchanged.append(same)
    readonly_ok = all(unchanged)
    enforce_readonly = bool(config.output.read_only_assertion)
    report.readonly = {
        "enforced": enforce_readonly,
        "all_unchanged": readonly_ok,
        "files_checked": len(file_paths),
        "sha256_files": sum(1 for entry in before.values() if entry.get("sha256")),
        "sha256_skipped": [
            {"path": path, "reason": entry.get("sha256_note")}
            for path, entry in before.items()
            if entry.get("sha256") is None
        ],
    }

    # 7. report content
    stats = dataset_statistics(processed_ds)
    report.variables = {
        name: {
            "dims": info["dims"],
            "units": info["units"],
            "long_name": info["long_name"],
            "min": info["min"],
            "max": info["max"],
            "mean": info["mean"],
            "std": info["std"],
            "size": info["size"],
        }
        for name, info in stats.items()
    }
    report.missing = {
        name: {
            "count": info["nan_count"],
            "total": info["size"],
            "ratio_percent": info["nan_ratio_percent"],
        }
        for name, info in stats.items()
    }
    report.coordinates = _coordinates_block(processed_ds)
    time_values = np.asarray(processed_ds["time"].values)
    report.time = {
        "start": str(np.datetime_as_string(time_values.min(), unit="s")),
        "end": str(np.datetime_as_string(time_values.max(), unit="s")),
        "steps": int(time_values.size),
        "step_hours": _step_hours(time_values),
    }
    report.derived = derived_info
    report.sources = [
        {
            "path": str(path),
            "size_bytes": before[str(path)]["size_bytes"],
            "size_mb": round(before[str(path)]["size_bytes"] / (1024**2), 2),
            "sha256": before[str(path)].get("sha256"),
            "unchanged": unchanged[index],
        }
        for index, (path, _canonical, _mapped) in enumerate(file_paths)
    ]
    report.outputs = {}
    for key, path in (("raw", report.raw_path), ("processed", report.processed_path)):
        if path is None:
            continue
        report.outputs[key] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "size_mb": round(path.stat().st_size / (1024**2), 3),
            "sha256": sha256_file(path),
        }

    report.checks = {
        "source_files_found": "PASS",
        "time_axis_consistency": "PASS",
        "readonly_sources_unchanged": (
            "PASS" if readonly_ok else "FAIL"
        )
        if enforce_readonly
        else "NOT_RUN",
        "contract_valid": "PASS",
        "roundtrip_readback": "PASS" if not readback_problems else "FAIL",
        "no_missing_values": "PASS"
        if all(info["nan_count"] == 0 for info in stats.values())
        else "CONDITIONAL_PASS",
        "derived_tcwv_valid": "NOT_RUN"
        if not derived_info
        else ("PASS" if derived_info["tcwv"]["finite"] and derived_info["tcwv"]["negative_count"] == 0 else "FAIL"),
    }
    if readback_problems:
        report.notes.extend(readback_problems)
    report.notes.append(
        "tcwv 为 q 垂直积分派生量（derived），不得在报告中当作 ERA5 官方 tcwv"
    )
    report.notes.append(
        "嵌入式契约不含 source.sha256（文件校验和写进文件自身是自引用、不可成立）；"
        "最终文件校验和写入侧车契约与 test_a_report.json"
    )
    report.status = _overall_status(report.checks)
    return _finalize_report(report, config=config, config_path=config_path, tracker=tracker, run_dir=run_dir)


def _readback_problems(expected: xr.Dataset, readback: xr.Dataset) -> list[str]:
    problems: list[str] = []
    missing = [name for name in expected.data_vars if name not in readback.data_vars]
    if missing:
        problems.append(f"readback is missing variables {missing}")
    for name in expected.data_vars:
        if name not in readback.data_vars:
            continue
        if tuple(expected[name].dims) != tuple(readback[name].dims):
            problems.append(
                f"readback dims mismatch for {name}: {tuple(expected[name].dims)} vs {tuple(readback[name].dims)}"
            )
        elif not np.array_equal(
            np.asarray(expected[name].values, dtype="float64"),
            np.asarray(readback[name].values, dtype="float64"),
            equal_nan=True,
        ):
            problems.append(f"readback values differ for {name}")
    return problems


def _ingest_local_netcdf(
    config: IngestConfigFile,
    *,
    config_path: str | Path | None,
    overwrite: bool,
) -> IngestReport:
    source = config.source
    if source.input is None:
        raise IngestError("local_netcdf source requires input")
    input_path = resolve_path(source.input)
    if not input_path.exists():
        raise IngestError(f"source file not found: {input_path}")

    subset = source.subset or SubsetConfig()
    run_dir, tracker, seed = _run_context(config, config_path=config_path)
    report = IngestReport(kind="local_netcdf")

    before = _source_fingerprints([input_path])
    ds = open_source(input_path, chunks=None, decode_times=True)
    try:
        rename_map: dict[str, str] = {}
        if "valid_time" in ds.coords or "valid_time" in ds.dims:
            rename_map["valid_time"] = "time"
        if "pressure_level" in ds.coords or "pressure_level" in ds.dims:
            rename_map["pressure_level"] = "level"
        if "model_level" in ds.coords or "model_level" in ds.dims:
            # model-level ERA5 files use a ``model_level`` dimension; the
            # AtmosGen contract always names the vertical axis ``level``.
            rename_map["model_level"] = "level"
        selected: dict[str, xr.DataArray] = {}
        for canonical, mapped in (source.variable_map or {}).items():
            if canonical not in known_variables():
                raise IngestError(f"variable_map contains unknown canonical name {canonical!r}")
            if mapped not in ds.data_vars:
                raise IngestError(
                    f"variable {mapped!r} not present in {input_path}; available: {list(ds.data_vars)}"
                )
            data = ds[mapped]
            if rename_map:
                data = data.rename({old: new for old, new in rename_map.items() if old in data.dims or old in data.coords})
            data = _select_times(data, subset)
            data = _select_horizontal(data, subset)
            data = _select_levels(data, subset.levels)
            data = data.load().rename(canonical)
            data.attrs = {
                "units": str(ds[mapped].attrs.get("units", VAR_SPECS[canonical].units)),
                "long_name": str(ds[mapped].attrs.get("long_name", VAR_SPECS[canonical].long_name)),
            }
            selected[canonical] = data
        if not selected:
            raise IngestError("variable_map is empty; nothing to ingest")
        processed_ds = xr.merge(list(selected.values()))
        # drop scalar helper coordinates that are not part of the contract
        for name in list(processed_ds.coords):
            if name not in {"time", "level", "latitude", "longitude"}:
                processed_ds = processed_ds.drop_vars(name)
        _check_vertical_type(processed_ds, source.vertical)
    finally:
        ds.close()

    processed_contract = build_contract(
        processed_ds,
        generation_method="ingest",
        source={
            "kind": "real",
            "dataset": "cds_era5_sample",
            "uri": str(input_path),
            "variable_map": dict(source.variable_map),
            "access_date": date.today().isoformat(),
            "readonly": True,
            "license": "Copernicus Climate Change Service (C3S) licence",
        },
        random_seed=seed,
        level_type=source.vertical,
        generation_config={
            "config_path": str(config_path) if config_path else None,
            "config_sha256": config_sha256(config_path) if config_path else None,
        },
        metadata={
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "notes": "Phase 1 ingest: CDS ERA5 pressure-level sample (cloud water)",
        },
    )
    processed_target = assert_writable_target(config.output.processed_path)
    save_canonical(
        processed_ds,
        processed_target,
        contract=processed_contract,
        format=config.output.format,
        overwrite=overwrite,
    )
    report.processed_path = processed_target

    processed_sha = sha256_file(processed_target)
    processed_contract.setdefault("source", {})["sha256"] = processed_sha
    validate_contract(processed_contract, processed_ds)
    write_contract_file(sidecar_contract_path(processed_target), processed_contract)
    report.contract_sha256 = processed_sha

    readback = open_source(processed_target, chunks=None).load()
    try:
        readback_problems = _readback_problems(processed_ds, readback)
    finally:
        readback.close()

    after = _source_fingerprints([input_path])
    prev, now = before[str(input_path)], after[str(input_path)]
    readonly_ok = prev["size_bytes"] == now["size_bytes"] and prev["mtime_ns"] == now["mtime_ns"]
    if prev.get("sha256") is not None:
        readonly_ok = readonly_ok and prev["sha256"] == now["sha256"]
    enforce_readonly = bool(config.output.read_only_assertion)

    stats = dataset_statistics(processed_ds)
    report.variables = {
        name: {
            "dims": info["dims"],
            "units": info["units"],
            "long_name": info["long_name"],
            "min": info["min"],
            "max": info["max"],
            "mean": info["mean"],
            "std": info["std"],
            "size": info["size"],
        }
        for name, info in stats.items()
    }
    report.missing = {
        name: {"count": info["nan_count"], "total": info["size"], "ratio_percent": info["nan_ratio_percent"]}
        for name, info in stats.items()
    }
    report.coordinates = _coordinates_block(processed_ds)
    time_values = np.asarray(processed_ds["time"].values)
    report.time = {
        "start": str(np.datetime_as_string(time_values.min(), unit="s")),
        "end": str(np.datetime_as_string(time_values.max(), unit="s")),
        "steps": int(time_values.size),
        "step_hours": _step_hours(time_values),
    }
    report.sources = [
        {
            "path": str(input_path),
            "size_bytes": prev["size_bytes"],
            "size_mb": round(prev["size_bytes"] / (1024**2), 3),
            "sha256": prev.get("sha256"),
            "unchanged": readonly_ok,
        }
    ]
    report.outputs = {
        "processed": {
            "path": str(processed_target),
            "size_bytes": processed_target.stat().st_size,
            "size_mb": round(processed_target.stat().st_size / (1024**2), 3),
            "sha256": sha256_file(processed_target),
        }
    }
    report.checks = {
        "source_files_found": "PASS",
        "readonly_sources_unchanged": (
            "PASS" if readonly_ok else "FAIL"
        )
        if enforce_readonly
        else "NOT_RUN",
        "contract_valid": "PASS",
        "roundtrip_readback": "PASS" if not readback_problems else "FAIL",
        "no_missing_values": "PASS"
        if all(info["nan_count"] == 0 for info in stats.values())
        else "CONDITIONAL_PASS",
    }
    if readback_problems:
        report.notes.extend(readback_problems)
    report.notes.append(
        "crwc/cswc 仍为 NOT_RUN：需 CDS 时间盒下载（Phase 1 补充任务）"
    )
    report.status = _overall_status(report.checks)
    return _finalize_report(report, config=config, config_path=config_path, tracker=tracker, run_dir=run_dir)


__all__ = [
    "IngestError",
    "IngestReport",
    "dataset_statistics",
    "derive_tcwv",
    "ingest",
]
