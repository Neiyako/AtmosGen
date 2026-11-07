"""Hybrid generator adapter (Mode 3): physics base + conditional SDV + screening.

Design (plan §4.3)
------------------
* **Physics base** — the advection–diffusion solver evolves a tracer that
  starts from a real field (``from_real``), i.e. the column-integrated water
  vapour ``tcwv`` transported by the solver. Its values are the physical
  constraint the statistical part has to respect.
* **Conditional generation** — the SDV model is fitted per pressure level
  (as in Mode 1). For every level the *physics base* is fed as the known column
  through ``sample_remaining_columns``, so ``tcwv`` comes back bit-identical
  and the other variables (``t``, ``q``, ``u``, ``v``) are drawn from the
  conditional distribution given that base value.
* **Screening** — every cell is checked against the hard constraints of plan
  §5.4 (finite, non-negative, ranges, saturation). Rejected cells are written
  to ``rejected.jsonl`` with their reasons and are masked out of the dataset
  (NaN) so the delivered product contains only physically acceptable cells;
  the rejection rate and the *before/after* screening metrics are recorded in
  the contract. Rejection is never silent (plan §4.3 step ⑤).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from ..contract import attach_contract, build_contract
from ..hybrid import (
    HybridError,
    conditional_generate,
    screen_samples,
)
from ..runner import ResourceTracker, set_global_seed
from .base import BaseGenerator, GenerationResult
from .physics.advection_diffusion import AdvectionDiffusionGenerator


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.datetime64):
        return str(value)
    return value


class HybridGenerator(BaseGenerator):
    """Mode 3: physics base + conditional statistical generation + screening."""

    name: ClassVar[str] = "hybrid"
    mode: ClassVar[str] = "hybrid"
    description = "物理基底（tcwv）+ SDV 条件生成 + 物理硬约束筛检"
    implemented_phase: ClassVar[int] = 4

    def __init__(
        self,
        config: Any = None,
        *,
        seed: int = 42,
        model_name: str | None = None,
        config_path: str | None = None,
        config_sha256: str | None = None,
        trained_from: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(config, seed=seed)
        set_global_seed(seed)
        self.config_path = config_path
        self.config_sha256 = config_sha256
        self.trained_from = dict(trained_from) if trained_from else None
        self._real: xr.Dataset | None = None
        self._physics: BaseGenerator | None = None
        self._statistical: BaseGenerator | None = None
        self.fit_resources: dict[str, Any] = {}
        self.sample_resources: dict[str, Any] = {}
        self.rejected_path: Path | None = None
        self._rejected_records: list[dict[str, Any]] = []
        self._coordinate_rows: dict[str, np.ndarray] = {}

    @classmethod
    def dependency_problems(cls) -> list[str]:
        problems = AdvectionDiffusionGenerator.dependency_problems()
        for module in ("sdv", "sdmetrics"):
            try:
                __import__(module)
            except Exception as exc:  # pragma: no cover
                problems.append(f"missing module {module!r}: {exc}")
        return problems

    # ------------------------------------------------------------------ #
    def fit(self, dataset: xr.Dataset | None = None) -> "HybridGenerator":
        """Fit the physics solver and the per-level statistical model."""

        hybrid = getattr(self.config, "hybrid", None)
        if hybrid is None:
            raise HybridError("mode=hybrid 需要 hybrid 段")
        if dataset is None:
            raise HybridError("mode=hybrid 需要真实数据训练统计模型（data 段）")
        self._real = dataset
        tracker = ResourceTracker(label="fit:hybrid").start()
        try:
            self._physics = AdvectionDiffusionGenerator(
                self.config,
                seed=self.seed,
                config_path=self.config_path,
                config_sha256=self.config_sha256,
                trained_from=self.trained_from,
            )
            self._physics.fit(dataset)

            from .sdv_adapter import GENERATOR_CLASSES

            model_name = hybrid.statistical_component.model
            statistical_cls = GENERATOR_CLASSES[model_name]
            self._statistical = statistical_cls(
                self.config,
                seed=self.seed,
                config_path=self.config_path,
                config_sha256=self.config_sha256,
                trained_from=self.trained_from,
            )
            self._statistical.fit(dataset)
        finally:
            self.fit_resources = tracker.stop()
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    def sample(self) -> GenerationResult:
        """Physics base -> conditional generation -> screening."""

        self.require_fitted()
        hybrid = getattr(self.config, "hybrid", None)
        assert self._physics is not None and self._statistical is not None

        tracker = ResourceTracker(label="sample:hybrid").start()
        try:
            physics_result = self._physics.sample()
            base = physics_result.dataset
            known_column = hybrid.physics_component.output_variable
            # the solver emits the passive tracer as `c`; for the hybrid mode the
            # base variable is the transported real field (tcwv by default)
            if known_column not in base.data_vars:
                tracer = "c" if "c" in base.data_vars else None
                if tracer is None:
                    raise HybridError(
                        f"hybrid.physics_component.output_variable={known_column!r} 不在物理场中；"
                        f"可用: {list(base.data_vars)}"
                    )
                base = base.rename({tracer: known_column})
                base[known_column].attrs.update(
                    {
                        "units": "kg m**-2",
                        "long_name": f"{known_column} transported by the advection-diffusion solver",
                        "atmosgen_derived": (
                            "physics base: real field advected/diffused by the solver "
                            "(initial condition kind=from_real)"
                        ),
                    }
                )
            base_values = np.asarray(base[known_column].values, dtype=float)
            if "level" in base[known_column].dims:  # (time, level, lat, lon)
                level_axis = base[known_column].dims.index("level")
                base_values = np.take(base_values, 0, axis=level_axis)
            base_flat = base_values.reshape(-1)

            level_models = getattr(self._statistical, "level_models", None)
            if not level_models:
                raise HybridError("统计模型未按层训练，无法做条件生成")
            # every level is conditioned on the same physics base (column quantity)
            per_level_rows: dict[Any, pd.DataFrame] = {}
            for level_value, synthesizer in level_models.items():
                per_level_rows[level_value] = conditional_generate(
                    synthesizer,
                    known_column=known_column,
                    known_values=base_flat,
                )
        finally:
            tracker.stop()
        return self._assemble(base, per_level_rows, known_column, physics_result)

    # ------------------------------------------------------------------ #
    def _assemble(
        self,
        base: xr.Dataset,
        per_level_rows: Mapping[Any, pd.DataFrame],
        known_column: str,
        physics_result: GenerationResult,
    ) -> GenerationResult:
        """Build the canonical hybrid dataset, then screen it cell by cell."""

        hybrid = getattr(self.config, "hybrid", None)
        levels = list(per_level_rows)
        n_time = int(base["time"].size)
        n_lat = int(base["latitude"].size)
        n_lon = int(base["longitude"].size)
        n_cells = n_time * n_lat * n_lon

        base_flat = np.asarray(base[known_column].values, dtype=float)
        if "level" in base[known_column].dims:
            level_axis = base[known_column].dims.index("level")
            base_flat = np.take(base_flat, 0, axis=level_axis)
        base_flat = base_flat.reshape(-1)

        variables = [
            name
            for name in next(iter(per_level_rows.values())).columns
            if name != known_column
        ]
        frames: list[pd.DataFrame] = []
        pressures: list[np.ndarray] = []
        for level_value in levels:
            frame = per_level_rows[level_value].copy()
            frame["__level__"] = level_value
            frames.append(frame)
            pressures.append(np.full(n_cells, float(level_value)))
        combined = pd.concat(frames, ignore_index=True)
        pressure_all = np.concatenate(pressures) if pressures else None

        screening = screen_samples(
            combined.drop(columns=["__level__"]), pressure_hpa=pressure_all
        )
        accepted = screening.accepted

        coords = {
            name: base[name]
            for name in ("latitude", "longitude")
            if name in base.coords
        }
        # the physics base is 2-D; the level dimension comes from the per-level
        # statistical models, so its coordinate must be built from their keys
        # (without it the saturation check cannot pair t/q with pressure)
        coords["level"] = np.asarray(levels, dtype=float)
        # Time axis: the solver advances with its own dt (10 min here), which is
        # far coarser/finer than the real 6-hourly cadence. For the four-way
        # comparison (plan §9 E3) the hybrid output is therefore labelled with
        # the FIRST N real timestamps — an index-style mapping that is stated
        # in the contract, never silently implied to be a physical clock.
        time_note = ""
        real = self._real
        n_saved = int(base["time"].size)
        if real is not None and "time" in real.coords:
            real_times = np.asarray(real["time"].values)
            if real_times.size >= n_saved:
                coords["time"] = real_times[:n_saved]
                time_note = (
                    "hybrid 输出的 time 坐标沿用真实数据前 "
                    f"{n_saved} 个时次标签（索引式映射）；物理演化步长为 "
                    "physics.time.dt_s，两者不构成物理时钟对应关系"
                )
        if "time" not in coords:
            coords["time"] = base["time"]
        dataset = xr.Dataset(coords=coords)
        for name in variables:
            # combined rows are level-major (level, time, lat, lon); the dataset
            # declares time-major dims, so transpose after the reshape
            values = np.asarray(combined[name].values, dtype=float).reshape(
                len(levels), n_time, n_lat, n_lon
            ).transpose(1, 0, 2, 3)
            dataset[name] = (("time", "level", "latitude", "longitude"), values)
        # the physics base is a COLUMN quantity: schema §2.1 keeps tcwv 3-D
        # (time, latitude, longitude) and forbids faking a 4-D copy
        dataset[known_column] = (
            ("time", "latitude", "longitude"),
            base_flat.reshape(n_time, n_lat, n_lon).astype(float),
        )

        # coordinates per flattened cell (row order: level, time, lat, lon)
        level_axis_values = np.asarray(combined["__level__"].values, dtype=float)
        self._coordinate_rows = {
            "time": np.repeat(np.asarray(base["time"].values), len(levels) * n_lat * n_lon),
            "level": np.repeat(level_axis_values, n_lat * n_lon),
            "latitude": np.tile(
                np.repeat(np.asarray(base["latitude"].values), n_lon), len(levels) * n_time
            ),
            "longitude": np.tile(
                np.asarray(base["longitude"].values), len(levels) * n_time * n_lat
            ),
        }
        reasons_by_row: dict[int, set[str]] = {}
        for entry in screening.per_sample:
            reasons_by_row.setdefault(int(entry["row"]), set()).add(str(entry["reason"]))
        self._rejected_records = [
            {
                "row": int(index),
                "reasons": sorted(reasons_by_row.get(int(index), set())),
                "values": {
                    name: float(combined[name].values[index])
                    for name in combined.columns
                    if name != "__level__"
                },
            }
            for index in np.where(~accepted)[0]
        ]

        mask = accepted.reshape(len(levels), n_time, n_lat, n_lon).transpose(1, 0, 2, 3)
        for name in ("t", "q", "u", "v"):
            if name not in dataset.data_vars:
                continue
            values = np.asarray(dataset[name].values, dtype=float).copy()
            values[~mask] = np.nan
            dataset[name] = (dataset[name].dims, values)
        # a column cell is only kept when every level passed the screening
        # (mask is time-major now, so the level axis is 1)
        cell_mask = mask.all(axis=1)
        if known_column in dataset.data_vars:
            values = np.asarray(dataset[known_column].values, dtype=float).copy()
            values[~cell_mask] = np.nan
            dataset[known_column] = (dataset[known_column].dims, values)

        before = screening.summary(mask_applied=True)
        after_frame = combined.drop(columns=["__level__"]).loc[accepted]
        after_screening = screen_samples(
            after_frame,
            pressure_hpa=pressure_all[accepted] if pressure_all is not None else None,
        )
        after = {
            **after_screening.summary(mask_applied=False),
            "note": "仅统计通过筛检的单元，用于对比修正前后的指标",
        }

        contract = build_contract(
            dataset,
            generation_method="hybrid",
            source={
                "kind": "synthetic",
                "dataset": "hybrid:physics+conditional_sdv",
                "uri": None,
                "trained_from": self.trained_from,
                "access_date": None,
                "readonly": True,
                "license": "AtmosGen hybrid output (physics base + SDV conditional sampling)",
            },
            model_name=(
                f"hybrid.{hybrid.statistical_component.model}"
                "+physics.advection_diffusion"
            ),
            model_version=getattr(self._statistical, "sdv_version", None),
            random_seed=int(self.seed),
            initial_conditions=physics_result.contract.get("initial_conditions"),
            boundary_conditions=physics_result.contract.get("boundary_conditions"),
            physical_parameters=physics_result.contract.get("physical_parameters"),
            generation_config={
                "config_path": self.config_path,
                "config_sha256": self.config_sha256,
                "samples": {
                    "unit": "cell_time",
                    "rows_generated": int(screening.total),
                    "rows_accepted": int(accepted.sum()),
                    "fields": int(base["time"].size),
                    "physics_base": known_column,
                    "conditioning": "sample_remaining_columns",
                },
                "training": {
                    "statistical_model": hybrid.statistical_component.model,
                    "trained_on": list(hybrid.statistical_component.trained_on),
                },
            },
            quality_metrics={
                "screening_before": before,
                "screening_after": after,
                # machine-readable mask declaration: physical validation reads
                # this to allow the documented NaN cells (plan §4.3 step ⑤)
                "screening": before,
            },
            metadata={
                "notes": (
                    "Mode 3 hybrid: the physics base is preserved exactly (known column "
                    "comes back bit-identical), the other variables are conditionally "
                    "sampled, and cells failing the §5.4 hard constraints are masked "
                    "and listed in rejected.jsonl (never silently dropped)."
                    + (f" {time_note}" if time_note else "")
                ),
            },
        )
        dataset = attach_contract(dataset, contract)
        return GenerationResult(
            dataset=dataset,
            contract=contract,
            metadata={
                "screening_before": before,
                "screening_after": after,
                "rejected_records": len(self._rejected_records),
                "physics_diagnostics": physics_result.metadata.get("diagnostics"),
            },
        )

    # ------------------------------------------------------------------ #
    def write_rejected(self, path: str | Path) -> Path:
        """Write the rejected-cell log (one JSON line per rejected cell)."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for record in self._rejected_records:
                row = int(record["row"])
                payload = {
                    "row": row,
                    "reasons": record["reasons"],
                    "values": record["values"],
                }
                for key, values in self._coordinate_rows.items():
                    if row < len(values):
                        payload[key] = _jsonable(values[row])
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.rejected_path = target
        return target


__all__ = ["HybridGenerator"]
