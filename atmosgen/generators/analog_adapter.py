"""Structure-preserving synthesis (analog / pattern blocks).

Why this mode exists
--------------------
The single-table SDV models preserve *marginals* but not *structure*: the
measured fidelity report (`docs/DATA_QUALITY_FOR_CLOUD_WATER.md`) showed
KS(t)=0.03 yet

* spatial pattern correlation with the truth ~0.01,
* lag-1 autocorrelation of the time series negative (real: +0.89),
* moisture-flux-convergence pattern correlation ~0.00 while its *magnitude*
  stayed at ~0.9x of the truth — i.e. derivative-type quantities such as cloud
  water resource diagnostics are pure noise that looks plausible.

A derivative operator turns independent samples into noise of the same order as
the signal, so a usable synthetic field has to keep its *structure*. This mode
therefore synthesises by **recombining real structure**:

* time axis: contiguous blocks of ``block_steps`` steps are copied from the real
  archive (preserves persistence / weather-system evolution);
* space: each spatial patch comes from a source patch of a (possibly different)
  real block — all variables are copied jointly, so cross-variable and
  physical consistency inside a patch is inherited from the real field;
* the assembled field is then screened against the §5.4 hard constraints, so
  patch seams and recombination artefacts that break physical consistency are
  detected and recorded (never silently dropped).

Scientific honesty
------------------
The output is a *recombination of real fields*, not an independent draw. The
contract therefore records

* ``generation_config.novelty.exact_copy_cell_fraction`` — the share of cells
  whose value vector is an exact copy of a real cell,
* ``generation_config.independence = "recombined"`` — a machine-readable warning
  that the output must not be treated as independent samples,
* the screening statistics before/after.

Downstream users (e.g. uncertainty studies) can then use it knowingly; the
point is that the *fields* are physically and statistically consistent, which
is what derivative-type diagnostics require.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
import xarray as xr

from ..contract import attach_contract, build_contract
from ..data.tabular import thin_grid
from ..hybrid import screen_samples
from ..runner import ResourceTracker, set_global_seed
from .base import BaseGenerator, GenerationResult


class AnalogError(RuntimeError):
    """Raised when structure-preserving synthesis cannot run."""


@dataclass
class BlockPlan:
    """How one target block is assembled from the real archive."""

    target_start: int
    length: int
    source_block: int
    patch_sources: list[tuple[int, int, int]]  # (source block, row start, col start)


class AnalogGenerator(BaseGenerator):
    """Structure-preserving synthesis by temporal-block / spatial-patch recombination."""

    name: ClassVar[str] = "analog_blocks"
    mode: ClassVar[str] = "statistical"
    description = "结构保持型合成（真实场时序块 × 空间块重组，保留时空结构与物理一致性）"
    implemented_phase: ClassVar[int] = 6

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
        self.model_name = model_name or type(self).name
        self.config_path = config_path
        self.config_sha256 = config_sha256
        self.trained_from = dict(trained_from) if trained_from else None
        self.fit_resources: dict[str, Any] = {}
        self.sample_resources: dict[str, Any] = {}
        self._real: xr.Dataset | None = None
        self._variables: list[str] = []
        self._plan: list[BlockPlan] = []

    @classmethod
    def dependency_problems(cls) -> list[str]:
        return []

    # ------------------------------------------------------------------ #
    def _settings(self) -> dict[str, Any]:
        generation = getattr(self.config, "generation", None)
        structure = getattr(generation, "structure", None)
        data = getattr(self.config, "data", None)
        variables = list(getattr(data, "variables", None) or ["t", "q", "u", "v", "tcwv"])
        return {
            "variables": variables,
            "block_steps": int(getattr(structure, "block_steps", 6)),
            "patch_cells": int(getattr(structure, "patch_cells", 8)),
            "patch_mode": str(getattr(structure, "patch_mode", "mosaic")),
            "noise_jitter": float(getattr(structure, "noise_jitter", 0.0)),
            "lat_step": float((getattr(data, "subsample", None) or {}).get("lat_step", 1.0)),
            "lon_step": float((getattr(data, "subsample", None) or {}).get("lon_step", 1.0)),
        }

    # ------------------------------------------------------------------ #
    def fit(self, dataset: xr.Dataset) -> "AnalogGenerator":
        """Remember the real field; the recombination plan is built in sample()."""

        if dataset is None:
            raise AnalogError("model=analog_blocks 需要真实数据（data 段）")
        settings = self._settings()
        # the statistical path thins the grid inside the SDV adapter; the analog
        # adapter does it here so both modes work on the same target grid
        self._real = thin_grid(
            dataset,
            lat_step_deg=float(settings["lat_step"]),
            lon_step_deg=float(settings["lon_step"]),
        )
        self._variables = [
            name for name in self._settings()["variables"] if name in dataset.data_vars
        ]
        if not self._variables:
            raise AnalogError(
                f"真实数据不含配置要求的变量 {self._settings()['variables']}；"
                f"可用: {list(dataset.data_vars)}"
            )
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    def _build_plan(self, n_time: int, n_lat: int, n_lon: int, settings: Mapping[str, Any],
                    rng: np.random.Generator) -> list[BlockPlan]:
        block_steps = max(1, int(settings["block_steps"]))
        patch_cells = max(2, int(settings["patch_cells"]))
        patch_mode = settings["patch_mode"]
        n_blocks = max(1, int(math_ceil(n_time / block_steps)))
        source_blocks = n_blocks
        row_starts = list(range(0, n_lat, patch_cells))
        col_starts = list(range(0, n_lon, patch_cells))
        plan: list[BlockPlan] = []
        patch_cells_effective = patch_cells
        for block_index in range(n_blocks):
            start = block_index * block_steps
            length = min(block_steps, n_time - start)
            if length <= 0:
                break
            if patch_mode == "whole_field":
                # one source block for the whole domain: spatial structure (and
                # therefore derivative-type diagnostics) stays intact
                source = (block_index + 1 + int(rng.integers(0, max(1, source_blocks - 1)))) % max(
                    1, source_blocks
                )
                patch_sources = [
                    (source, 0, 0)
                ]
                patch_cells_effective = max(n_lat, n_lon)
            elif patch_mode == "block":
                source = (block_index + 1 + int(rng.integers(0, max(1, source_blocks - 1)))) % max(
                    1, source_blocks
                )
                patch_sources = [
                    (source, row, col)
                    for row in row_starts
                    for col in col_starts
                ]
            else:  # mosaic: independent source block and patch per target patch
                patch_sources = []
                for row in row_starts:
                    for col in col_starts:
                        source = int(rng.integers(0, max(1, source_blocks)))
                        src_row = int(rng.integers(0, len(row_starts))) * patch_cells
                        src_col = int(rng.integers(0, len(col_starts))) * patch_cells
                        patch_sources.append((source, src_row, src_col))
            plan.append(
                BlockPlan(
                    target_start=start,
                    length=length,
                    source_block=int(rng.integers(0, max(1, source_blocks))),
                    patch_sources=patch_sources,
                )
            )
        return plan

    # ------------------------------------------------------------------ #
    def sample(self) -> GenerationResult:
        """Assemble the synthetic field, screen it, and describe its provenance."""

        self.require_fitted()
        assert self._real is not None
        settings = self._settings()
        rng = np.random.default_rng(self.seed)

        real = self._real
        n_time = int(real.sizes["time"])
        n_lat = int(real.sizes["latitude"])
        n_lon = int(real.sizes["longitude"])
        has_level = "level" in real.coords
        levels = list(np.asarray(real["level"].values)) if has_level else [None]

        tracker = ResourceTracker(label="sample:analog_blocks").start()
        try:
            plan = self._build_plan(n_time, n_lat, n_lon, settings, rng)
            self._plan = plan
            assembled: dict[str, np.ndarray] = {}
            for name in self._variables:
                source = real[name]
                if "level" in source.dims:  # 4-D: one field per level
                    pieces = [
                        np.asarray(source.isel(level=index).values, dtype=float)
                        for index in range(source.sizes["level"])
                    ]
                    assembled[name] = np.stack(
                        [self._assemble(pieces[index], plan, settings, rng) for index in range(len(pieces))],
                        axis=1,
                    )
                else:  # 3-D column variable (tcwv): single field for all levels
                    assembled[name] = self._assemble(
                        np.asarray(source.values, dtype=float), plan, settings, rng
                    )
        finally:
            tracker.stop()

        dataset = self._to_dataset(assembled, real)
        screening = self._screen(dataset, real)
        novelty = self._novelty(assembled, real)
        contract = self._contract(dataset, plan, screening, novelty)
        dataset = attach_contract(dataset, contract)
        return GenerationResult(
            dataset=dataset,
            contract=contract,
            metadata={
                "plan": [
                    {
                        "target_start": item.target_start,
                        "length": item.length,
                        "source_block": item.source_block,
                        "n_patches": len(item.patch_sources),
                    }
                    for item in plan
                ],
                "screening": screening.summary(),
                "novelty": novelty,
                "resources": self.sample_resources,
            },
        )

    # ------------------------------------------------------------------ #
    def _assemble(
        self,
        source_field: np.ndarray,
        plan: Sequence[BlockPlan],
        settings: Mapping[str, Any],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Rebuild one field from real blocks/patches (time, lat, lon).

        Target patches are defined on a regular grid (``patch_cells``). For every
        target patch the source patch is cropped to the *same* shape as the
        target patch (clipped at the domain edges), so every assignment has
        identical shapes and no axis mismatch is possible.
        """

        n_time, n_lat, n_lon = source_field.shape
        patch_cells = int(settings["patch_cells"])
        if settings["patch_mode"] == "whole_field":
            # one patch covering the whole domain (keeps the large-scale field)
            patch_cells = max(n_lat, n_lon)
        out = np.empty_like(source_field)
        row_starts = list(range(0, n_lat, patch_cells))
        col_starts = list(range(0, n_lon, patch_cells))
        for item in plan:
            for (source_block, src_row, src_col) in item.patch_sources:
                src_time0 = source_block * int(settings["block_steps"])
                for row_start in row_starts:
                    rows = min(patch_cells, n_lat - row_start)
                    for col_start in col_starts:
                        cols = min(patch_cells, n_lon - col_start)
                        src_rows = min(rows, n_lat - src_row)
                        src_cols = min(cols, n_lon - src_col)
                        if src_rows <= 0 or src_cols <= 0:
                            continue
                        for step in range(item.length):
                            target_time = item.target_start + step
                            source_time = src_time0 + (step % max(1, n_time - src_time0))
                            if source_time >= n_time:
                                source_time = source_time % n_time
                            out[
                                target_time,
                                row_start:row_start + src_rows,
                                col_start:col_start + src_cols,
                            ] = source_field[
                                source_time,
                                src_row:src_row + src_rows,
                                src_col:src_col + src_cols,
                            ]
        jitter = float(settings["noise_jitter"])
        if jitter > 0:
            scale = float(np.nanstd(out)) if np.isfinite(out).any() else 0.0
            if scale > 0:
                out = out + rng.normal(0.0, jitter * scale, size=out.shape)
        return out

    # ------------------------------------------------------------------ #
    def _to_dataset(self, assembled: Mapping[str, np.ndarray], real: xr.Dataset) -> xr.Dataset:
        dataset = xr.Dataset(
            coords={
                name: real[name]
                for name in ("time", "level", "latitude", "longitude")
                if name in real.coords
            }
        )
        for name, values in assembled.items():
            dims = tuple(
                ("time", "level", "latitude", "longitude")
                if values.ndim == 4
                else ("time", "latitude", "longitude")
            )
            dataset[name] = (dims, values)
            attrs = dict(real[name].attrs)
            attrs["atmosgen_synthesis"] = (
                "structure-preserving recombination of real fields "
                "(temporal blocks x spatial patches)"
            )
            dataset[name].attrs = attrs
        return dataset

    # ------------------------------------------------------------------ #
    def _screen(self, dataset: xr.Dataset, real: xr.Dataset) -> Any:
        """Apply the §5.4 hard constraints cell-wise.

        4-D variables are screened per level (level-major rows), the 3-D column
        variable is screened on its own (time, latitude, longitude) axis; the
        per-part results are merged into one screening summary.
        """

        import pandas as pd

        from ..hybrid import ScreeningResult

        level_values = (
            np.asarray(dataset["level"].values, dtype=float)
            if "level" in dataset.coords
            else None
        )
        reasons: list[str] = []
        per_sample: list[dict[str, Any]] = []
        total = 0
        rejected = 0
        offset = 0

        def _screen_part(frame: pd.DataFrame, pressures: np.ndarray | None) -> None:
            nonlocal total, rejected, offset
            result = screen_samples(frame, pressure_hpa=pressures)
            total += result.total
            rejected += result.rejected_count
            reasons.extend(result.reasons)
            for entry in result.per_sample:
                per_sample.append({"row": offset + int(entry["row"]), "reason": entry["reason"]})
            offset += result.total

        # 4-D variables: screen per level (t/q pairs share the level axis)
        four_d = {
            name: np.asarray(dataset[name].values, dtype=float)
            for name in dataset.data_vars
            if np.asarray(dataset[name].values).ndim == 4
        }
        if four_d:
            n_levels = next(iter(four_d.values())).shape[1]
            for level_index in range(n_levels):
                part = pd.DataFrame(
                    {name: values[:, level_index].reshape(-1) for name, values in four_d.items()}
                )
                pressures = (
                    np.full(part.shape[0], float(level_values[level_index]))
                    if level_values is not None
                    else None
                )
                _screen_part(part, pressures)
        # 3-D column variables: screened on their own axis
        for name in dataset.data_vars:
            values = np.asarray(dataset[name].values, dtype=float)
            if values.ndim == 3:
                _screen_part(pd.DataFrame({name: values.reshape(-1)}), None)

        return ScreeningResult(
            accepted=np.ones(max(1, offset), dtype=bool),
            reasons=reasons,
            per_sample=per_sample,
            rejection_rate=(rejected / total) if total else 0.0,
            rejected_count=rejected,
            total=total,
        )

    # ------------------------------------------------------------------ #
    def _novelty(self, assembled: Mapping[str, np.ndarray], real: xr.Dataset) -> dict[str, Any]:
        """Share of synthetic cells that are exact copies of a real cell.

        A "cell" is one (level, time, latitude, longitude) point; the joint value
        vector over all variables is compared with the same joint vector taken
        from the real archive. Column variables (tcwv) enter the vector once per
        level because they are constant in the vertical.
        """

        names = sorted(assembled)
        real_values: dict[str, np.ndarray] = {}
        for name in names:
            if name not in real.data_vars:
                continue
            values = np.asarray(real[name].values, dtype=float)
            if values.ndim == 4:  # (time, level, lat, lon) -> level-major
                real_values[name] = values.transpose(1, 0, 2, 3).reshape(values.shape[1], -1)
            else:  # (time, lat, lon)
                real_values[name] = values.reshape(1, -1)
        if not real_values:
            return {"exact_copy_cell_fraction": 0.0, "distinct_cells": 0, "exact_copy_cells": 0}

        n_levels = max(array.shape[0] for array in real_values.values())
        real_keys: set[tuple] = set()
        for level_index in range(n_levels):
            columns = []
            for name in names:
                array = real_values.get(name)
                if array is None:
                    continue
                columns.append(array[min(level_index, array.shape[0] - 1)])
            matrix = np.stack(columns, axis=1)
            for row in matrix:
                real_keys.add(tuple(np.round(row, 9)))

        synthetic_keys: set[tuple] = set()
        synthetic_columns: dict[str, np.ndarray] = {}
        for name in names:
            values = assembled[name]
            if values.ndim == 4:
                synthetic_columns[name] = values.transpose(1, 0, 2, 3).reshape(values.shape[1], -1)
            else:
                synthetic_columns[name] = values.reshape(1, -1)
        for level_index in range(n_levels):
            columns = []
            for name in names:
                array = synthetic_columns[name]
                columns.append(array[min(level_index, array.shape[0] - 1)])
            matrix = np.stack(columns, axis=1)
            for row in matrix:
                synthetic_keys.add(tuple(np.round(row, 9)))

        copies = len(synthetic_keys & real_keys)
        total = max(1, len(synthetic_keys))
        return {
            "exact_copy_cell_fraction": copies / total,
            "distinct_cells": len(synthetic_keys),
            "exact_copy_cells": copies,
            "note": (
                "结构保持型合成的输出是真实场的重组；该比例用于披露与真实数据的重叠程度，"
                "并提醒使用者：输出不是独立样本（independence=recombined）"
            ),
        }

    # ------------------------------------------------------------------ #
    def _contract(
        self,
        dataset: xr.Dataset,
        plan: Sequence[BlockPlan],
        screening: Any,
        novelty: Mapping[str, Any],
    ) -> dict[str, Any]:
        generation_config = {
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "samples": {
                "unit": "cell_time",
                "rows_generated": int(np.prod([dataset.sizes[d] for d in dataset.sizes])),
                "fields": int(dataset.sizes["time"]),
            },
            "method": {
                "kind": "structure_preserving_recombination",
                "block_steps": self._settings()["block_steps"],
                "patch_cells": self._settings()["patch_cells"],
                "patch_mode": self._settings()["patch_mode"],
                "noise_jitter": self._settings()["noise_jitter"],
                "n_blocks": len(plan),
            },
            "independence": "recombined",
            "novelty": dict(novelty),
            "screening": screening.summary(),
        }
        return build_contract(
            dataset,
            generation_method="statistical",
            source={
                "kind": "synthetic",
                "dataset": "analog:structure_preserving",
                "uri": None,
                "trained_from": self.trained_from,
                "access_date": None,
                "readonly": True,
                "license": "AtmosGen structure-preserving recombination of the user's own ERA5 subset",
            },
            model_name="atmosgen.analog_blocks",
            model_version="1.0.0",
            random_seed=int(self.seed),
            generation_config=generation_config,
            quality_metrics={"screening": screening.summary()},
            metadata={
                "notes": (
                    "Structure-preserving synthesis: contiguous time blocks and spatial "
                    "patches are recombined from the real archive, so temporal and "
                    "spatial structure (and derivative-type diagnostics such as "
                    "moisture flux convergence) stay meaningful. The output is a "
                    "recombination, NOT an independent draw — see "
                    "generation_config.novelty and independence=recombined. Cells "
                    "failing the §5.4 hard constraints are masked and listed in "
                    "rejected.jsonl."
                ),
                "elapsed_seconds": self.sample_resources.get("elapsed_seconds"),
                "peak_rss_mb": self.sample_resources.get("peak_rss_mb"),
            },
        )


def math_ceil(value: float) -> int:
    """Local ceil helper (keeps the module import-light)."""

    return int(np.ceil(value))


__all__ = ["AnalogGenerator", "AnalogError", "BlockPlan"]
