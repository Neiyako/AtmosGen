"""SDV statistical generator adapters (Mode 1, plan §4.1).

Three models share one adapter: ``gaussian_copula`` (copula), ``ctgan`` (GAN)
and ``tvae`` (VAE).

Phase 2 training scheme: **one model per pressure level** (``level_wise``).
Each level's joint distribution is fitted on that level's cells only, so
per-level marginals and variable correlations are preserved. Between-level,
temporal and spatial structure is *not* modelled — the validation modules
measure that and the run report says so; nothing is silently assumed.

Why not one model with covariates? Measured in this environment (SDV 2.1.0):
conditioning the Gaussian copula on continuous covariates is exact-interval
inversion and degenerates at the empirical boundary (level=500 conditioning
collapsed ``t`` to 248.0 K ± 0.02 K, and sampling took 148 s for 32 448 rows);
CTGAN/TVAE condition by rejection sampling (1 accepted row per 100 tries on 4
continuous covariates, ~6 s/row). Both are unsuitable for full grids, so the
level-wise scheme is used and recorded in the contract.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from ..contract import attach_contract, build_contract
from ..data.tabular import build_grid, rows_to_fields, thin_grid
from ..runner import ResourceTracker, set_global_seed
from .base import BaseGenerator, GenerationResult

TABLE_NAME = "atmosgen"

MODEL_CLASSES: dict[str, str] = {
    "gaussian_copula": "GaussianCopulaSynthesizer",
    "ctgan": "CTGANSynthesizer",
    "tvae": "TVAESynthesizer",
}

DEFAULT_TRAINING: dict[str, Any] = {"epochs": 100, "batch_size": 500}

#: Training scheme recorded in the contract (see module docstring).
TRAINING_SCHEME = "level_wise"


def _sdv_factories() -> dict[str, Any]:
    from sdv.metadata import Metadata
    from sdv.single_table import (
        CTGANSynthesizer,
        GaussianCopulaSynthesizer,
        TVAESynthesizer,
    )

    return {
        "metadata_cls": Metadata,
        "synthesizers": {
            "gaussian_copula": GaussianCopulaSynthesizer,
            "ctgan": CTGANSynthesizer,
            "tvae": TVAESynthesizer,
        },
    }


def _filter_signature(model: Any, kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Split kwargs into (accepted, ignored) using the introspected signature."""

    try:
        signature = inspect.signature(model)
    except (TypeError, ValueError):  # pragma: no cover - C-built signatures
        return dict(kwargs), []
    has_var_keyword = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )
    accepted: dict[str, Any] = {}
    ignored: list[str] = []
    for key, value in kwargs.items():
        if key in signature.parameters or has_var_keyword:
            accepted[key] = value
        else:
            ignored.append(key)
    return accepted, ignored


def _numerical_metadata(metadata_cls: Any, df: pd.DataFrame) -> Any:
    """All features are float columns of one table (sdv 2.x unified Metadata)."""

    metadata = metadata_cls()
    metadata.add_table(TABLE_NAME)
    for column in df.columns:
        metadata.add_column(column_name=column, sdtype="numerical", table_name=TABLE_NAME)
    metadata.validate()
    return metadata


def _unwrap_sampled(sampled: Any) -> pd.DataFrame:
    """sdv 2.x returns ``{table: DataFrame}``; sdv 1.x returned a DataFrame."""

    if isinstance(sampled, Mapping):
        if TABLE_NAME not in sampled:
            raise RuntimeError(f"SDV sample is missing table {TABLE_NAME!r}: {list(sampled)}")
        frame = sampled[TABLE_NAME]
    else:
        frame = sampled
    if not isinstance(frame, pd.DataFrame):
        raise RuntimeError(
            f"SDV sample returned {type(frame).__name__}, expected pandas.DataFrame"
        )
    return frame


class SDVGenerator(BaseGenerator):
    """Fit/sample/save for the three SDV single-table models (level-wise)."""

    name: ClassVar[str] = ""
    mode: ClassVar[str] = "statistical"
    description: ClassVar[str] = "SDV statistical generator"
    implemented_phase: ClassVar[int] = 2

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
        resolved = model_name or getattr(type(self), "name", None) or "gaussian_copula"
        if resolved not in MODEL_CLASSES:
            raise ValueError(
                f"unknown SDV model {resolved!r}; expected one of {sorted(MODEL_CLASSES)}"
            )
        self.model_name = resolved
        self.config_path = config_path
        self.config_sha256 = config_sha256
        self.trained_from = dict(trained_from) if trained_from else None
        self.created_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        training_block = (
            getattr(getattr(config, "generation", None), "training", None) if config else None
        )
        self.training: dict[str, Any] = {
            "epochs": int(getattr(training_block, "epochs", 100)) if training_block else 100,
            "batch_size": int(getattr(training_block, "batch_size", 500)) if training_block else 500,
            "discriminator_steps": int(getattr(training_block, "discriminator_steps", 1))
            if training_block
            else 1,
        }
        self.fit_resources: dict[str, Any] = {}
        self.sample_resources: dict[str, Any] = {}
        self.ignored_hyperparameters: list[str] = []
        self.adjusted_hyperparameters: dict[str, Any] = {}
        self._level_models: dict[Any, Any] = {}
        self._grid: Mapping[str, np.ndarray] | None = None
        self._variables: list[str] = []
        self._covariates: list[str] = []
        try:
            import sdv

            self.sdv_version = str(getattr(sdv, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover - availability checked earlier
            raise RuntimeError(f"sdv is not importable: {type(exc).__name__}: {exc}") from exc

    # ------------------------------------------------------------------ #
    @classmethod
    def dependency_problems(cls) -> list[str]:
        problems: list[str] = []
        for module in ("sdv", "sdmetrics", "torch"):
            try:
                __import__(module)
            except Exception as exc:  # pragma: no cover - environment dependent
                problems.append(f"missing module {module!r}: {type(exc).__name__}: {exc}")
        return problems

    # ------------------------------------------------------------------ #
    def _data_settings(self) -> dict[str, Any]:
        data = getattr(self.config, "data", None)
        variables = list(getattr(data, "variables", None) or ["t", "q", "u", "v", "tcwv"])
        covariates = list(
            getattr(getattr(self.config, "generation", None), "covariates", None) or []
        )
        subsample = dict(getattr(data, "subsample", None) or {})
        return {
            "variables": variables,
            "covariates": covariates,
            "lat_step": float(subsample.get("lat_step", 1.0)),
            "lon_step": float(subsample.get("lon_step", 1.0)),
        }

    def fit(self, dataset: xr.Dataset) -> "SDVGenerator":
        """Fit one synthesizer per pressure level (see module docstring)."""

        settings = self._data_settings()
        self._variables = list(settings["variables"])
        self._covariates = list(settings["covariates"])
        tracker = ResourceTracker(label=f"fit:{self.model_name}").start()
        try:
            learned_ds = thin_grid(
                dataset, lat_step_deg=settings["lat_step"], lon_step_deg=settings["lon_step"]
            )
            self._grid = build_grid(learned_ds)
            factories = _sdv_factories()
            synthesizer_cls = factories["synthesizers"][self.model_name]
            levels = self._grid.get("level")
            level_values: Sequence[Any] = (
                list(np.asarray(levels).tolist()) if levels is not None else [None]
            )
            for level_value in level_values:
                subset = (
                    learned_ds.sel(level=level_value) if level_value is not None else learned_ds
                )
                values = {
                    name: np.asarray(subset[name].values, dtype=float).reshape(-1)
                    for name in self._variables
                }
                df = pd.DataFrame(values, columns=self._variables)
                df = df.replace([np.inf, -np.inf], np.nan).dropna()
                metadata = _numerical_metadata(factories["metadata_cls"], df)
                hyper: dict[str, Any] = {"enforce_min_max_values": True}
                if self.model_name in {"ctgan", "tvae"}:
                    hyper.update(
                        epochs=int(self.training["epochs"]),
                        batch_size=int(self.training["batch_size"]),
                    )
                    if self.model_name == "ctgan":
                        hyper["discriminator_steps"] = int(self.training["discriminator_steps"])
                if self.model_name == "ctgan":
                    requested_batch = int(hyper["batch_size"])
                    hyper["batch_size"] = max(10, requested_batch - requested_batch % 10)
                    if hyper["batch_size"] != requested_batch:
                        self.adjusted_hyperparameters["batch_size"] = {
                            "requested": requested_batch,
                            "applied": hyper["batch_size"],
                            "reason": "ctgan requires batch_size % 10 == 0",
                        }
                accepted, ignored = _filter_signature(synthesizer_cls.__init__, hyper)
                self.ignored_hyperparameters = sorted(
                    set(self.ignored_hyperparameters) | set(ignored)
                )
                synthesizer = synthesizer_cls(metadata, **accepted)
                synthesizer.fit({TABLE_NAME: df})
                self._level_models[level_value] = synthesizer
        finally:
            self.fit_resources = tracker.stop()
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    def _level_values(self) -> list[Any]:
        levels = self._grid.get("level") if self._grid else None
        return list(np.asarray(levels).tolist()) if levels is not None else [None]

    def _rows_per_level(self) -> int:
        grid = self._grid
        return (
            int(np.asarray(grid["time"]).size)
            * int(np.asarray(grid["latitude"]).size)
            * int(np.asarray(grid["longitude"]).size)
        )

    def _sample_rows(self, synthesizer: Any, rows: int) -> np.ndarray:
        collected: list[np.ndarray] = []
        drawn = 0
        while drawn < rows:
            need = max(min(rows - drawn, 8192), 1)
            frame = _unwrap_sampled(
                synthesizer.sample(num_rows=need, batch_size=min(need, 4096))
            )
            batch = frame[self._variables].to_numpy(dtype=float, na_value=np.nan)
            if batch.shape[0] == 0:
                raise RuntimeError("SDV returned 0 rows; is the synthesizer fitted?")
            collected.append(batch)
            drawn += batch.shape[0]
        return np.vstack(collected)[:rows]

    def sample(self) -> GenerationResult:
        """Draw rows per level and rebuild fields on the fitted grid."""

        self.require_fitted()
        grid = self._grid
        n_t = int(np.asarray(grid["time"]).size)
        n_a = int(np.asarray(grid["latitude"]).size)
        n_o = int(np.asarray(grid["longitude"]).size)
        levels = self._level_values()
        n_l = len(levels)
        per_level = n_t * n_a * n_o
        grid_rows = per_level * n_l

        tracker = ResourceTracker(label=f"sample:{self.model_name}").start()
        try:
            full = np.empty((n_t, n_l, n_a, n_o, len(self._variables)), dtype=float)
            for li, level_value in enumerate(levels):
                rows = self._sample_rows(self._level_models[level_value], per_level)
                full[:, li, :, :, :] = rows.reshape(n_t, n_a, n_o, len(self._variables))
            rows_flat = full.reshape(-1, len(self._variables))
            meta = {
                "variables": list(self._variables),
                "features": list(self._variables),
                "grid": {
                    "time": np.asarray(grid["time"]).tolist(),
                    "level": np.asarray(grid["level"]).tolist() if "level" in grid else None,
                    "latitude": np.asarray(grid["latitude"]).tolist(),
                    "longitude": np.asarray(grid["longitude"]).tolist(),
                },
            }
            dataset = rows_to_fields(rows_flat, meta, variables=list(self._variables))
        finally:
            self.sample_resources = tracker.stop()

        contract = self._contract(
            dataset, rows_generated=int(rows_flat.shape[0]), rows_requested=grid_rows
        )
        dataset = attach_contract(dataset, contract)
        return GenerationResult(
            dataset=dataset,
            contract=contract,
            metadata={
                "rows_requested": grid_rows,
                "rows_generated": int(rows_flat.shape[0]),
                "grid_rows": grid_rows,
                "rows_per_level": per_level,
                "levels": levels,
                "training_scheme": TRAINING_SCHEME,
                "ignored_hyperparameters": list(self.ignored_hyperparameters),
                "adjusted_hyperparameters": dict(self.adjusted_hyperparameters),
                "resources": self.sample_resources,
                "fit_resources": self.fit_resources,
            },
        )

    # ------------------------------------------------------------------ #
    def _contract(
        self, dataset: xr.Dataset, *, rows_generated: int, rows_requested: int
    ) -> dict[str, Any]:
        generation_config = {
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "samples": {
                "unit": "cell_time",
                "rows_requested": rows_requested,
                "rows_generated": rows_generated,
                "fields": int(dataset["time"].size),
                "training_scheme": TRAINING_SCHEME,
                "levels": self._level_values(),
                "covariates_configured": list(self._covariates),
                "covariates_not_used": [
                    name for name in self._covariates if name != "level"
                ],
            },
            "training": {
                key: value
                for key, value in self.training.items()
                if key != "discriminator_steps" or self.model_name == "ctgan"
            },
        }
        return build_contract(
            dataset,
            generation_method="statistical",
            source={
                "kind": "synthetic",
                "dataset": f"sdv:{self.model_name}",
                "uri": None,
                "trained_from": self.trained_from,
                "access_date": None,
                "readonly": True,
                "license": "Synthetic output of AtmosGen Phase 2 (statistical generation)",
            },
            model_name=f"sdv.{MODEL_CLASSES[self.model_name]}",
            model_version=self.sdv_version,
            random_seed=int(self.seed),
            generation_config=generation_config,
            metadata={
                "created_at": self.created_at,
                "elapsed_seconds_fit": self.fit_resources.get("elapsed_seconds"),
                "peak_rss_mb_fit": self.fit_resources.get("peak_rss_mb"),
                "notes": (
                    "Mode 1 statistical generation, level-wise scheme: per-level joint "
                    "distributions from SDV; between-level/temporal/spatial structure is "
                    "not modelled and is measured by validation (plan §4.1 limitation)."
                ),
            },
        )

    # ------------------------------------------------------------------ #
    def sample_conditional(
        self,
        known_values: Mapping[str, Any],
        *,
        num_rows: int = 1,
        default_upon_rejection: bool = False,
    ) -> pd.DataFrame:
        """Phase 4 hook: conditional sampling via SDV conditions.

        A ``level`` key selects the corresponding level-wise model (removed from
        the conditions); remaining known columns go to ``sample_from_conditions``.
        """

        self.require_fitted()
        from sdv.sampling import Condition

        conditions = dict(known_values)
        level_value = conditions.pop("level", self._level_values()[0])
        if level_value not in self._level_models:
            raise KeyError(
                f"level={level_value!r} 无对应模型；已训练层: {list(self._level_models)}"
            )
        synthesizer = self._level_models[level_value]
        if not conditions:
            return _unwrap_sampled(synthesizer.sample(num_rows=int(num_rows)))
        condition = Condition(column_values=dict(conditions), num_rows=int(num_rows))
        accepted, _ignored = _filter_signature(
            synthesizer.sample_from_conditions,
            {"batch_size": None, "default_upon_rejection": default_upon_rejection},
        )
        return _unwrap_sampled(synthesizer.sample_from_conditions([condition], **accepted))


class GaussianCopulaGenerator(SDVGenerator):
    name = "gaussian_copula"
    description = "SDV GaussianCopulaSynthesizer（统计 Copula，快）"


class CTGANGenerator(SDVGenerator):
    name = "ctgan"
    description = "SDV CTGANSynthesizer（深度 GAN，CPU 慢）"


class TVAEGenerator(SDVGenerator):
    name = "tvae"
    description = "SDV TVAESynthesizer（深度 VAE）"


GENERATOR_CLASSES: dict[str, type[SDVGenerator]] = {
    "gaussian_copula": GaussianCopulaGenerator,
    "ctgan": CTGANGenerator,
    "tvae": TVAEGenerator,
}


__all__ = [
    "CTGANGenerator",
    "GENERATOR_CLASSES",
    "GaussianCopulaGenerator",
    "MODEL_CLASSES",
    "SDVGenerator",
    "TABLE_NAME",
    "TRAINING_SCHEME",
    "TVAEGenerator",
    "_filter_signature",
]
