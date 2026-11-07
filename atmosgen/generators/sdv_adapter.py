"""SDV statistical generator adapters (Mode 1, plan §4.1).

Three models share one adapter class parameterised by registry name:

- ``gaussian_copula`` — GaussianCopulaSynthesizer (copula; fast, default)
- ``ctgan``           — CTGANSynthesizer (deep GAN, CPU-slow)
- ``tvae``            — TVAESynthesizer (deep VAE)

Sampling reconstruction follows plan §4.1: enough rows for the *full* target
grid are drawn from the fitted model and assigned to real coordinates in
raster order by :func:`atmosgen.data.tabular.rows_to_fields`. Spatio-temporal
coherence is NOT promised and is reported by the validation modules.

Hyper-parameters are matched against the installed SDV version by signature
introspection; unsupported keys are recorded as ignored instead of breaking a
run (the ignored list lands in run metadata).
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from ..contract import attach_contract, build_contract
from ..data.tabular import (
    SUPPORTED_COVARIATES,
    TableError,
    build_grid,
    dataset_to_table,
    rows_to_fields,
    table_feature_count,
    thin_grid,
)
from ..runner import ResourceTracker, set_global_seed
from .base import BaseGenerator, GenerationResult

MODEL_CLASSES: dict[str, str] = {
    "gaussian_copula": "GaussianCopulaSynthesizer",
    "ctgan": "CTGANSynthesizer",
    "tvae": "TVAESynthesizer",
}

DEFAULT_TRAINING: dict[str, Any] = {"epochs": 100, "batch_size": 500}


TABLE_NAME = "atmosgen"


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
    if not metadata.to_dict().get("tables"):
        metadata.add_table(TABLE_NAME)
    for column in df.columns:
        metadata.add_column(column_name=column, sdtype="numerical", table_name=TABLE_NAME)
    metadata.validate()
    return metadata


class SDVGenerator(BaseGenerator):
    """Fit/sample/save for the three SDV single-table models."""

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
        training_block = getattr(getattr(config, "generation", None), "training", None) if config else None
        self.training: dict[str, Any] = {
            "epochs": int(getattr(training_block, "epochs", 100)) if training_block else 100,
            "batch_size": int(getattr(training_block, "batch_size", 500)) if training_block else 500,
            "discriminator_steps": int(getattr(training_block, "discriminator_steps", 1)) if training_block else 1,
        }
        self.fit_resources: dict[str, Any] = {}
        self.sample_resources: dict[str, Any] = {}
        self.ignored_hyperparameters: list[str] = []
        self.adjusted_hyperparameters: dict[str, Any] = {}
        self._synthesizer: Any = None
        self._table_meta: Mapping[str, Any] | None = None
        self._feature_names: list[str] = []
        self._grid: Mapping[str, np.ndarray] | None = None
        try:
            import sdv

            self.sdv_version = str(getattr(sdv, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover - availability checked earlier
            raise RuntimeError(f"sdv is not importable: {type(exc).__name__}: {exc}") from exc

    # ------------------------------------------------------------------ #
    # availability
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
    # learning
    # ------------------------------------------------------------------ #
    def _data_settings(self) -> dict[str, Any]:
        data = getattr(self.config, "data", None)
        variables = list(getattr(data, "variables", None) or ["t", "q", "u", "v", "tcwv"])
        covariates = [
            name
            for name in (getattr(getattr(self.config, "generation", None), "covariates", None) or [])
            if name in SUPPORTED_COVARIATES
        ]
        subsample = dict(getattr(data, "subsample", None) or {})
        return {
            "variables": variables,
            "covariates": covariates,
            "lat_step": float(subsample.get("lat_step", 1.0)),
            "lon_step": float(subsample.get("lon_step", 1.0)),
        }

    def fit(self, dataset: xr.Dataset) -> "SDVGenerator":
        """Fit on a canonical real dataset (already subset by the runner)."""

        settings = self._data_settings()
        tracker = ResourceTracker(label=f"fit:{self.model_name}").start()
        try:
            learned_ds = thin_grid(
                dataset, lat_step_deg=settings["lat_step"], lon_step_deg=settings["lon_step"]
            )
            X, features, table_meta = dataset_to_table(
                learned_ds,
                variables=settings["variables"],
                covariates=settings["covariates"],
            )
            self._table_meta = table_meta
            self._feature_names = features
            self._grid = build_grid(learned_ds)

            factories = _sdv_factories()
            metadata = _numerical_metadata(factories["metadata_cls"], pd.DataFrame(X, columns=features))
            hyper: dict[str, Any] = {"enforce_min_max_values": True}
            if self.model_name in {"ctgan", "tvae"}:
                hyper.update(
                    epochs=int(self.training["epochs"]),
                    batch_size=int(self.training["batch_size"]),
                )
                if self.model_name == "ctgan":
                    hyper["discriminator_steps"] = int(self.training["discriminator_steps"])
            # CTGAN pads the discriminator input with `pac=10` columns, so the
            # batch size must be a multiple of 10; round instead of exploding.
            if self.model_name == "ctgan":
                requested_batch = int(hyper["batch_size"])
                hyper["batch_size"] = max(10, requested_batch - requested_batch % 10)
                if hyper["batch_size"] != requested_batch:
                    self.adjusted_hyperparameters[f"batch_size"] = {
                        "requested": requested_batch,
                        "applied": hyper["batch_size"],
                        "reason": "ctgan requires batch_size % 10 == 0",
                    }
            synthesizer_cls = factories["synthesizers"][self.model_name]
            accepted, ignored = _filter_signature(synthesizer_cls.__init__, hyper)
            self.ignored_hyperparameters = ignored
            self._synthesizer = synthesizer_cls(metadata, **accepted)
            self._synthesizer.fit({TABLE_NAME: pd.DataFrame(X, columns=features)})
        finally:
            self.fit_resources = tracker.stop()
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #
    def _requested_rows(self) -> int:
        samples = getattr(getattr(self.config, "generation", None), "samples", None)
        unit = getattr(samples, "unit", "cell_time") if samples is not None else "cell_time"
        if unit != "cell_time":
            raise TableError(
                "samples.unit=field 与表格模型不兼容；统计模式请使用 samples.unit=cell_time"
            )
        requested = int(getattr(samples, "rows", 0) or 0) if samples is not None else 0
        grid_rows = table_feature_count(self._grid)
        return max(requested, grid_rows)

    def sample(self) -> GenerationResult:
        """Draw rows until the target grid is filled, then rebuild the fields.

        Records ``rows_requested`` vs ``rows_generated`` in the contract: SDV
        synthesizers may return fewer rows than asked for, so the adapter
        batches until ``rows_generated >= grid_rows`` and truncates.
        """

        self.require_fitted()
        requested = self._requested_rows()
        grid_rows = table_feature_count(self._grid)
        tracker = ResourceTracker(label=f"sample:{self.model_name}").start()
        try:
            collected: list[np.ndarray] = []
            drawn = 0
            while drawn < grid_rows:
                need = max(min(grid_rows - drawn, max(requested, 512)), 1)
                collected.append(self._sample_batch(need))
                drawn += collected[-1].shape[0]
                if collected[-1].shape[0] == 0:
                    raise RuntimeError("SDV returned 0 rows; is the synthesizer fitted?")
            rows = np.vstack(collected)[:grid_rows]
            dataset = rows_to_fields(
                rows, self._table_meta, variables=self._variable_names()
            )
        finally:
            self.sample_resources = tracker.stop()
        contract = self._contract(
            dataset, rows_generated=int(rows.shape[0]), rows_requested=requested
        )
        # attach before returning: readers can then trust the dataset alone
        dataset = attach_contract(dataset, contract)
        return GenerationResult(
            dataset=dataset,
            contract=contract,
            metadata={
                "rows_requested": requested,
                "rows_generated": int(rows.shape[0]),
                "grid_rows": grid_rows,
                "ignored_hyperparameters": list(self.ignored_hyperparameters),
                "adjusted_hyperparameters": dict(self.adjusted_hyperparameters),
                "resources": self.sample_resources,
                "fit_resources": self.fit_resources,
            },
        )

    def _sample_batch(self, rows: int) -> np.ndarray:
        sampled = self._synthesizer.sample(num_rows=rows, batch_size=min(rows, 4096))
        # sdv 2.x returns {table_name: DataFrame}; sdv 1.x returned a DataFrame
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
        missing = [name for name in self._feature_names if name not in frame.columns]
        if missing:
            raise RuntimeError(f"SDV sample is missing columns {missing}")
        return frame[self._feature_names].to_numpy(dtype=float, na_value=np.nan)

    def _variable_names(self) -> list[str]:
        if self._table_meta is None:
            return []
        return [name for name in self._table_meta["variables"] if name in self._feature_names]

    # ------------------------------------------------------------------ #
    # contract
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
                    "Mode 1 statistical generation; rows carry generated values only, "
                    "grid coordinates are real (plan §4.1). Temporal/spatial coherence "
                    "is measured by validation, not assumed."
                ),
            },
        )

    # ------------------------------------------------------------------ #
    # conditional sampling (Phase 4 hook)
    # ------------------------------------------------------------------ #
    def sample_conditional(
        self,
        known_values: Mapping[str, Any],
        *,
        num_rows: int = 1,
        default_upon_rejection: bool = False,
    ) -> pd.DataFrame:
        """Sample with known column values (hybrid mode entry point)."""

        self.require_fitted()
        from sdv.sampling import Condition

        condition = Condition(column_values=dict(known_values), num_rows=int(num_rows))
        accepted, _ignored = _filter_signature(
            self._synthesizer.sample_from_conditions,
            {"batch_size": None, "default_upon_rejection": default_upon_rejection},
        )
        sampled = self._synthesizer.sample_from_conditions([condition], **accepted)
        frame = sampled[TABLE_NAME] if isinstance(sampled, Mapping) else sampled
        return frame


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
    "TVAEGenerator",
    "_filter_signature",
]
