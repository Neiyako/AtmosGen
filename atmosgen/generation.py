"""Generation orchestration (plan §2.1 执行编排).

Phase 2 implements Mode 1 (statistical generation with SDV). Modes 2/3 stay
explicit :class:`NotImplementedInPhase` until Phases 3/4 land them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import xarray as xr

from .config import GenerationConfigFile, config_sha256
from .contract import CONTRACT_ATTR, read_contract
from .data.io import (
    sidecar_contract_path,
    open_source,
    save_canonical,
)
from .data.selection import select_subset, subset_summary
from .data.tabular import thin_grid
from .generators.registry import MODELS, get_model
from .runner import (
    create_run_dir,
    environment_snapshot,
    set_global_seed,
    snapshot_config,
    write_json,
)


class NotImplementedInPhase(RuntimeError):
    """Raised when the requested generation mode is not implemented yet."""

    def __init__(self, mode: str, model: str, phase: int, detail: str = "") -> None:
        super().__init__(
            f"mode={mode!r} model={model!r} 将在 Phase {phase} 实现；"
            f"{detail or '当前不产出占位数据（NOT_RUN）'}"
        )


class GenerationError(RuntimeError):
    """Raised when a generation run fails after starting."""


@dataclass
class RunOutcome:
    """Result of one ``atmosgen generate`` invocation."""

    run_dir: Path
    data_path: Path
    contract: Mapping[str, Any] | None = None
    status: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)


def _trained_from(real_path: Path, real: xr.Dataset) -> dict[str, Any]:
    """Real-data provenance recorded in the generated dataset's contract.

    Reads provenance without full contract validation: a *subset* dataset no
    longer contains every variable declared by the original contract, which is
    expected here and must not fail the run.
    """

    import json

    payload: dict[str, Any] = dict(real.attrs.get("_atmosgen_source_hint", {}) or {})
    doc: dict[str, Any] = {}
    sidecar = sidecar_contract_path(real_path)
    if sidecar.exists():
        try:
            doc = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - defensive
            doc = {}
    if not doc and CONTRACT_ATTR in real.attrs:
        try:
            doc = json.loads(real.attrs[CONTRACT_ATTR])
        except Exception:  # pragma: no cover - defensive
            doc = {}
    if doc:
        source = doc.get("source", {}) or {}
        payload.update(
            {
                key: source.get(key)
                for key in ("dataset", "uri", "access_date", "license", "sha256")
            }
        )
    payload["canonical_path"] = str(real_path)
    return {key: value for key, value in payload.items() if value is not None}


def run_generation(
    config: GenerationConfigFile,
    *,
    config_path: str | Path | None = None,
    overwrite: bool = False,
) -> RunOutcome:
    """Execute one generation run and write ``<run_dir>/data.nc``."""

    model_name = config.generation.model
    mode = config.generation.mode
    spec = get_model(model_name)
    if spec.mode != mode:
        # the config validator already rejects this; belt and braces here
        raise ValueError(f"model {model_name!r} does not belong to mode {mode!r}")
    if not spec.implemented or spec.factory is None:
        raise NotImplementedInPhase(mode, model_name, spec.phase)

    seed = int(config.project.seed)
    set_global_seed(seed)
    run_dir = create_run_dir(
        config.output_directory(), kind=mode, model=model_name, seed=seed
    )
    if config_path is not None:
        snapshot_config(config_path, run_dir)
    write_json(run_dir / "environment.json", environment_snapshot())

    data_path = run_dir / "data.nc"
    if mode == "statistical":
        return _run_statistical(
            config,
            config_path=config_path,
            overwrite=overwrite,
            run_dir=run_dir,
            data_path=data_path,
            seed=seed,
            spec=spec,
        )
    if mode == "physics":
        return _run_physics(
            config,
            config_path=config_path,
            overwrite=overwrite,
            run_dir=run_dir,
            data_path=data_path,
            seed=seed,
            spec=spec,
        )
    raise NotImplementedInPhase(mode, model_name, MODELS[model_name].phase)


def _run_statistical(
    config: GenerationConfigFile,
    *,
    config_path: str | Path | None,
    overwrite: bool,
    run_dir: Path,
    data_path: Path,
    seed: int,
    spec: Any,
) -> RunOutcome:
    data_cfg = config.data
    if data_cfg is None:  # rejected by validation; guard anyway
        raise GenerationError("mode=statistical 需要 data 段")
    real_path = Path(data_cfg.input)
    if not real_path.exists():
        raise FileNotFoundError(
            f"data.input={real_path} 不存在；先运行 atmosgen ingest 生成规范数据"
        )
    real = open_source(real_path, chunks=None)
    try:
        real_sub = select_subset(
            real,
            variables=list(data_cfg.variables),
            time_range=data_cfg.time_range,
            levels=data_cfg.levels,
            spatial_extent=data_cfg.spatial_extent,
        )
        # materialise while the file handle is open
        real_sub = real_sub.load()
    finally:
        real.close()
    trained_from = _trained_from(real_path, real_sub)

    generator_cls = spec.factory()
    generator = generator_cls(
        config,
        seed=seed,
        config_path=str(config_path) if config_path else None,
        config_sha256=config_sha256(config_path) if config_path else None,
        trained_from=trained_from,
    )
    generator.fit(real_sub)
    result = generator.sample()

    compression = config.output.compression.model_dump()
    chunks = dict(data_cfg.chunks) if data_cfg.chunks else None
    written = save_canonical(
        result.dataset,
        data_path,
        contract=result.contract,
        format=config.output.format,
        compression=compression,
        chunks=chunks,
        overwrite=overwrite,
    )
    # report the *effective* training grid (after data.subsample thinning), not
    # the pre-thinning selection: the contract's samples refer to this grid
    subsample = dict(data_cfg.subsample or {})
    effective = thin_grid(
        real_sub,
        lat_step_deg=float(subsample.get("lat_step", 1.0)),
        lon_step_deg=float(subsample.get("lon_step", 1.0)),
    )
    meta = dict(result.metadata)
    meta.update(
        {
            "run_id": run_dir.name,
            "mode": config.generation.mode,
            "model": generator_cls.name,
            "seed": seed,
            "subset": subset_summary(real_sub),
            "effective_grid": subset_summary(effective),
            "subsample": subsample,
            "ignored_hyperparameters": meta.get("ignored_hyperparameters", []),
            "real_dataset": trained_from,
        }
    )
    write_json(run_dir / "metadata.json", meta)
    return RunOutcome(
        run_dir=run_dir,
        data_path=written,
        contract=result.contract,
        status="ok",
        metadata=meta,
    )


def _run_physics(
    config: GenerationConfigFile,
    *,
    config_path: str | Path | None,
    overwrite: bool,
    run_dir: Path,
    data_path: Path,
    seed: int,
    spec: Any,
) -> RunOutcome:
    """Mode 2: run the self-written advection-diffusion solver."""

    from .generators.physics.advection_diffusion import PhysicsError

    physics = config.physics
    if physics is None:
        raise GenerationError("mode=physics 需要 physics 段")
    real: xr.Dataset | None = None
    trained_from: dict[str, Any] | None = None
    data_cfg = config.data
    needs_real = physics.initial_condition.kind == "from_real"
    if needs_real:
        if data_cfg is None:
            raise GenerationError("physics.initial_condition.kind=from_real 需要 data 段")
        real_path = Path(data_cfg.input)
        if not real_path.exists():
            raise FileNotFoundError(f"data.input={real_path} 不存在；先运行 atmosgen ingest")
        handle = open_source(real_path, chunks=None)
        try:
            real = select_subset(
                handle,
                variables=list(data_cfg.variables),
                time_range=data_cfg.time_range,
                levels=data_cfg.levels,
                spatial_extent=data_cfg.spatial_extent,
            ).load()
        finally:
            handle.close()
        trained_from = _trained_from(real_path, real)

    generator_cls = spec.factory()
    generator = generator_cls(
        config,
        seed=seed,
        config_path=str(config_path) if config_path else None,
        config_sha256=config_sha256(config_path) if config_path else None,
        trained_from=trained_from,
    )
    try:
        generator.fit(real)
        result = generator.sample()
    except PhysicsError as exc:
        raise GenerationError(str(exc)) from exc

    written = save_canonical(
        result.dataset,
        data_path,
        contract=result.contract,
        format=config.output.format,
        compression=config.output.compression.model_dump(),
        overwrite=overwrite,
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "run_id": run_dir.name,
            "mode": "physics",
            "model": generator_cls.name,
            "seed": seed,
            "real_dataset": trained_from,
        }
    )
    write_json(run_dir / "metadata.json", metadata)
    return RunOutcome(
        run_dir=run_dir,
        data_path=written,
        contract=result.contract,
        status="ok",
        metadata=metadata,
    )


__all__ = [
    "GenerationError",
    "NotImplementedInPhase",
    "RunOutcome",
    "run_generation",
]
