"""Generation orchestration (plan §2.1 执行编排).

Phase 2 implements Mode 1 (statistical generation with SDV). Modes 2/3 stay
explicit :class:`NotImplementedInPhase` until Phases 3/4 land them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import xarray as xr

from .config import GenerationConfigFile, config_sha256
from .contract import CONTRACT_ATTR, attach_contract, build_contract, read_contract
from .data.io import (
    sidecar_contract_path,
    open_source,
    save_canonical,
)
from .data.selection import select_subset, subset_summary
from .data.tabular import thin_grid
from .derived import compute_derived
from .generators.base import GenerationResult
from .generators.registry import MODELS, get_model
from .runner import (
    create_run_dir,
    environment_snapshot,
    set_global_seed,
    snapshot_config,
    write_json,
)
from .variables import spec_for, transitive_dependencies

#: Primary variables the coupled multifield generator can produce.
MULTIFIELD_PRIMARY_VARIABLES: tuple[str, ...] = (
    "t",
    "q",
    "u",
    "v",
    "w",
    "z",
    "sp",
    "clwc",
    "ciwc",
    "crwc",
    "cswc",
    "tp",
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


def _primary_dependencies(name: str) -> tuple[str, ...]:
    """Primary (non-derived) variables ``name`` transitively depends on."""

    closure = transitive_dependencies([name])
    return tuple(
        dep for dep in closure.get(name, ()) if not spec_for(dep).is_derived
    )


def check_requested_variables(
    requested: Iterable[str],
    capabilities: Iterable[str],
    *,
    model: str,
) -> list[str]:
    """Return the problems of a requested variable set (empty = satisfiable).

    A requested variable is satisfiable when the model can provide it, or when
    every *primary* dependency of the variable can be provided (the derived
    dependencies are computed by the derived-diagnostics engine).
    """

    available = set(capabilities)
    problems: list[str] = []
    for name in requested:
        if name in available:
            continue
        spec = spec_for(name)
        if not spec.is_derived:
            problems.append(
                f"model={model} 无法提供主变量 {name!r}"
                f"（可用: {sorted(available)}）"
            )
            continue
        missing = [dep for dep in _primary_dependencies(name) if dep not in available]
        if missing:
            problems.append(
                f"派生变量 {name!r} 的依赖 {missing} 无法由 model={model} 提供"
                f"（可用: {sorted(available)}）"
            )
    return problems


def rebuild_contract(
    dataset: xr.Dataset,
    contract: Mapping[str, Any],
    *,
    unavailable: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Rebuild a contract for a dataset that changed after the generator ran.

    Every field of the original contract is preserved; only the variable list,
    the units and the metadata are recomputed. Variables that could not be
    produced are recorded in ``metadata.unavailable_variables`` with their
    reason — they are never written with a placeholder value.
    """

    metadata = dict(contract.get("metadata") or {})
    metadata.pop("created_at", None)
    # the variable registry block is recomputed from the new dataset
    metadata.pop("variable_definitions", None)
    if unavailable:
        metadata["unavailable_variables"] = dict(unavailable)
        notes = str(metadata.get("notes") or "")
        extra = "; 请求的变量不可用 (NOT_AVAILABLE): " + ", ".join(
            f"{name} ({reason})" for name, reason in unavailable.items()
        )
        metadata["notes"] = (notes + extra).lstrip("; ")
    return build_contract(
        dataset,
        generation_method=str(contract["generation_method"]),
        source=dict(contract.get("source") or {}),
        model_name=contract.get("model_name"),
        model_version=contract.get("model_version"),
        random_seed=contract.get("random_seed"),
        initial_conditions=contract.get("initial_conditions"),
        boundary_conditions=contract.get("boundary_conditions"),
        physical_parameters=contract.get("physical_parameters"),
        generation_config=contract.get("generation_config"),
        quality_metrics=contract.get("quality_metrics"),
        metadata=metadata,
    )


def apply_variable_plan(
    result: GenerationResult,
    config: GenerationConfigFile,
    *,
    capabilities: Iterable[str],
    model: str,
) -> GenerationResult:
    """Apply the requested variable selection to a generation result.

    * the dependency check runs **before** anything is computed: a request that
      cannot be satisfied raises :class:`GenerationError` (no silent guessing);
    * the derived variables are computed by the derived-diagnostics engine;
    * a variable that still cannot be computed is reported as
      ``NOT_AVAILABLE`` with its reason (never as a zero-filled placeholder);
    * the dataset is subset to the requested variables and the contract is
      rebuilt so that units, coordinates and attributes stay consistent.
    """

    requested = config.generation.variables
    if requested is None:
        # Legacy behaviour: the generator decides which variables exist.
        return result
    problems = check_requested_variables(requested, capabilities, model=model)
    if problems:
        raise GenerationError("变量依赖检查未通过: " + "; ".join(problems))

    derived_targets = [name for name in requested if spec_for(name).is_derived]
    dataset, report = compute_derived(result.dataset, variables=derived_targets)

    delivered = [name for name in requested if name in dataset.data_vars]
    unavailable = {
        name: str(report["unavailable"].get(name, "生成器未提供该变量"))
        for name in requested
        if name not in dataset.data_vars
    }
    if delivered:
        dataset = dataset[delivered]

    contract = rebuild_contract(dataset, result.contract or {}, unavailable=unavailable)
    dataset = attach_contract(dataset, contract)

    metadata = dict(result.metadata)
    metadata["variable_plan"] = {
        "requested": list(requested),
        "delivered": delivered,
        "unavailable": unavailable,
        "derived_report": report,
    }
    return GenerationResult(dataset=dataset, contract=contract, metadata=metadata)


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
    if mode == "physics" and model_name == "multifield":
        return _run_multifield(
            config,
            config_path=config_path,
            overwrite=overwrite,
            run_dir=run_dir,
            data_path=data_path,
            seed=seed,
            spec=spec,
        )
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
    if mode == "hybrid":
        return _run_hybrid(
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

    result = apply_variable_plan(
        result, config, capabilities=list(real_sub.data_vars), model=spec.name
    )

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

    result = apply_variable_plan(
        result, config, capabilities=("c",), model=spec.name
    )

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


def _run_multifield(
    config: GenerationConfigFile,
    *,
    config_path: str | Path | None,
    overwrite: bool,
    run_dir: Path,
    data_path: Path,
    seed: int,
    spec: Any,
) -> RunOutcome:
    """Mode 2b: the coupled multi-variable physics field generator."""

    from .generators.physics.multifield import PhysicsError as MultifieldError

    if config.multifield is None:
        raise GenerationError("mode=physics + model=multifield 需要 multifield 段")
    generator_cls = spec.factory()
    generator = generator_cls(
        config,
        seed=seed,
        config_path=str(config_path) if config_path else None,
        config_sha256=config_sha256(config_path) if config_path else None,
        trained_from=None,
    )
    try:
        generator.fit(None)
        result = generator.sample()
    except MultifieldError as exc:
        raise GenerationError(str(exc)) from exc

    result = apply_variable_plan(
        result, config, capabilities=MULTIFIELD_PRIMARY_VARIABLES, model=spec.name
    )

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
            "real_dataset": None,
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


def _run_hybrid(
    config: GenerationConfigFile,
    *,
    config_path: str | Path | None,
    overwrite: bool,
    run_dir: Path,
    data_path: Path,
    seed: int,
    spec: Any,
) -> RunOutcome:
    """Mode 3: physics base + conditional generation + physical screening."""

    from .generators.hybrid_adapter import HybridGenerator

    data_cfg = config.data
    if data_cfg is None:
        raise GenerationError("mode=hybrid 需要 data 段（训练统计模型 + 物理基底初值）")
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
    except Exception as exc:  # noqa: BLE001 - wrap into a runtime error with context
        from .hybrid import HybridError

        if isinstance(exc, HybridError):
            raise GenerationError(str(exc)) from exc
        raise

    result = apply_variable_plan(
        result,
        config,
        capabilities=[*real.data_vars, "tcwv"],
        model=spec.name,
    )

    written = save_canonical(
        result.dataset,
        data_path,
        contract=result.contract,
        format=config.output.format,
        compression=config.output.compression.model_dump(),
        overwrite=overwrite,
    )
    rejected_path = run_dir / "rejected.jsonl"
    if hasattr(generator, "write_rejected"):
        generator.write_rejected(rejected_path)
    metadata = dict(result.metadata)
    metadata.update(
        {
            "run_id": run_dir.name,
            "mode": "hybrid",
            "model": generator_cls.name,
            "seed": seed,
            "real_dataset": trained_from,
            "rejected_path": str(rejected_path) if rejected_path.exists() else None,
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
    "MULTIFIELD_PRIMARY_VARIABLES",
    "NotImplementedInPhase",
    "RunOutcome",
    "apply_variable_plan",
    "check_requested_variables",
    "rebuild_contract",
    "run_generation",
]
