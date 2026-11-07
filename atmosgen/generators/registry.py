"""Generator registry (plan §6.1: unknown model names must list real choices).

Registry entries are intentionally import-light: adapter modules are imported
lazily by the phase that implements them, so ``atmosgen list-models`` and
``validate-config`` work even in an environment without SDV / torch.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


def _module_available(module: str) -> tuple[bool, str]:
    if importlib.util.find_spec(module) is None:
        return False, f"missing module {module!r}"
    try:
        imported = importlib.import_module(module)
    except Exception as exc:  # pragma: no cover - broken installs
        return False, f"import {module} failed: {type(exc).__name__}: {exc}"
    return True, str(getattr(imported, "__version__", "unknown"))


def _statistical_factory(model_name: str) -> Callable[[], type]:
    """Lazy factory: importing sdv happens on first use, not import time."""

    def _factory() -> type:
        from .sdv_adapter import GENERATOR_CLASSES

        return GENERATOR_CLASSES[model_name]

    return _factory


def _sdv_dependency() -> list[str]:
    problems: list[str] = []
    for module in ("sdv", "sdmetrics", "sklearn"):
        available, detail = _module_available(module)
        if not available:
            problems.append(detail)
    return problems


@dataclass(frozen=True)
class ModelSpec:
    """Registry entry describing one selectable generator."""

    name: str
    mode: str
    description: str
    phase: int
    dependency_problems: Callable[[], list[str]]
    implemented: bool
    factory: Optional[Callable[[], type]] = None

    def availability(self) -> tuple[bool, str]:
        problems = self.dependency_problems()
        if problems:
            return False, "; ".join(problems)
        return True, "ok"

    def status(self) -> str:
        available, detail = self.availability()
        if not self.implemented:
            return f"not implemented yet (Phase {self.phase}); deps: {detail}"
        if not available:
            return f"implemented, missing deps: {detail}"
        return "implemented"

    def as_dict(self) -> dict[str, Any]:
        available, detail = self.availability()
        return {
            "name": self.name,
            "mode": self.mode,
            "phase": self.phase,
            "implemented": self.implemented,
            "dependencies_available": available,
            "detail": detail,
            "status": self.status(),
            "description": self.description,
        }


MODELS: dict[str, ModelSpec] = {
    "gaussian_copula": ModelSpec(
        name="gaussian_copula",
        mode="statistical",
        description="SDV GaussianCopulaSynthesizer（统计 Copula；CPU 快，主实验默认）",
        phase=2,
        dependency_problems=_sdv_dependency,
        implemented=True,
        factory=_statistical_factory("gaussian_copula"),
    ),
    "ctgan": ModelSpec(
        name="ctgan",
        mode="statistical",
        description="SDV CTGANSynthesizer（深度 GAN；CPU 慢，作对比实验）",
        phase=2,
        dependency_problems=_sdv_dependency,
        implemented=True,
        factory=_statistical_factory("ctgan"),
    ),
    "tvae": ModelSpec(
        name="tvae",
        mode="statistical",
        description="SDV TVAESynthesizer（深度 VAE；CPU 中等）",
        phase=2,
        dependency_problems=_sdv_dependency,
        implemented=True,
        factory=_statistical_factory("tvae"),
    ),
    "advection_diffusion": ModelSpec(
        name="advection_diffusion",
        mode="physics",
        description="自研最小 2D 平流–扩散显式求解器（迎风 + 中心扩散）",
        phase=3,
        dependency_problems=lambda: [],
        implemented=False,
    ),
    "hybrid": ModelSpec(
        name="hybrid",
        mode="hybrid",
        description="物理基底（tcwv）→ SDV 条件生成 → 物理硬约束筛检",
        phase=4,
        dependency_problems=_sdv_dependency,
        implemented=False,
    ),
}


def list_models(mode: str | None = None) -> list[ModelSpec]:
    """All registry entries, optionally filtered by mode."""

    specs = list(MODELS.values())
    if mode is not None:
        specs = [spec for spec in specs if spec.mode == mode]
    return specs


def model_names(mode: str | None = None) -> list[str]:
    return [spec.name for spec in list_models(mode)]


def get_model(name: str) -> ModelSpec:
    """Look up one model, raising a message that lists valid names."""

    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(
            f"unknown model {name!r}; registered models are {model_names()} "
            f"(any registered name works only in its own generation mode)"
        ) from None


def validate_model_choice(mode: str, name: str) -> ModelSpec:
    """Check that ``name`` belongs to ``mode``; raise with the allowed list."""

    spec = get_model(name)
    if spec.mode != mode:
        allowed = model_names(mode)
        raise ValueError(
            f"model {name!r} belongs to mode {spec.mode!r}, not {mode!r}; "
            f"allowed models for mode {mode!r}: {allowed}"
        )
    return spec


def registry_report() -> list[dict[str, Any]]:
    """Serialisable view used by the CLI and reports."""

    return [spec.as_dict() for spec in list_models()]


__all__ = [
    "MODELS",
    "ModelSpec",
    "get_model",
    "list_models",
    "model_names",
    "registry_report",
    "validate_model_choice",
]
