"""Generator base class (plan §2.1, Phase 1 skeleton).

Model adapters land in Phase 2 (SDV statistical), Phase 3 (physics) and Phase 4
(hybrid). This module fixes the interface they must implement so configuration
validation, the registry and the CLI stay stable across phases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping

import xarray as xr

from ..contract import ContractError, build_contract
from ..data.io import save_canonical


@dataclass
class GenerationResult:
    """Return value of :meth:`BaseGenerator.sample`."""

    dataset: xr.Dataset
    contract: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_contract(
        self,
        *,
        generation_method: str,
        source: Mapping[str, Any],
        **kwargs: Any,
    ) -> "GenerationResult":
        """Attach a freshly built contract (idempotent)."""

        self.contract = build_contract(
            self.dataset, generation_method=generation_method, source=source, **kwargs
        )
        return self


class BaseGenerator(ABC):
    """Fit / sample / save contract for every AtmosGen generator."""

    #: registry key, e.g. ``gaussian_copula``
    name: ClassVar[str] = ""
    #: statistical | physics | hybrid
    mode: ClassVar[str] = ""
    #: human-readable description used by ``atmosgen list-models``
    description: ClassVar[str] = ""
    #: phase in which this generator became runnable
    implemented_phase: ClassVar[int] = 1

    def __init__(self, config: Any, *, seed: int = 42) -> None:
        self.config = config
        self.seed = seed
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Availability
    # ------------------------------------------------------------------ #
    @classmethod
    def dependency_problems(cls) -> list[str]:
        """Return human-readable missing-dependency problems (empty = usable)."""

        return []

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        problems = cls.dependency_problems()
        if problems:
            return False, "; ".join(problems)
        return True, "ok"

    # ------------------------------------------------------------------ #
    # Life cycle
    # ------------------------------------------------------------------ #
    @abstractmethod
    def fit(self, dataset: xr.Dataset) -> "BaseGenerator":
        """Fit on a canonical real dataset (statistical/hybrid modes)."""

    @abstractmethod
    def sample(self) -> GenerationResult:
        """Produce a generated canonical dataset."""

    def save(self, result: GenerationResult, path: str | Path, **kwargs: Any) -> Path:
        """Persist a result through the canonical IO layer."""

        if result.contract is None:
            raise ContractError(
                "GenerationResult has no contract; call result.with_contract(...) before saving"
            )
        return save_canonical(result.dataset, path, contract=result.contract, **kwargs)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__}.fit() must run before sample()")

    @classmethod
    def summary(cls) -> dict[str, Any]:
        available, detail = cls.is_available()
        return {
            "name": cls.name,
            "mode": cls.mode,
            "description": cls.description,
            "phase": cls.implemented_phase,
            "available": available,
            "detail": detail,
        }


__all__ = ["BaseGenerator", "GenerationResult"]
