"""Generator registry and base-class contract (Phase 1 skeleton)."""

from __future__ import annotations

import pytest

from atmosgen.generators.base import BaseGenerator, GenerationResult
from atmosgen.generators.registry import (
    get_model,
    list_models,
    model_names,
    registry_report,
    validate_model_choice,
)


def test_registry_contains_all_planned_models():
    assert set(model_names()) == {
        "gaussian_copula",
        "ctgan",
        "tvae",
        "advection_diffusion",
        "hybrid",
    }
    assert [spec.name for spec in list_models("statistical")] == [
        "gaussian_copula",
        "ctgan",
        "tvae",
    ]
    assert [spec.name for spec in list_models("physics")] == ["advection_diffusion"]


def test_unknown_model_error_lists_registered_models():
    with pytest.raises(KeyError) as excinfo:
        get_model("magic_net")
    message = str(excinfo.value)
    for name in model_names():
        assert name in message


def test_model_mode_validation():
    assert validate_model_choice("statistical", "ctgan").name == "ctgan"
    with pytest.raises(ValueError) as excinfo:
        validate_model_choice("physics", "ctgan")
    assert "advection_diffusion" in str(excinfo.value)


def test_registry_report_marks_phase1_not_implemented():
    report = registry_report()
    assert all(entry["phase"] >= 2 for entry in report)
    assert all(entry["implemented"] is False for entry in report)


def test_base_generator_requires_lifecycle_implementation():
    class Dummy(BaseGenerator):
        name = "dummy"
        mode = "statistical"

        def fit(self, dataset):  # pragma: no cover - trivial
            self._fitted = True
            return self

        def sample(self):  # pragma: no cover - trivial
            self.require_fitted()
            return GenerationResult(dataset=None)

    generator = Dummy(config=None, seed=1)
    with pytest.raises(RuntimeError):
        generator.sample()
    assert Dummy.is_available() == (True, "ok")
