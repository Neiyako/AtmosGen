"""Hybrid mode (Mode 3) unit tests: physics base, conditional sampling, screening."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from atmosgen.config import GenerationConfigFile
from atmosgen.contract import read_contract
from atmosgen.hybrid import (
    HybridError,
    conditional_generate,
    q_sat_from_t_and_p,
    screen_samples,
    write_rejected_jsonl,
)

pytest.importorskip("sdv")


def _config(tmp_path: Path, *, model: str = "gaussian_copula") -> GenerationConfigFile:
    return GenerationConfigFile.model_validate(
        {
            "project": {"name": "hybrid_test", "seed": 42},
            "data": {
                "input": str(tmp_path / "real.nc"),
                "variables": ["t", "q", "u", "v", "tcwv"],
                "levels": [500.0, 700.0, 850.0],
                "subsample": {"lat_step": 1.0, "lon_step": 1.0},
            },
            "generation": {
                "mode": "hybrid",
                "model": "hybrid",
                "samples": {"unit": "cell_time", "rows": 1000},
            },
            "physics": {
                "equation": "advection_diffusion",
                "grid": {"nx": 8, "ny": 8, "dx_m": 42000.0, "dy_m": 42000.0},
                "time": {"dt_s": 600.0, "steps": 4},
                "parameters": {"u0": 8.0, "v0": 2.0, "kappa": 100000.0},
                "initial_condition": {"kind": "from_real", "variable": "tcwv"},
                "boundary_condition": {"kind": "periodic"},
            },
            "hybrid": {
                "physics_component": {"enabled": True, "output_variable": "tcwv"},
                "statistical_component": {
                    "model": model,
                    "trained_on": ["real", "physics"],
                },
                "screening": {
                    "hard_constraints": True,
                    "reject_ratio_limit": 0.2,
                    "soft_metrics": ["mass_budget"],
                },
            },
            "output": {"format": "netcdf", "directory": str(tmp_path / "output")},
        }
    )


def _real_dataset() -> xr.Dataset:
    rng = np.random.default_rng(0)
    times = np.array(
        ["2010-05-01T00", "2010-05-01T06", "2010-05-01T12"], dtype="datetime64[ns]"
    )
    levels = np.array([500.0, 700.0, 850.0])
    lats = np.linspace(20.0, 45.0, 5)
    lons = np.linspace(100.0, 125.0, 5)
    ds = xr.Dataset(
        {
            "t": (
                ("time", "level", "latitude", "longitude"),
                250.0 + 20.0 * rng.random((3, 3, 5, 5)),
                {"units": "K", "long_name": "Temperature"},
            ),
            "q": (
                ("time", "level", "latitude", "longitude"),
                1e-3 * rng.random((3, 3, 5, 5)),
                {"units": "kg kg**-1", "long_name": "Specific humidity"},
            ),
            "u": (
                ("time", "level", "latitude", "longitude"),
                5.0 * rng.normal(size=(3, 3, 5, 5)),
                {"units": "m s**-1", "long_name": "U wind"},
            ),
            "v": (
                ("time", "level", "latitude", "longitude"),
                5.0 * rng.normal(size=(3, 3, 5, 5)),
                {"units": "m s**-1", "long_name": "V wind"},
            ),
            "tcwv": (
                ("time", "latitude", "longitude"),
                20.0 + 10.0 * rng.random((3, 5, 5)),
                {"units": "kg m**-2", "long_name": "Total column water vapour"},
            ),
        },
        coords={
            "time": times,
            "level": (("level",), levels, {"units": "hPa"}),
            "latitude": (("latitude",), lats, {"units": "degrees_north"}),
            "longitude": (("longitude",), lons, {"units": "degrees_east"}),
        },
    )
    return ds


def test_q_sat_matches_tetens_reference():
    # 300 K -> e_s = 6.112*exp(17.67*26.85/270.35) = 35.36 hPa (standard value)
    q_sat = q_sat_from_t_and_p(np.array([300.0]), np.array([850.0]))
    expected = 0.622 * 35.36 / (850.0 - 0.378 * 35.36)
    np.testing.assert_allclose(q_sat, [expected], rtol=1e-3)


def test_screening_detects_each_rule():
    frame = pd.DataFrame(
        {
            "t": [280.0, 400.0, 280.0, 280.0],
            "q": [0.001, 0.001, -0.1, 0.5],
            "u": [1.0, 1.0, 1.0, 1.0],
            "v": [1.0, 1.0, 1.0, 200.0],
            "tcwv": [20.0, 20.0, 20.0, 20.0],
        }
    )
    result = screen_samples(frame, pressure_hpa=np.array([850.0, 850.0, 850.0, 850.0]))
    assert result.accepted.tolist() == [True, False, False, False]
    reasons = set(result.reasons)
    assert "t_out_of_range" in reasons
    assert "q_negative" in reasons
    assert "u_out_of_range" in reasons or "v_out_of_range" in reasons
    assert "supersaturated" in reasons
    assert result.rejection_rate == pytest.approx(0.75)


def test_screening_keeps_valid_rows_and_reports_rate():
    frame = pd.DataFrame(
        {
            "t": [280.0, 281.0],
            "q": [0.001, 0.002],
            "u": [1.0, 2.0],
            "v": [1.0, 2.0],
            "tcwv": [20.0, 21.0],
        }
    )
    result = screen_samples(frame, pressure_hpa=np.array([850.0, 850.0]))
    assert result.accepted.all()
    assert result.rejection_rate == 0.0
    summary = result.summary()
    assert summary["total_samples"] == 2
    assert summary["rejected_samples"] == 0


def test_conditional_generation_preserves_known_column():
    from sdv.metadata import Metadata
    from sdv.single_table import GaussianCopulaSynthesizer

    rng = np.random.default_rng(1)
    n = 500
    tcwv = 20.0 + 30.0 * rng.random(n)
    frame = pd.DataFrame(
        {
            "t": 240.0 + 0.6 * tcwv + 2.0 * rng.normal(size=n),
            "q": 1e-4 * (tcwv / 30.0) ** 2 + 1e-4 * rng.random(n),
            "u": 3.0 * rng.normal(size=n),
            "v": 3.0 * rng.normal(size=n),
            "tcwv": tcwv,
        }
    )
    metadata = Metadata()
    metadata.add_table("atmosgen")
    for column in frame.columns:
        metadata.add_column(column_name=column, sdtype="numerical", table_name="atmosgen")
    metadata.validate()
    synthesizer = GaussianCopulaSynthesizer(metadata, enforce_min_max_values=True)
    synthesizer.fit({"atmosgen": frame})
    known = frame["tcwv"].values[:200]
    out = conditional_generate(
        synthesizer, known_column="tcwv", known_values=known
    )
    np.testing.assert_array_equal(out["tcwv"].values, known)
    # conditioning actually correlates the generated variables with the base
    assert abs(float(np.corrcoef(out["t"].values, known)[0, 1])) > 0.3


def test_hybrid_generator_end_to_end(tmp_path):
    from atmosgen.generators.hybrid_adapter import HybridGenerator

    config = _config(tmp_path)
    real = _real_dataset()
    generator = HybridGenerator(config, seed=42)
    generator.fit(real)
    result = generator.sample()
    dataset = result.dataset

    assert set(dataset.data_vars) == {"t", "q", "u", "v", "tcwv"}
    assert dataset["tcwv"].dims == ("time", "latitude", "longitude")
    assert dataset["t"].dims == ("time", "level", "latitude", "longitude")

    contract = read_contract(dataset)
    assert contract["generation_method"] == "hybrid"
    assert contract["model_name"].startswith("hybrid.")
    assert contract["quality_metrics"]["screening_before"]["total_samples"] > 0
    assert contract["quality_metrics"]["screening_after"]["rejected_samples"] == 0

    before = result.metadata["screening_before"]
    after = result.metadata["screening_after"]
    assert before["rejection_rate"] >= 0.0
    assert after["rejected_samples"] == 0

    # rejected cells are masked (NaN) and logged, never silently dropped
    masked = int(np.isnan(dataset["q"].values).sum())
    assert masked == before["rejected_samples"]
    log = generator.write_rejected(tmp_path / "rejected.jsonl")
    lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == before["rejected_samples"]
    import json

    first = json.loads(lines[0])
    assert "reasons" in first and first["reasons"]
    assert "row" in first


def test_hybrid_requires_real_data(tmp_path):
    from atmosgen.generators.hybrid_adapter import HybridGenerator

    config = _config(tmp_path)
    generator = HybridGenerator(config, seed=42)
    with pytest.raises(HybridError):
        generator.fit(None)
