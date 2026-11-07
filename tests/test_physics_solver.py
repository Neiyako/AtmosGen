"""Unit tests for the self-written advection-diffusion solver (plan §4.2)."""

from __future__ import annotations

import numpy as np
import pytest

from atmosgen.contract import read_contract
from atmosgen.data.io import save_canonical
from atmosgen.generators.physics.advection_diffusion import (
    AdvectionDiffusionGenerator,
    PhysicsError,
    check_stability,
    solve_advection_diffusion,
)
from atmosgen.generators.registry import get_model

CENTER = [30.0, 110.0]


def _plume(center=CENTER, sigma_deg=0.45, amplitude=1.0):
    return {
        "kind": "gaussian_plume",
        "center": list(center),
        "sigma_deg": sigma_deg,
        "amplitude": amplitude,
    }


def test_analytic_pure_diffusion_matches():
    """Pure diffusion vs analytic Gaussian at a coarse resolution (<5% L2)."""

    kappa = 1.0e5
    dx = 20_000.0
    dt = 0.2 * dx**2 / (2.0 * kappa)  # 2-D stability headroom
    steps = 25
    dataset, _ = solve_advection_diffusion(
        grid={"nx": 50, "ny": 50, "dx_m": dx, "dy_m": dx},
        time={"dt_s": dt, "steps": steps},
        parameters={"u0": 0.0, "v0": 0.0, "kappa": kappa},
        initial_condition=_plume(),
        boundary_condition={"kind": "periodic"},
    )
    analytic_sigma = 0.45 * 111_320.0
    variance = analytic_sigma**2 + 2.0 * kappa * dt * steps
    lat = np.asarray(dataset["latitude"].values, dtype=float)
    lon = np.asarray(dataset["longitude"].values, dtype=float)
    yy, xx = np.meshgrid(
        (lat - CENTER[0]) * 111_320.0,
        (lon - CENTER[1]) * 111_320.0 * np.cos(np.radians(CENTER[0])),
        indexing="ij",
    )
    analytic = (analytic_sigma**2 / variance) * np.exp(-(xx**2 + yy**2) / (2.0 * variance))
    numeric = np.asarray(dataset["c"].values[-1], dtype=float)
    np.testing.assert_allclose(numeric, analytic, rtol=0.05, atol=1e-3)


def test_advection_centroid_displacement():
    dx = 2000.0
    dt = 100.0
    u0 = 10.0
    steps = 120  # 120 km << 256 km half-domain
    dataset, _ = solve_advection_diffusion(
        grid={"nx": 256, "ny": 256, "dx_m": dx, "dy_m": dx},
        time={"dt_s": dt, "steps": steps},
        parameters={"u0": u0, "v0": 0.0, "kappa": 0.0},
        initial_condition=_plume(),
        boundary_condition={"kind": "periodic"},
    )
    field = np.asarray(dataset["c"].values[-1], dtype=float)
    lon = np.asarray(dataset["longitude"].values, dtype=float)
    x = (lon - CENTER[1]) * 111_320.0 * np.cos(np.radians(CENTER[0]))
    centroid = float((field * x[None, :]).sum() / field.sum())
    assert abs(centroid - u0 * dt * steps) / (u0 * dt * steps) < 0.02


def test_periodic_mass_conservation():
    dataset, diagnostics = solve_advection_diffusion(
        grid={"nx": 64, "ny": 64, "dx_m": 42_000.0, "dy_m": 42_000.0},
        time={"dt_s": 600.0, "steps": 240},
        parameters={"u0": 8.0, "v0": 2.0, "kappa": 1.0e5},
        initial_condition=_plume(sigma_deg=1.5),
        boundary_condition={"kind": "periodic"},
    )
    assert diagnostics.mass_drift_max < 1e-6


def test_unstable_parameters_raise():
    with pytest.raises(PhysicsError):
        check_stability(u0=100.0, v0=0.0, kappa=0.0, dt_s=600.0, dx_m=42_000.0, dy_m=42_000.0)
    with pytest.raises(PhysicsError):
        # 2-D diffusion: kappa*dt*(1/dx^2+1/dy^2) > 0.5 must be rejected
        # (1e5*600*2/42000^2 = 0.068 is stable; 8e5 gives 0.545 > 0.5)
        check_stability(u0=0.0, v0=0.0, kappa=8.0e5, dt_s=600.0, dx_m=42_000.0, dy_m=42_000.0)


def test_joint_cfl_rejected_even_if_each_axis_passes():
    """Phase 3 review F1: Cx=Cy=1.0 explodes despite per-axis conditions."""

    # old per-axis check alone would accept this; the joint condition must not
    with pytest.raises(PhysicsError):
        check_stability(u0=8.0, v0=8.0, kappa=0.0, dt_s=5250.0, dx_m=42_000.0, dy_m=42_000.0)
    # ... and the shipped config must still pass comfortably
    check_stability(u0=8.0, v0=2.0, kappa=1.0e5, dt_s=600.0, dx_m=42_000.0, dy_m=42_000.0)


def test_from_real_negative_field_warns():
    import xarray as xr

    real = xr.Dataset(
        {
            "tcwv": (
                ("time", "latitude", "longitude"),
                -np.abs(np.random.default_rng(1).normal(5.0, 2.0, (2, 6, 6))),
                {"units": "kg m**-2", "long_name": "tcwv"},
            )
        },
        coords={
            "time": np.array(["2020-01-01T00", "2020-01-01T06"], dtype="datetime64[ns]"),
            "latitude": np.linspace(20.0, 45.0, 6),
            "longitude": np.linspace(100.0, 125.0, 6),
        },
    )
    _, diagnostics = solve_advection_diffusion(
        grid={"nx": 24, "ny": 24, "dx_m": 42_000.0, "dy_m": 42_000.0},
        time={"dt_s": 600.0, "steps": 4},
        parameters={"u0": 5.0, "v0": 1.0, "kappa": 1.0e5},
        initial_condition={"kind": "from_real", "variable": "tcwv"},
        boundary_condition={"kind": "fixed_from_real"},
        real=real,
    )
    assert any("负值" in message for message in diagnostics.warnings), diagnostics.warnings


def test_open_boundaries_and_saved_fields():
    dataset, diagnostics = solve_advection_diffusion(
        grid={"nx": 32, "ny": 32, "dx_m": 42_000.0, "dy_m": 42_000.0},
        time={"dt_s": 600.0, "steps": 10},
        parameters={"u0": 8.0, "v0": 2.0, "kappa": 1.0e5},
        initial_condition=_plume(sigma_deg=1.5),
        boundary_condition={"kind": "open"},
        fields=4,
    )
    assert dataset.sizes["time"] == 4
    assert diagnostics.boundary == "open"
    assert np.all(np.isfinite(np.asarray(dataset["c"].values)))


def test_from_real_initial_condition(tmp_path):
    """from_real: the solver can start from a real tcwv-like field."""

    import xarray as xr

    real = xr.Dataset(
        {
            "tcwv": (
                ("time", "latitude", "longitude"),
                20.0 + 10.0 * np.random.default_rng(0).random((1, 8, 8)),
                {"units": "kg m**-2", "long_name": "tcwv"},
            )
        },
        coords={
            "time": np.array(["2020-01-01T00"], dtype="datetime64[ns]"),
            "latitude": np.linspace(20.0, 45.0, 8),
            "longitude": np.linspace(100.0, 125.0, 8),
        },
    )
    dataset, diagnostics = solve_advection_diffusion(
        grid={"nx": 24, "ny": 24, "dx_m": 42_000.0, "dy_m": 42_000.0},
        time={"dt_s": 600.0, "steps": 6},
        parameters={"u0": 5.0, "v0": 1.0, "kappa": 1.0e5},
        initial_condition={"kind": "from_real", "variable": "tcwv"},
        boundary_condition={"kind": "fixed_from_real"},
        real=real,
    )
    assert dataset["c"].values[0].mean() > 0
    assert diagnostics.boundary == "fixed_from_real"
    assert diagnostics.initial_condition["variable"] == "tcwv"
    # synthetic clock follows the real dataset's first timestamp
    assert str(np.asarray(dataset["time"].values)[0]).startswith("2020-01-01")


def test_generator_contract_and_fields():
    from atmosgen.config import GenerationConfigFile

    payload = {
        "project": {"name": "phys", "seed": 5},
        "generation": {
            "mode": "physics",
            "model": "advection_diffusion",
            "samples": {"unit": "field", "fields": 5},
        },
        "physics": {
            "equation": "advection_diffusion",
            "grid": {"nx": 32, "ny": 32, "dx_m": 42000.0, "dy_m": 42000.0},
            "time": {"dt_s": 600.0, "steps": 32},
            "parameters": {"u0": 8.0, "v0": 2.0, "kappa": 100000.0},
            "initial_condition": {"kind": "gaussian_plume", "center": [30.0, 110.0], "sigma_deg": 1.5},
            "boundary_condition": {"kind": "periodic"},
        },
        "output": {"format": "netcdf", "directory": "./output"},
    }
    config = GenerationConfigFile.model_validate(payload)
    spec = get_model("advection_diffusion")
    assert spec.implemented is True
    generator = spec.factory()(config, seed=5, config_path="x.yaml", config_sha256="0" * 64)
    generator.fit(None)
    result = generator.sample()
    dataset = result.dataset
    assert dataset["c"].dims == ("time", "latitude", "longitude")
    contract = read_contract(dataset)
    assert contract["generation_method"] == "physics"
    assert contract["model_name"] == "physics.advection_diffusion"
    assert contract["initial_conditions"]["kind"] == "gaussian_plume"
    assert contract["boundary_conditions"]["kind"] == "periodic"
    assert contract["physical_parameters"]["kappa"] == 100000.0
    assert "level" not in contract["coordinates"]  # physics fields are 2-D
    # mass must be conserved under periodic BC
    diagnostics = result.metadata["diagnostics"]
    assert diagnostics["mass_drift_max"] < 1e-6


def test_generator_output_roundtrip(tmp_path):
    payload = {
        "project": {"name": "phys_save", "seed": 5},
        "generation": {
            "mode": "physics",
            "model": "advection_diffusion",
            "samples": {"unit": "field", "fields": 3},
        },
        "physics": {
            "equation": "advection_diffusion",
            "grid": {"nx": 24, "ny": 24, "dx_m": 42000.0, "dy_m": 42000.0},
            "time": {"dt_s": 600.0, "steps": 12},
            "parameters": {"u0": 8.0, "v0": 2.0, "kappa": 100000.0},
            "initial_condition": {"kind": "gaussian_plume", "center": [30.0, 110.0], "sigma_deg": 1.5},
            "boundary_condition": {"kind": "periodic"},
        },
        "output": {"format": "netcdf", "directory": str(tmp_path / "out")},
    }
    from atmosgen.config import GenerationConfigFile

    config = GenerationConfigFile.model_validate(payload)
    generator = get_model("advection_diffusion").factory()(config, seed=5)
    generator.fit(None)
    result = generator.sample()
    path = save_canonical(
        result.dataset, tmp_path / "data.nc", contract=result.contract, overwrite=True
    )
    from atmosgen.data.io import read_canonical

    reread = read_canonical(path)
    np.testing.assert_array_equal(
        np.asarray(reread["c"].values), np.asarray(result.dataset["c"].values)
    )
