"""Tests for the cloud-water-oriented fidelity diagnostics script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "quality_fidelity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("quality_fidelity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


qf = _load_module()


def _field(n_time: int = 8, n_lat: int = 5, n_lon: int = 6, seed: int = 0) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    times = (np.arange(n_time).astype("timedelta64[ns]")
             + np.datetime64("2020-01-01T00").astype("timedelta64[ns]")).astype("datetime64[ns]")
    levels = np.array([500.0, 700.0, 850.0])
    lats = np.linspace(20.0, 40.0, n_lat)
    lons = np.linspace(100.0, 125.0, n_lon)
    shape = (n_time, len(levels), n_lat, n_lon)
    return xr.Dataset(
        {
            "t": (("time", "level", "latitude", "longitude"), 250.0 + 10 * rng.random(shape),
                  {"units": "K"}),
            "q": (("time", "level", "latitude", "longitude"), 1e-3 * rng.random(shape),
                  {"units": "kg kg**-1"}),
            "u": (("time", "level", "latitude", "longitude"), rng.normal(size=shape),
                  {"units": "m s**-1"}),
            "v": (("time", "level", "latitude", "longitude"), rng.normal(size=shape),
                  {"units": "m s**-1"}),
        },
        coords={
            "time": times,
            "level": (("level",), levels, {"units": "hPa"}),
            "latitude": (("latitude",), lats, {"units": "degrees_north"}),
            "longitude": (("longitude",), lons, {"units": "degrees_east"}),
        },
    )


def test_ks_statistic_matches_expectations():
    a = np.linspace(0.0, 1.0, 500)
    assert qf.ks_statistic(a, a) == pytest.approx(0.0, abs=1e-9)
    shifted = a + 1.0
    assert qf.ks_statistic(a, shifted) > 0.9


def test_acf_lag1_detects_persistence():
    persistent = np.sin(np.linspace(0, 12, 400))
    noise = np.random.default_rng(0).normal(size=400)
    assert qf.acf_lag1(persistent) > 0.8
    assert abs(qf.acf_lag1(noise)) < 0.3


def test_mfc_of_spatially_smooth_field_is_finite_and_scales():
    ds = _field()
    mfc = qf.moisture_flux_convergence(ds)
    assert mfc is not None
    assert mfc.shape == (ds.sizes["time"], ds.sizes["latitude"], ds.sizes["longitude"])
    assert np.all(np.isfinite(mfc))


def test_mfc_requires_pressure_levels_and_winds():
    ds = _field().drop_vars(["v"])
    assert qf.moisture_flux_convergence(ds) is None


def test_saturation_rate_detects_violation():
    ds = _field()
    # q = 0.05 kg/kg exceeds 1.1 * q_sat(t, p) at every level (q_sat ~ 1e-2)
    values = np.asarray(ds["q"].values)
    values[:] = 0.05
    ds["q"] = (("time", "level", "latitude", "longitude"), values, {"units": "kg kg**-1"})
    result = qf.saturation_violation_rate(ds)
    assert result["rate"] is not None
    assert result["rate"] > 0.8

    clean = _field(seed=3)
    ok = qf.saturation_violation_rate(clean)
    assert ok["rate"] is not None
    assert ok["rate"] < 0.2


def test_saturation_rate_needs_level_units():
    ds = _field()
    ds["level"].attrs.pop("units")
    result = qf.saturation_violation_rate(ds)
    assert result["rate"] is None
    assert "单位" in result["reason"]
    hinted = qf.saturation_violation_rate(ds, level_units_hint="hPa")
    assert hinted["rate"] is not None


def test_align_real_to_skips_unalignable_clock():
    real = _field()
    target = _field()
    shifted_times = (
        np.arange(target.sizes["time"]).astype("timedelta64[ns]")
        + np.datetime64("2000-01-01T00").astype("timedelta64[ns]")
    ).astype("datetime64[ns]")
    target = target.assign_coords(time=shifted_times)
    aligned, skipped = qf.align_real_to(real, target)
    assert "time" in skipped
    assert aligned.sizes["time"] == target.sizes["time"]


def test_diagnose_run_reports_all_metric_families():
    real = _field(seed=1)
    generated = _field(seed=2)
    block = qf.diagnose_run(real, generated)
    assert "ks" in block and "t" in block["ks"]
    assert "temporal_series_correlation" in block
    assert "spatial_pattern_correlation" in block
    assert "saturation" in block
    assert block["mfc"]["available"] is True
    assert np.isfinite(block["mfc"]["pattern_correlation"])
