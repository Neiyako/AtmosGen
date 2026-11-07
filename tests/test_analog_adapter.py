"""Tests for the structure-preserving synthesis mode (analog_blocks)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from atmosgen.config import GenerationConfigFile
from atmosgen.contract import read_contract
from atmosgen.generators.analog_adapter import AnalogGenerator, AnalogError

from conftest import build_small_dataset


def _config(tmp_path: Path, **structure) -> GenerationConfigFile:
    payload = {
        "project": {"name": "analog_test", "seed": 42},
        "data": {
            "input": str(tmp_path / "real.nc"),
            "variables": ["t", "q", "u", "v", "tcwv"],
            "levels": [500.0, 700.0, 850.0],
            "subsample": {"lat_step": 1.0, "lon_step": 1.0},
        },
        "generation": {
            "mode": "statistical",
            "model": "analog_blocks",
            "samples": {"unit": "cell_time", "rows": 100},
            "structure": {"block_steps": 2, "patch_cells": 4, "patch_mode": "whole_field"},
        },
        "output": {"format": "netcdf", "directory": str(tmp_path / "output")},
    }
    payload["generation"]["structure"].update(structure)
    return GenerationConfigFile.model_validate(payload)


def _real(n_time: int = 4, n_lat: int = 5, n_lon: int = 6) -> xr.Dataset:
    rng = np.random.default_rng(0)
    times = (
        np.arange(n_time).astype("timedelta64[ns]")
        + np.datetime64("2020-01-01T00").astype("timedelta64[ns]")
    ).astype("datetime64[ns]")
    levels = np.array([500.0, 700.0, 850.0])
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
            "tcwv": (("time", "latitude", "longitude"), 20.0 + 5 * rng.random((n_time, n_lat, n_lon)),
                  {"units": "kg m**-2"}),
        },
        coords={
            "time": times,
            "level": (("level",), levels, {"units": "hPa"}),
            "latitude": (("latitude",), np.linspace(20.0, 40.0, n_lat), {"units": "degrees_north"}),
            "longitude": (("longitude",), np.linspace(100.0, 125.0, n_lon), {"units": "degrees_east"}),
        },
    )


def test_analog_requires_real_data(tmp_path):
    generator = AnalogGenerator(_config(tmp_path), seed=42)
    with pytest.raises(AnalogError):
        generator.fit(None)


def test_analog_preserves_dimensional_structure(tmp_path):
    generator = AnalogGenerator(_config(tmp_path), seed=42)
    generator.fit(_real())
    result = generator.sample()
    dataset = result.dataset
    assert dataset["t"].dims == ("time", "level", "latitude", "longitude")
    assert dataset["tcwv"].dims == ("time", "latitude", "longitude")
    assert dataset.sizes["level"] == 3
    assert np.all(np.isfinite(np.asarray(dataset["t"].values)))


def test_analog_keeps_temporal_persistence(tmp_path):
    """The generated series must contain *contiguous real blocks*.

    The synthetic time axis is a rearrangement of the real one, so the check is
    "the generated sequence equals a contiguous slice of the real sequence",
    not "the generated sequence matches the real one step by step".
    """

    real = _real()
    generator = AnalogGenerator(_config(tmp_path, patch_mode="whole_field"), seed=42)
    generator.fit(real)
    dataset = generator.sample().dataset
    generated = np.asarray(dataset["t"].isel(level=0, latitude=2, longitude=3).values, dtype=float)
    source = np.asarray(real["t"].isel(level=0, latitude=2, longitude=3).values, dtype=float)
    # every generated step must come from the real series (recombined, not invented)
    assert set(np.round(generated, 9)).issubset(set(np.round(source, 9)))
    # and a contiguous real block must appear inside the generated series
    block = source[:2]
    found = any(
        np.allclose(generated[start:start + 2], block)
        for start in range(len(generated) - 1)
    )
    assert found, (generated, source)


def test_analog_preserves_spatial_structure(tmp_path):
    """Each generated field equals a real field of the archive (whole-field mode)."""

    real = _real()
    generator = AnalogGenerator(_config(tmp_path, patch_mode="whole_field"), seed=42)
    generator.fit(real)
    generated = generator.sample().dataset
    real_fields = np.asarray(real["t"].isel(level=0).values, dtype=float)
    gen_fields = np.asarray(generated["t"].isel(level=0).values, dtype=float)
    matched = False
    for real_field in real_fields:
        for gen_field in gen_fields:
            if np.allclose(real_field, gen_field):
                matched = True
                break
        if matched:
            break
    assert matched, "no generated field matched a real field"


def test_analog_contract_declares_recombination_and_novelty(tmp_path):
    generator = AnalogGenerator(_config(tmp_path), seed=42)
    generator.fit(_real())
    result = generator.sample()
    contract = read_contract(result.dataset)
    generation = contract["generation_config"]
    assert generation["independence"] == "recombined"
    assert 0.0 <= generation["novelty"]["exact_copy_cell_fraction"] <= 1.0
    assert generation["method"]["kind"] == "structure_preserving_recombination"
    assert generation["method"]["patch_mode"] == "whole_field"
    assert contract["generation_method"] == "statistical"
    assert contract["model_name"] == "atmosgen.analog_blocks"


def test_analog_screening_is_reported(tmp_path):
    generator = AnalogGenerator(_config(tmp_path), seed=42)
    generator.fit(_real())
    result = generator.sample()
    screening = result.metadata["screening"]
    assert screening["total_samples"] > 0
    assert 0.0 <= screening["rejection_rate"] <= 1.0


def test_analog_jitter_reduces_exact_copies(tmp_path):
    """Jitter must break the exact-copy property (and be disclosed as such)."""

    plain = AnalogGenerator(_config(tmp_path, noise_jitter=0.0), seed=42)
    plain.fit(_real())
    plain_result = plain.sample()
    jittered = AnalogGenerator(_config(tmp_path, noise_jitter=0.01), seed=42)
    jittered.fit(_real())
    jittered_result = jittered.sample()
    assert (
        jittered_result.metadata["novelty"]["exact_copy_cell_fraction"]
        < plain_result.metadata["novelty"]["exact_copy_cell_fraction"]
    )


def test_analog_is_deterministic_for_a_fixed_seed(tmp_path):
    first = AnalogGenerator(_config(tmp_path), seed=7)
    first.fit(_real())
    second = AnalogGenerator(_config(tmp_path), seed=7)
    second.fit(_real())
    np.testing.assert_array_equal(
        np.asarray(first.sample().dataset["t"].values),
        np.asarray(second.sample().dataset["t"].values),
    )
