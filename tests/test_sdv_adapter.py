"""Adapters for SDV models (Test B): fit -> sample -> save -> reread."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from atmosgen.config import GenerationConfigFile
from atmosgen.contract import CONTRACT_ATTR, read_contract
from atmosgen.data.io import read_canonical, save_canonical
from atmosgen.generators.sdv_adapter import (
    CTGANGenerator,
    GaussianCopulaGenerator,
    TVAEGenerator,
    _filter_signature,
)

from conftest import build_small_dataset, make_contract

pytest.importorskip("sdv")


def _tiny_config(tmp_path: Path, real_path: Path, *, model: str, **samples) -> GenerationConfigFile:
    payload = {
        "project": {"name": "adapter", "seed": 11},
        "data": {"input": str(real_path), "variables": ["t", "q", "u", "v", "tcwv"]},
        "generation": {
            "mode": "statistical",
            "model": model,
            "samples": {"unit": "cell_time", **samples},
            "covariates": ["hour_sin", "level", "lat_norm", "lon_norm"],
        },
        "output": {"format": "netcdf", "directory": str(tmp_path / "output")},
    }
    if model == "ctgan":
        payload["generation"]["training"] = {"epochs": 2, "batch_size": 64, "discriminator_steps": 1}
    if model == "tvae":
        payload["generation"]["training"] = {"epochs": 2, "batch_size": 64}
    return GenerationConfigFile.model_validate(payload)


def test_gaussian_copula_fit_sample_contract(tmp_path):
    ds = build_small_dataset(seed=6)
    real_path = tmp_path / "real.nc"
    save_canonical(ds, real_path, contract=make_contract(ds))
    config = _tiny_config(tmp_path, real_path, model="gaussian_copula", rows=128)
    gen = GaussianCopulaGenerator(
        config,
        seed=11,
        config_path="configs/unit_test.yaml",
        config_sha256="0" * 64,
        trained_from={"dataset": "unit-test", "sha256": "1" * 64},
    )
    gen.fit(build_small_dataset(seed=6))
    result = gen.sample()
    dataset = result.dataset
    assert dataset["t"].dims == ("time", "level", "latitude", "longitude")
    assert dataset["tcwv"].dims == ("time", "latitude", "longitude")
    assert float(np.asarray(dataset["t"]).min()) >= float(ds["t"].min()) - 5.0
    assert float(np.asarray(dataset["t"]).max()) <= float(ds["t"].max()) + 5.0
    contract = read_contract(dataset)
    assert contract["generation_method"] == "statistical"
    assert contract["model_name"] == "sdv.GaussianCopulaSynthesizer"
    assert contract["source"]["kind"] == "synthetic"
    assert contract["source"]["trained_from"]["dataset"] == "unit-test"
    samples = contract["generation_config"]["samples"]
    assert samples["training_scheme"] == "level_wise"
    assert samples["rows_generated"] == 3 * 2 * 3 * 4  # grid rows (time*level*lat*lon)
    assert samples["levels"] == [500.0, 700.0]


def test_test_b_roundtrip_save_and_reread(tmp_path):
    """Test B: generate -> save -> reread equals the in-memory result bit-for-bit."""

    ds = build_small_dataset(seed=6)
    real_path = tmp_path / "real.nc"
    save_canonical(ds, real_path, contract=make_contract(ds))
    config = _tiny_config(tmp_path, real_path, model="gaussian_copula", rows=128)
    gen = GaussianCopulaGenerator(config, seed=11)
    gen.fit(build_small_dataset(seed=6))
    result = gen.sample()
    out_path = save_canonical(
        result.dataset,
        tmp_path / "data.nc",
        contract=result.contract,
        compression={"zlib": True, "complevel": 4},
    )
    reread = read_canonical(out_path)
    try:
        assert CONTRACT_ATTR in reread.attrs
        np.testing.assert_array_equal(
            np.asarray(reread["t"].values), np.asarray(result.dataset["t"].values)
        )
        np.testing.assert_array_equal(
            np.asarray(reread["q"].values), np.asarray(result.dataset["q"].values)
        )
        assert read_contract(reread)["model_name"] == result.contract["model_name"]
    finally:
        reread.close()


def test_ctgan_tiny_epochs_produces_grid(tmp_path):
    ds = build_small_dataset(seed=6)
    real_path = tmp_path / "real.nc"
    save_canonical(ds, real_path, contract=make_contract(ds))
    config = _tiny_config(tmp_path, real_path, model="ctgan", rows=128)
    gen = CTGANGenerator(config, seed=11)
    gen.fit(build_small_dataset(seed=6))
    result = gen.sample()
    assert result.dataset.sizes["time"] == 3
    assert np.isfinite(np.asarray(result.dataset["q"].values)).all()
    samples = read_contract(result.dataset)["generation_config"]["samples"]
    assert samples["rows_generated"] == 72
    assert samples["training_scheme"] == "level_wise"


def test_tvae_tiny_epochs_produces_grid(tmp_path):
    ds = build_small_dataset(seed=6)
    real_path = tmp_path / "real.nc"
    save_canonical(ds, real_path, contract=make_contract(ds))
    config = _tiny_config(tmp_path, real_path, model="tvae", rows=128)
    gen = TVAEGenerator(config, seed=11)
    gen.fit(build_small_dataset(seed=6))
    result = gen.sample()
    assert result.dataset.sizes["latitude"] == 3
    assert np.isfinite(np.asarray(result.dataset["t"].values)).all()


def test_sample_before_fit_raises():
    config = _tiny_config(Path("/tmp"), Path("/tmp") / "x", model="gaussian_copula", rows=8)
    gen = GaussianCopulaGenerator(config, seed=1)
    with pytest.raises(RuntimeError):
        gen.sample()


def test_filter_signature_records_unsupported_keys():
    def model(alpha: float = 1.0) -> None:  # pragma: no cover - signature only
        return None

    accepted, ignored = _filter_signature(model, {"alpha": 2.0, "beta": 3.0})
    assert accepted == {"alpha": 2.0}
    assert ignored == ["beta"]


def test_conditional_sample_hook(tmp_path):
    ds = build_small_dataset(seed=6)
    real_path = tmp_path / "real.nc"
    save_canonical(ds, real_path, contract=make_contract(ds))
    config = _tiny_config(tmp_path, real_path, model="gaussian_copula", rows=128)
    gen = GaussianCopulaGenerator(config, seed=11)
    gen.fit(build_small_dataset(seed=6))
    # level selects the level-wise model; no other conditions -> plain sample
    frame = gen.sample_conditional({"level": 500.0}, num_rows=3)
    assert len(frame) == 3 and set(frame.columns) >= {"t", "q", "u", "v", "tcwv"}
    # continuous conditioning is SDV's boundary-sensitive path (see adapter
    # docstring); the hook exists for Phase 4 and is exercised there, not in
    # the Phase 2 end-to-end path.  A wrong level must fail loudly instead.
    with pytest.raises(KeyError):
        gen.sample_conditional({"level": 123.0}, num_rows=1)
