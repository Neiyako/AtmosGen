"""Shared fixtures for the AtmosGen test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pytest
import xarray as xr

from atmosgen.contract import build_contract
from atmosgen.data.io import save_canonical


def build_small_dataset(
    *,
    ntime: int = 3,
    nlevel: int = 2,
    nlat: int = 3,
    nlon: int = 4,
    seed: int = 0,
    with_tcwv: bool = True,
    with_level: bool = True,
) -> xr.Dataset:
    """Small canonical dataset used by unit tests (no real data needed)."""

    rng = np.random.default_rng(seed)
    times = np.array(
        ["2020-01-01T00", "2020-01-01T06", "2020-01-01T12", "2020-01-01T18"],
        dtype="datetime64[ns]",
    )[:ntime]
    levels = np.array([500.0, 700.0, 850.0])[:nlevel]
    lats = np.array([20.0, 25.0, 30.0, 40.0])[:nlat]
    lons = np.array([100.0, 105.0, 110.0, 115.0])[:nlon]
    coords: dict[str, Any] = {"time": times, "latitude": lats, "longitude": lons}
    if with_level:
        coords["level"] = levels
    dims4 = ("time", "level", "latitude", "longitude") if with_level else None
    shape4 = (ntime, nlevel, nlat, nlon) if with_level else (ntime, nlat, nlon)

    ds = xr.Dataset(coords=coords)
    ds["t"] = (dims4, 250.0 + 20.0 * rng.random(shape4), {"units": "K", "long_name": "Temperature"})
    ds["q"] = (dims4, 1e-3 * rng.random(shape4), {"units": "kg kg**-1", "long_name": "Specific humidity"})
    ds["u"] = (dims4, 10.0 * (rng.random(shape4) - 0.5), {"units": "m s**-1", "long_name": "U wind"})
    ds["v"] = (dims4, 10.0 * (rng.random(shape4) - 0.5), {"units": "m s**-1", "long_name": "V wind"})
    if with_tcwv and with_level:
        ds["tcwv"] = (
            ("time", "latitude", "longitude"),
            20.0 + 5.0 * rng.random((ntime, nlat, nlon)),
            {"units": "kg m**-2", "long_name": "Total column water vapour"},
        )
    ds["level"].attrs["units"] = "hPa"
    ds["latitude"].attrs["units"] = "degrees_north"
    ds["longitude"].attrs["units"] = "degrees_east"
    return ds


def make_contract(ds: xr.Dataset, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generation_method": "statistical",
        "source": {"kind": "synthetic", "dataset": "unit-test", "readonly": True},
        "model_name": "sdv.GaussianCopulaSynthesizer",
        "model_version": "2.1.0",
        "random_seed": 42,
    }
    payload.update(overrides)
    return build_contract(ds, **payload)


@pytest.fixture()
def small_ds() -> xr.Dataset:
    return build_small_dataset()


@pytest.fixture()
def make_dataset() -> Callable[..., xr.Dataset]:
    return build_small_dataset


@pytest.fixture()
def contract_factory() -> Callable[..., dict[str, Any]]:
    return make_contract


@pytest.fixture()
def saved_dataset(tmp_path: Path) -> Callable[[xr.Dataset, str], Path]:
    """Save a dataset as canonical NetCDF inside ``tmp_path``."""

    def _save(ds: xr.Dataset, name: str = "data.nc", **kwargs: Any) -> Path:
        target = tmp_path / name
        contract = kwargs.pop("contract", None) or make_contract(ds)
        return save_canonical(ds, target, contract=contract, **kwargs)

    return _save


@pytest.fixture()
def write_yaml(tmp_path: Path) -> Callable[[str, Mapping[str, Any]], Path]:
    """Write a mapping as a YAML file inside ``tmp_path``."""

    import yaml

    def _write(name: str, payload: Mapping[str, Any]) -> Path:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False), encoding="utf-8")
        return target

    return _write
