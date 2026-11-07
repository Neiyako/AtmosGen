"""Ingest pipeline: tcwv derivation, local ERA5 slice, CDS-style sample."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from atmosgen.config import IngestConfigFile, load_config_file
from atmosgen.contract import read_contract
from atmosgen.data.io import file_fingerprint, open_source
from atmosgen.data.preprocess import derive_tcwv, ingest

GRAVITY = 9.80665


def _tiny_era5_file(path: Path, varname: str, *, seed: int, values: str = "dynamic") -> None:
    """One ERA5-like daily file: 4 times x 3 levels x 3 lat x 4 lon."""

    rng = np.random.default_rng(seed)
    times = np.arange("2020-05-01T00", "2020-05-02T00", dtype="datetime64[6h]").astype("datetime64[ns]")
    levels = np.array([500.0, 700.0, 850.0])
    lats = np.array([40.0, 30.0, 20.0])  # descending, like ERA5
    lons = np.array([100.0, 105.0, 110.0, 115.0])
    shape = (times.size, levels.size, lats.size, lons.size)
    if values == "positive":
        data = 1e-3 * rng.random(shape)
    elif values == "warm":
        data = 250.0 + 10.0 * rng.random(shape)
    else:
        data = rng.normal(0.0, 5.0, shape)
    ds = xr.Dataset(
        {
            varname: (
                ("time", "level", "latitude", "longitude"),
                data,
                {"units": "kg kg**-1" if values == "positive" else "K", "long_name": varname},
            )
        },
        coords={
            "time": times,
            "level": (("level",), levels, {"units": "hPa"}),
            "latitude": (("latitude",), lats, {"units": "degrees_north"}),
            "longitude": (("longitude",), lons, {"units": "degrees_east"}),
        },
    )
    ds.to_netcdf(path, engine="netcdf4")


def test_derive_tcwv_matches_analytic_constant_profile():
    q = xr.DataArray(
        np.full((2, 3, 4, 5), 0.01),
        dims=("time", "level", "latitude", "longitude"),
        coords={"level": [500.0, 700.0, 850.0]},
        name="q",
    )
    tcwv = derive_tcwv(q)
    expected = 0.01 * (850.0 - 500.0) * 100.0 / GRAVITY
    np.testing.assert_allclose(tcwv.values, expected, rtol=1e-12)
    assert tcwv.dims == ("time", "latitude", "longitude")
    assert tcwv.attrs["units"] == "kg m**-2"


def test_derive_tcwv_is_level_order_independent():
    q = xr.DataArray(
        np.linspace(0.001, 0.02, 3)[None, :, None, None] * np.ones((1, 3, 2, 2)),
        dims=("time", "level", "latitude", "longitude"),
        coords={"level": [500.0, 700.0, 850.0]},
        name="q",
    )
    ascending = derive_tcwv(q)
    descending = derive_tcwv(q.isel(level=[2, 1, 0]).assign_coords(level=[850.0, 700.0, 500.0]))
    np.testing.assert_allclose(ascending.values, descending.values, rtol=1e-12)


def test_ingest_local_era5_fake_archive(tmp_path, write_yaml):
    archive = tmp_path / "archive"
    day = date(2020, 5, 1)
    (archive / "2020").mkdir(parents=True)
    mapping = {
        "t": "temperature",
        "q": "specific_humidity",
        "u": "u_component_of_wind",
        "v": "v_component_of_wind",
    }
    for index, (canonical, segment) in enumerate(mapping.items()):
        _tiny_era5_file(
            archive / "2020" / f"ERA5_025dg_20200501_{segment}.nc",
            canonical,
            seed=index,
            values="positive" if canonical == "q" else "dynamic",
        )
    config_path = write_yaml(
        "era5.yaml",
        {
            "project": {"name": "fake_ingest", "seed": 3, "output_dir": str(tmp_path / "output")},
            "source": {
                "kind": "local_era5",
                "root": str(archive),
                "date_range": ["2020-05-01", "2020-05-01"],
                "file_pattern": "ERA5_025dg_{date:%Y%m%d}_{variable}.nc",
                "variable_map": mapping,
                "subset": {
                    "levels": [500.0, 700.0],
                    "latitude": [20.0, 30.0],
                    "longitude": [100.0, 110.0],
                    "time_step_hours": 6,
                },
                "derive": {"tcwv": {"method": "vertical_integral_q"}},
            },
            "output": {
                "raw_path": str(tmp_path / "raw" / "slice.nc"),
                "processed_path": str(tmp_path / "processed" / "era5.nc"),
                "format": "netcdf",
            },
        },
    )
    config = load_config_file(config_path)
    assert isinstance(config, IngestConfigFile)
    report = ingest(config, config_path=config_path)
    assert report.status == "PASS", report.checks
    assert report.processed_path is not None and report.processed_path.exists()
    assert report.raw_path is not None and report.raw_path.exists()
    assert report.raw_path != report.processed_path

    ds = open_source(report.processed_path)
    try:
        assert set(ds.data_vars) == {"t", "q", "u", "v", "tcwv"}
        assert ds.sizes["time"] == 4
        assert ds.sizes["level"] == 2
        assert ds.sizes["latitude"] == 2
        assert ds.sizes["longitude"] == 3
        assert float(ds["tcwv"].min()) >= 0.0
        contract = read_contract(ds)
        assert contract["generation_method"] == "ingest"
        assert contract["source"]["kind"] == "real"
        assert "tcwv" in contract["source"]["derivation"]
        assert contract["coordinates"]["time"]["step_hours"] == 6.0
    finally:
        ds.close()
    # raw slice has no derived variable and is smaller
    raw = open_source(report.raw_path)
    try:
        assert "tcwv" not in raw.data_vars
    finally:
        raw.close()
    # source archive untouched
    for path in archive.rglob("*.nc"):
        assert path.stat().st_size > 0
    assert report.readonly["all_unchanged"] is True


def test_ingest_local_netcdf_cloud_sample(tmp_path, write_yaml):
    rng = np.random.default_rng(0)
    source = tmp_path / "cloud.nc"
    ds = xr.Dataset(
        {
            "clwc": (
                ("valid_time", "pressure_level", "latitude", "longitude"),
                1e-4 * rng.random((1, 3, 2, 3)),
                {"units": "kg kg**-1", "long_name": "Specific cloud liquid water content"},
            ),
            "ciwc": (
                ("valid_time", "pressure_level", "latitude", "longitude"),
                5e-5 * rng.random((1, 3, 2, 3)),
                {"units": "kg kg**-1", "long_name": "Specific cloud ice water content"},
            ),
        },
        coords={
            "valid_time": (("valid_time",), np.array(["2024-01-01T00"], dtype="datetime64[ns]")),
            "pressure_level": (("pressure_level",), [500.0, 400.0, 300.0], {"units": "hPa"}),
            "latitude": (("latitude",), [25.0, 30.0], {"units": "degrees_north"}),
            "longitude": (("longitude",), [70.0, 75.0, 80.0], {"units": "degrees_east"}),
        },
    )
    ds.to_netcdf(source, engine="netcdf4")
    before = file_fingerprint(source, with_sha256=True)
    config_path = write_yaml(
        "cloud.yaml",
        {
            "project": {"name": "cloud", "seed": 1, "output_dir": str(tmp_path / "output")},
            "source": {
                "kind": "local_netcdf",
                "input": str(source),
                "variable_map": {"clwc": "clwc", "ciwc": "ciwc"},
            },
            "output": {"processed_path": str(tmp_path / "processed" / "cloud.nc")},
        },
    )
    report = ingest(load_config_file(config_path), config_path=config_path)
    assert report.status == "PASS", report.errors if hasattr(report, "errors") else report.checks
    after = file_fingerprint(source, with_sha256=True)
    assert before == after
    out = open_source(report.processed_path)
    try:
        assert set(out.data_vars) == {"clwc", "ciwc"}
        assert out.sizes["time"] == 1
        assert out.sizes["level"] == 3
        contract = read_contract(out)
        assert contract["variables"] == ["clwc", "ciwc"]
    finally:
        out.close()
