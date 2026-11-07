"""Canonical NetCDF/Zarr IO, raw-data write guard and read-only sources."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from atmosgen.contract import CONTRACT_ATTR, ContractError, read_contract
from atmosgen.data import io
from atmosgen.data.io import (
    RawWriteError,
    default_raw_roots,
    open_source,
    read_canonical,
    save_canonical,
)

from conftest import make_contract  # noqa: E402  (tests dir is on sys.path)


def test_netcdf_roundtrip_preserves_values_and_contract(small_ds, saved_dataset):
    path = saved_dataset(small_ds, "roundtrip.nc")
    assert path.exists()
    ds = read_canonical(path)
    try:
        assert set(ds.data_vars) == set(small_ds.data_vars)
        for name in small_ds.data_vars:
            np.testing.assert_allclose(
                np.asarray(ds[name].values), np.asarray(small_ds[name].values), rtol=0, atol=0
            )
        doc = read_contract(ds)
        assert doc["generation_method"] == "statistical"
    finally:
        ds.close()
    # sidecar contract.json is written next to the data file and also validates
    from atmosgen.contract import read_contract_file

    sidecar = read_contract_file(io.sidecar_contract_path(path))
    assert sidecar["variables"] == doc["variables"]


def test_refuses_to_write_under_raw_dir(tmp_path, small_ds, monkeypatch):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    monkeypatch.setenv("ATMOSGEN_RAW_DIRS", str(raw_root))
    assert any(io.is_within(raw_root / "x.nc", root) for root in default_raw_roots())
    with pytest.raises(RawWriteError):
        save_canonical(small_ds, raw_root / "slice.nc", contract=make_contract(small_ds))
    assert not (raw_root / "slice.nc").exists()


def test_allow_raw_only_for_ingest(tmp_path, small_ds, monkeypatch):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    monkeypatch.setenv("ATMOSGEN_RAW_DIRS", str(raw_root))
    path = save_canonical(
        small_ds, raw_root / "slice.nc", contract=make_contract(small_ds), allow_raw=True
    )
    assert path.exists()


def test_overwrite_requires_explicit_flag(tmp_path, small_ds, saved_dataset):
    path = saved_dataset(small_ds, "existing.nc")
    with pytest.raises(FileExistsError):
        save_canonical(small_ds, path, contract=make_contract(small_ds))
    replaced = save_canonical(small_ds, path, contract=make_contract(small_ds), overwrite=True)
    assert replaced.exists()


def test_open_source_snapshot_does_not_touch_source(small_ds, saved_dataset):
    """Reading a source must not modify it (the write guard is save_canonical's job)."""

    path = saved_dataset(small_ds, "source.nc")
    before = io.file_fingerprint(path, with_sha256=True)
    ds = open_source(path)
    try:
        assert "t" in ds.data_vars
        # reading through the dask/backend layer must not rewrite the file
        _ = float(ds["t"].isel(time=0, level=0, latitude=0, longitude=0).values)
    finally:
        ds.close()
    after = io.file_fingerprint(path, with_sha256=True)
    assert before == after


def test_sidecar_does_not_collide_within_one_directory(tmp_path, small_ds):
    first = save_canonical(small_ds, tmp_path / "a.nc", contract=make_contract(small_ds))
    second = save_canonical(small_ds, tmp_path / "b.nc", contract=make_contract(small_ds))
    from atmosgen.contract import read_contract_file

    sidecar_a = io.sidecar_contract_path(first)
    sidecar_b = io.sidecar_contract_path(second)
    assert sidecar_a != sidecar_b
    assert sidecar_a.exists() and sidecar_b.exists()
    assert read_contract_file(sidecar_a)["variables"] == ["t", "q", "u", "v", "tcwv"]
    # data.nc keeps the schema-mandated contract.json name
    run_like = save_canonical(small_ds, tmp_path / "run" / "data.nc", contract=make_contract(small_ds))
    assert io.sidecar_contract_path(run_like) == tmp_path / "run" / "contract.json"
    # same stem, different format must also stay distinct
    zarr_like = save_canonical(
        small_ds, tmp_path / "a.zarr", contract=make_contract(small_ds), format="zarr"
    )
    assert io.sidecar_contract_path(zarr_like) != sidecar_a
    # data.nc keeps contract.json but data.zarr must not overwrite it
    run_dir = tmp_path / "run2"
    nc_like = save_canonical(small_ds, run_dir / "data.nc", contract=make_contract(small_ds))
    zarr_data = save_canonical(
        small_ds, run_dir / "data.zarr", contract=make_contract(small_ds), format="zarr"
    )
    assert io.sidecar_contract_path(nc_like) == run_dir / "contract.json"
    assert io.sidecar_contract_path(zarr_data) == run_dir / "data.zarr.contract.json"


def test_canonical_save_requires_contract(tmp_path, small_ds):
    with pytest.raises(ContractError):
        save_canonical(small_ds, tmp_path / "no_contract.nc")


def test_zarr_roundtrip(tmp_path, small_ds):
    path = save_canonical(
        small_ds, tmp_path / "data.zarr", contract=make_contract(small_ds), format="zarr"
    )
    ds = read_canonical(path, format="zarr")
    try:
        assert set(ds.data_vars) == set(small_ds.data_vars)
        np.testing.assert_allclose(np.asarray(ds["t"].values), np.asarray(small_ds["t"].values))
        assert CONTRACT_ATTR in ds.attrs
    finally:
        ds.close()


def test_detect_format(tmp_path):
    assert io.detect_format(tmp_path / "a.nc") == "netcdf"
    assert io.detect_format(tmp_path / "a.zarr") == "zarr"
    with pytest.raises(ValueError):
        io.detect_format(tmp_path / "a.unknown")
