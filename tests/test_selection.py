"""Subset selection (time / level / spatial extent) unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from atmosgen.data.selection import SubsetError, select_subset, subset_summary

from conftest import build_small_dataset


def test_select_variables_and_time():
    ds = build_small_dataset(seed=0)
    out = select_subset(
        ds,
        variables=["t", "q"],
        time_range=("2020-01-01T00", "2020-01-01T06"),
    )
    assert list(out.data_vars) == ["t", "q"]
    assert out.sizes["time"] == 2


def test_select_levels_and_extent_respects_direction():
    # coarse synthetic grid, then mirror ERA5's descending latitude
    ds = build_small_dataset(nlat=9, nlon=7)
    ds = ds.isel(latitude=np.arange(ds.sizes["latitude"] - 1, -1, -1))
    lat = np.asarray(ds["latitude"].values)
    assert lat[0] > lat[-1]
    out = select_subset(
        ds,
        levels=[500.0],
        spatial_extent={"lat": [float(lat.min()), float(lat.max())], "lon": [100.0, 110.0]},
    )
    assert out.sizes["level"] == 1
    assert out.sizes["latitude"] == ds.sizes["latitude"]
    assert float(out["longitude"].min()) == 100.0
    assert float(out["longitude"].max()) <= 110.0  # grid may not contain 110.0 exactly


def test_out_of_range_requests_raise_with_available_info():
    ds = build_small_dataset(seed=0)
    with pytest.raises(SubsetError) as excinfo:
        select_subset(ds, variables=["t", "missing_var"])
    assert "missing_var" in str(excinfo.value)
    with pytest.raises(SubsetError) as excinfo:
        select_subset(ds, levels=[300.0])
    assert "300.0" in str(excinfo.value)
    with pytest.raises(SubsetError) as excinfo:
        select_subset(ds, time_range=("2019-01-01", "2019-01-02"))
    assert "2019-01-01" in str(excinfo.value)
    with pytest.raises(SubsetError) as excinfo:
        select_subset(ds, spatial_extent={"lat": [0.0, 10.0]})
    assert "spatial_extent.lat" in str(excinfo.value)


def test_subset_summary_shapes():
    ds = build_small_dataset(seed=0)
    summary = subset_summary(ds)
    assert summary["variables"] == ["q", "t", "tcwv", "u", "v"]
    assert summary["time"]["n"] == 3
    assert summary["level"]["range"] == [500.0, 700.0]
