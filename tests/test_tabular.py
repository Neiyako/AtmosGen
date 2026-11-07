"""Field<->table conversion (statistical Mode 1) unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from atmosgen.data.tabular import (
    SUPPORTED_COVARIATES,
    TableError,
    build_grid,
    dataset_to_table,
    rows_to_fields,
    table_feature_count,
    thin_grid,
)

from conftest import build_small_dataset


def test_roundtrip_reconstructs_fields_exactly():
    # 3 levels like production: 4D variables must be bit-exact, while the 3-D
    # tcwv is recovered as a mean over the level axis and may differ by 1 ULP.
    ds = build_small_dataset(seed=3, nlevel=3)
    X, names, meta = dataset_to_table(
        ds, variables=["t", "q", "u", "v", "tcwv"], covariates=["hour_sin", "level", "lat_norm", "lon_norm"]
    )
    assert meta["rows_total"] == meta["rows"] == 3 * 3 * 3 * 4
    assert meta["removed_nan_rows"] == 0
    rebuilt = rows_to_fields(X, meta)
    for name in ["t", "q", "u", "v"]:
        np.testing.assert_allclose(
            np.asarray(rebuilt[name].values), np.asarray(ds[name].values), rtol=0, atol=0
        )
    np.testing.assert_allclose(
        np.asarray(rebuilt["tcwv"].values),
        np.asarray(ds["tcwv"].values),
        rtol=0,
        atol=1e-12,  # 1 ULP scale: tcwv is averaged over the level axis
    )


def test_row_order_matches_raster_semantics():
    ds = build_small_dataset(seed=1)
    X, names, meta = dataset_to_table(ds, variables=["t"], covariates=["level"])
    level_column = X[:, names.index("level")]
    grid = meta["grid"]
    n_lat, n_lon, n_level = len(grid["latitude"]), len(grid["longitude"]), len(grid["level"])
    block = n_lat * n_lon
    # time outermost, level advances every lat*lon block, lat every lon step
    for t in range(3):
        for li, level in enumerate(grid["level"]):
            block_slice = slice((t * 2 + li) * block, (t * 2 + li) * block + block)
            np.testing.assert_allclose(level_column[block_slice], float(level), rtol=0)
    assert meta["lat_for_rows"][:block] == [float(v) for v in np.repeat(grid["latitude"], n_lon)]


def test_covariate_values_follow_expected_formulas():
    ds = build_small_dataset(seed=3)  # times 00/06/12
    X, names, meta = dataset_to_table(
        ds, variables=["t"], covariates=["hour_sin", "hour_cos", "level", "lat_norm", "lon_norm"]
    )
    tidx = np.asarray(meta["time_index_for_rows"])
    hour_sin = np.column_stack([X[tidx == t, names.index("hour_sin")][:1] for t in range(3)])[0]
    np.testing.assert_allclose(hour_sin, [0.0, 1.0, np.sin(2 * np.pi * 12 / 24) or 0.0 + np.sin(np.pi)], atol=1e-12)
    # normalised coordinates stay within [-1, 1]
    for name in ("lat_norm", "lon_norm"):
        assert X[:, names.index(name)].min() >= -1.0 - 1e-12
        assert X[:, names.index(name)].max() <= 1.0 + 1e-12
    level_column = X[:, names.index("level")]
    np.testing.assert_allclose(level_column[:12], 500.0, rtol=0)
    np.testing.assert_allclose(level_column[12:24], 700.0, rtol=0)


def test_nan_rows_are_dropped_from_training_table():
    ds = build_small_dataset(seed=3)
    ds["t"][0, 0, 0, :] = np.nan
    X, names, meta = dataset_to_table(ds, variables=["t"], covariates=[])
    assert meta["removed_nan_rows"] == 4  # one full lon row at the NaN cell
    assert meta["rows"] == meta["rows_total"] - 4
    assert not np.isnan(X).any()


def test_thin_grid_matches_step_request():
    ds = build_small_dataset(nlat=101, nlon=101)
    lat_values = np.arange(45.0, -10.0, -0.25)  # descending, like ERA5
    lat_values = lat_values[:101] if lat_values.size >= 101 else lat_values
    ds = ds.assign_coords(latitude=np.linspace(45.0, 20.0, 101))
    ds = ds.assign_coords(longitude=np.linspace(100.0, 125.0, 101))
    thin = thin_grid(ds, lat_step_deg=1.0, lon_step_deg=1.0)
    assert thin.sizes["latitude"] == 26 and thin.sizes["longitude"] == 26
    np.testing.assert_allclose(thin["latitude"].values, ds["latitude"].values[::4], rtol=0)
    np.testing.assert_allclose(thin["longitude"].values, ds["longitude"].values[::4], rtol=0)
    # no-op when steps equal grid spacing
    same = thin_grid(ds, lat_step_deg=0.25, lon_step_deg=0.25)
    assert same.sizes == ds.sizes


def test_unknown_covariate_and_variable_raise():
    ds = build_small_dataset(seed=0)
    with pytest.raises(TableError):
        dataset_to_table(ds, variables=["t"], covariates=["moon_phase"])
    with pytest.raises(TableError):
        dataset_to_table(ds, variables=["t", "not_a_var"])


def test_generated_rows_fill_grid_and_remaining_features_are_ignored():
    ds = build_small_dataset(seed=0)
    X, names, meta = dataset_to_table(ds, variables=["t", "q"], covariates=["hour_sin"])
    fake = np.repeat(X[:, [names.index("t"), names.index("q")]], 1, axis=0)
    built = rows_to_fields(fake, meta)
    assert built["t"].dims == ("time", "level", "latitude", "longitude")
    # extra rows beyond the grid are truncated, missing rows raise
    with pytest.raises(TableError):
        rows_to_fields(fake[:-1], meta)
    more = np.vstack([fake, fake])
    built_more = rows_to_fields(more, meta)
    np.testing.assert_allclose(
        np.asarray(built_more["t"].values), np.asarray(built["t"].values), rtol=0
    )


def test_table_feature_count_consistency():
    ds = build_small_dataset(seed=0)
    grid = build_grid(ds)
    assert table_feature_count(grid) == 3 * 2 * 3 * 4


def test_supported_covariates_are_documented():
    assert set(SUPPORTED_COVARIATES) == {"hour_sin", "hour_cos", "level", "lat_norm", "lon_norm"}
