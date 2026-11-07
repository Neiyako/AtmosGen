"""Field <-> table conversion for statistical generation (Mode 1, plan §4.1).

A ``cell_time`` table has one row per ``(time, level, latitude, longitude)``
grid cell; ``tcwv`` (3-D, column integral) is broadcast across the level axis
of each cell and reconstructed by averaging the level axis back out.

The generated table rows are assigned to real grid coordinates in raster order
``(time, level, latitude, longitude)`` — this is the documented reconstruction
step; it does *not* create spatio-temporal coherence (plan §4.1 acceptance).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import xarray as xr

SUPPORTED_COVARIATES: tuple[str, ...] = (
    "hour_sin",
    "hour_cos",
    "level",
    "lat_norm",
    "lon_norm",
)


class TableError(ValueError):
    """Raised for invalid field/table conversion requests."""


def thin_grid(ds: xr.Dataset, *, lat_step_deg: float = 1.0, lon_step_deg: float = 1.0) -> xr.Dataset:
    """Thin a canonical dataset onto a coarser grid (``data.subsample``)."""

    def steps_for(coord: str, step_deg: float) -> int:
        if step_deg <= 0:
            raise TableError(f"subsample 步长必须 > 0，得到 {step_deg}")
        values = np.asarray(ds[coord].values, dtype=float)
        if values.size < 2:
            return 1
        grid_step = float(np.median(np.abs(np.diff(values))))
        if grid_step <= 0:
            return 1
        return max(1, int(round(step_deg / grid_step)))

    lat_steps = steps_for("latitude", lat_step_deg) if "latitude" in ds.coords else 1
    lon_steps = steps_for("longitude", lon_step_deg) if "longitude" in ds.coords else 1
    if lat_steps == 1 and lon_steps == 1:
        return ds
    return ds.isel(
        latitude=slice(None, None, lat_steps),
        longitude=slice(None, None, lon_steps),
    )


def build_grid(ds: xr.Dataset) -> dict[str, np.ndarray]:
    """Coordinate arrays (as-is, preserving direction) for table assembly."""

    grid: dict[str, np.ndarray] = {}
    for name in ("time", "level", "latitude", "longitude"):
        if name in ds.coords:
            grid[name] = np.asarray(ds[name].values)
    for required in ("time", "latitude", "longitude"):
        if required not in grid:
            raise TableError(f"数据缺少 {required} 坐标，无法构造表格")
    return grid


def _covariate_values(name: str, *, times: np.ndarray, level: np.ndarray | None,
                      lat: np.ndarray, lon: np.ndarray, n_level: int) -> np.ndarray:
    """Compute covariate columns broadcast to the (time, level, lat, lon) grid."""

    def norm(v: np.ndarray) -> np.ndarray:
        vmin, vmax = float(v.min()), float(v.max())
        if vmax == vmin:
            return np.zeros_like(v, dtype=float)
        return (v - vmin) / (vmax - vmin) * 2.0 - 1.0

    lat_grid = norm(lat).reshape(1, 1, len(lat), 1)
    lon_grid = norm(lon).reshape(1, 1, 1, len(lon))
    hours = times.astype("datetime64[h]").astype(int) % 24
    hour_sin = np.sin(2.0 * np.pi * hours / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hours / 24.0)
    shape = (len(times), n_level, len(lat), len(lon))

    if name in {"hour_sin", "hour_cos"}:
        base = hour_sin if name == "hour_sin" else hour_cos
        return np.broadcast_to(base.reshape(-1, 1, 1, 1), shape)
    if name == "level":
        if level is None:
            raise TableError("数据没有 level 维，无法提供 level 协变量")
        return np.broadcast_to(
            np.asarray(level, dtype=float)[None, :, None, None], shape
        )
    if name == "lat_norm":
        return np.broadcast_to(lat_grid, shape)
    if name == "lon_norm":
        return np.broadcast_to(lon_grid, shape)
    raise TableError(f"不支持的协变量 {name!r}；可用: {list(SUPPORTED_COVARIATES)}")


def dataset_to_table(
    ds: xr.Dataset,
    *,
    variables: Sequence[str],
    covariates: Sequence[str] = (),
) -> tuple[np.ndarray, list[str], dict]:
    """Flatten ``ds`` into a ``(rows, features)`` matrix for SDV.

    Returns ``(X, feature_names, meta)``. ``meta`` carries the grid and both
    the row count and the number of NaN-dropped rows so the reverse transform
    can rebuild fields and the run report can document data loss.
    """

    unknown = [name for name in variables if name not in ds.data_vars]
    if unknown:
        raise TableError(
            f"变量 {unknown} 不在数据集中；可用: {sorted(ds.data_vars)}"
        )
    unknown_cov = [name for name in covariates if name not in SUPPORTED_COVARIATES]
    if unknown_cov:
        raise TableError(
            f"不支持的协变量 {unknown_cov}；可用: {list(SUPPORTED_COVARIATES)}"
        )

    grid = build_grid(ds)
    times = grid["time"]
    levels = np.asarray(grid.get("level"), dtype=float) if "level" in grid else None
    lat = np.asarray(grid["latitude"], dtype=float)
    lon = np.asarray(grid["longitude"], dtype=float)
    n_time, n_lat, n_lon = times.size, lat.size, lon.size
    n_level = 1 if levels is None else levels.size
    rows_total = n_time * n_level * n_lat * n_lon

    # Stack variables in raster order (time, level, lat, lon).
    variable_values: dict[str, np.ndarray] = {}
    for name in variables:
        values = np.asarray(ds[name].values, dtype=float)
        if "level" not in grid:
            expected = (n_time, n_lat, n_lon)
            if values.shape != expected:
                raise TableError(
                    f"变量 {name} 形状 {values.shape} 与网格 {expected} 不符"
                )
            values = values[:, None, :, :]  # broadcast onto the (absent) level axis
        elif values.shape == (n_time, n_lat, n_lon):
            # 3-D column integral (tcwv): broadcast across the level axis so a
            # cell_time row carries the column value; reconstruction averages
            # the level axis back out, which is exact for constant-across-level.
            values = np.broadcast_to(values[:, None, :, :], (n_time, n_level, n_lat, n_lon))
        elif values.shape != (n_time, n_level, n_lat, n_lon):
            raise TableError(
                f"变量 {name} 形状 {values.shape} 与网格 ({n_time}, {n_level}, {n_lat}, {n_lon}) 不符"
            )
        variable_values[name] = values.reshape(-1)

    feature_names = list(variables)
    feature_arrays = [variable_values[name] for name in variables]
    for name in covariates:
        cov = _covariate_values(
            name,
            times=times,
            level=levels,
            lat=np.asarray(grid["latitude"], dtype=float),
            lon=np.asarray(grid["longitude"], dtype=float),
            n_level=n_level,
        )
        feature_names.append(name)
        feature_arrays.append(cov.reshape(-1))

    level_for_rows = (
        # level advances within each time block:  [l0*lats*lons, l1*lats*lons] per time
        np.tile(np.repeat(np.asarray(levels, dtype=float), n_lat * n_lon), n_time)
        if levels is not None
        else np.zeros(rows_total)
    )
    lat_for_rows = np.tile(np.repeat(lat, n_lon), n_time * n_level)
    lon_for_rows = np.tile(lon, n_time * n_level * n_lat)
    time_index_for_rows = np.repeat(np.arange(n_time), n_level * n_lat * n_lon)

    # Drop NaN rows before fitting (retained NaN rows would poison the model).
    X = np.column_stack(feature_arrays)
    finite_mask = np.isfinite(X).all(axis=1)
    dropped = int(rows_total - int(finite_mask.sum()))
    X = X[finite_mask]
    meta = {
        "rows": int(X.shape[0]),
        "rows_total": int(rows_total),
        "removed_nan_rows": dropped,
        "removed_nan_fraction": round(dropped / rows_total, 6) if rows_total else 0.0,
        "features": feature_names,
        "variables": list(variables),
        "covariates": list(covariates),
        "grid": {
            "time": times.tolist(),
            "level": levels.tolist() if levels is not None else None,
            "latitude": np.asarray(grid["latitude"], dtype=float).tolist(),
            "longitude": np.asarray(grid["longitude"], dtype=float).tolist(),
        },
        "row_order": "time, level, latitude, longitude",
        # Full-length coordinate arrays: generated rows fill the whole grid,
        # so reconstruction always uses the unmasked raster-order arrays.
        "time_index_for_rows": time_index_for_rows.tolist(),
        "level_for_rows": level_for_rows.tolist(),
        "lat_for_rows": lat_for_rows.tolist(),
        "lon_for_rows": lon_for_rows.tolist(),
    }
    if dropped:
        meta["note"] = f"removed {dropped} rows with NaN features from the TRAINING table"
    return X, feature_names, meta


def table_feature_count(grid: Mapping[str, np.ndarray]) -> int:
    """Exact number of rows the raster-order grid produces."""

    return int(grid["time"].size) * (
        1 if "level" not in grid else np.asarray(grid["level"]).size
    ) * np.asarray(grid["latitude"]).size * np.asarray(grid["longitude"]).size


def rows_to_fields(
    rows: np.ndarray,
    meta: Mapping,
    variables: Sequence[str] | None = None,
) -> xr.Dataset:
    """Rebuild a canonical dataset from generated rows.

    Rows are assigned to grid coordinates in raster order ``(time, level,
    latitude, longitude)``; 3-D variables (e.g. ``tcwv``) are averaged over the
    level axis after reshape, which is exact for real data (the column
    integral is constant across levels) and the only available choice for
    generated rows.
    """

    names = list(variables if variables is not None else meta["variables"])
    positions = {name: index for index, name in enumerate(meta["features"])}
    missing = [name for name in names if name not in positions]
    if missing:
        raise TableError(f"生成行缺少变量 {missing}；features={meta['features']}")

    grid = meta["grid"]
    times = np.asarray(grid["time"], dtype="datetime64[ns]")
    lat = np.asarray(grid["latitude"], dtype=float)
    lon = np.asarray(grid["longitude"], dtype=float)
    has_level = grid.get("level") is not None
    levels = np.asarray(grid["level"], dtype=float) if has_level else None
    n_time, n_lat, n_lon = times.size, lat.size, lon.size
    n_level = 1 if levels is None else levels.size

    if rows.shape[0] < n_time * n_level * n_lat * n_lon:
        raise TableError(
            f"生成行数 {rows.shape[0]} 不足以填满网格 "
            f"{n_time}x{n_level}x{n_lat}x{n_lon}={n_time * n_level * n_lat * n_lon}"
        )

    dataset = xr.Dataset(
        coords={
            "time": times,
            **({"level": levels, } if has_level else {}),
            "latitude": lat,
            "longitude": lon,
        }
    )
    for name in names:
        column = np.asarray(rows[:, positions[name]], dtype=float)
        values = column[: n_time * n_level * n_lat * n_lon].reshape(
            n_time, n_level, n_lat, n_lon
        )
        if not has_level:
            data_dims = ("time", "latitude", "longitude")
            data_3d = values[:, 0, :, :]
        elif name in ("tcwv",) or values.shape[1] == 1:
            data_dims = ("time", "latitude", "longitude")
            data_3d = values.mean(axis=1)
        else:
            data_3d = values
            data_dims = ("time", "level", "latitude", "longitude")
        dataset[name] = xr.DataArray(data_3d, dims=data_dims)

    for name in names:
        spec_meta = {
            "t": ("K", "Temperature"),
            "q": ("kg kg**-1", "Specific humidity"),
            "u": ("m s**-1", "U component of wind"),
            "v": ("m s**-1", "V component of wind"),
            "tcwv": ("kg m**-2", "Total column water vapour (generated)"),
        }.get(name)
        if spec_meta:
            dataset[name].attrs.setdefault("units", spec_meta[0])
            dataset[name].attrs.setdefault("long_name", spec_meta[1])
    # NOTE: do not set a `calendar` attribute on `time` here — xarray's
    # netCDF writer already injects one from its default CF encoding, and a
    # duplicate raises "Key 'calendar' already exists in attrs" on save.
    return dataset


__all__ = [
    "SUPPORTED_COVARIATES",
    "TableError",
    "build_grid",
    "dataset_to_table",
    "rows_to_fields",
    "table_feature_count",
    "thin_grid",
]
