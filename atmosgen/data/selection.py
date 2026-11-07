"""Selection of time / level / spatial subsets from a canonical dataset.

Phase 2's ``atmosgen generate`` reads ``data.input`` and crops it with the
``data:`` section of the generation config before fitting; the raw archive and
the canonical file always stay untouched.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import xarray as xr


class SubsetError(ValueError):
    """Raised when a requested subset does not exist in the dataset."""


def select_subset(
    ds: xr.Dataset,
    *,
    variables: Sequence[str] | None = None,
    time_range: Sequence[str] | None = None,
    levels: Sequence[float] | None = None,
    spatial_extent: Mapping[str, Sequence[float]] | None = None,
) -> xr.Dataset:
    """Return a new dataset restricted to the requested variables and ranges.

    Raises :class:`SubsetError` with a repair hint when a requested element is
    missing — this never silently drops part of the request.
    """

    problems: list[str] = []
    out = ds

    if variables:
        missing = [name for name in variables if name not in ds.data_vars]
        if missing:
            problems.append(
                f"variables {missing} 在输入数据中不存在；可用: {sorted(ds.data_vars)}"
            )
        else:
            out = out[list(variables)]

    if time_range is not None:
        try:
            start, end = (np.datetime64(value) for value in time_range)
        except Exception:
            problems.append(f"time_range {list(time_range)} 无法解析为时间戳")
        else:
            if end < start:
                problems.append(f"time_range 结束早于起始: {list(time_range)}")
            elif "time" not in out.coords:
                problems.append("输入数据没有 time 坐标")
            else:
                available = np.asarray(out["time"].values)
                if start < available.min() or end > available.max():
                    problems.append(
                        f"time_range {list(time_range)} 超出数据范围 "
                        f"[{available.min()}, {available.max()}]"
                    )
                else:
                    out = out.sel(time=slice(start, end))

    if levels is not None:
        if "level" not in out.coords:
            problems.append("请求了 levels 但输入数据没有 level 坐标")
        else:
            available = {round(float(value), 6): value for value in out["level"].values}
            missing_levels = [value for value in levels if round(float(value), 6) not in available]
            if missing_levels:
                problems.append(
                    f"levels {missing_levels} 不存在；可用层: {[float(v) for v in sorted(available)]}"
                )
            else:
                out = out.sel(level=[available[round(float(value), 6)] for value in levels])

    if spatial_extent is not None:
        if "lat" in spatial_extent:
            out = _slice_axis(out, "latitude", spatial_extent["lat"], "lat", problems)
        if "lon" in spatial_extent:
            out = _slice_axis(out, "longitude", spatial_extent["lon"], "lon", problems)

    if problems:
        raise SubsetError("; ".join(problems))
    return out


def _slice_axis(
    ds: xr.Dataset,
    coord_name: str,
    bounds: Sequence[float],
    label: str,
    problems: list[str],
) -> xr.Dataset:
    low, high = (float(bounds[0]), float(bounds[1]))
    if low > high:
        problems.append(f"spatial_extent.{label} 下限 {low} 大于上限 {high}")
        return ds
    values = np.asarray(ds[coord_name].values, dtype=float)
    if low < values.min() or high > values.max():
        problems.append(
            f"spatial_extent.{label} {list(bounds)} 超出数据范围 "
            f"[{float(values.min())}, {float(values.max())}]"
        )
        return ds
    descending = values.size > 1 and values[0] > values[-1]
    return ds.sel(**{coord_name: slice(high, low) if descending else slice(low, high)})


def subset_summary(ds: xr.Dataset) -> dict:
    """Compact description of a selected dataset (run metadata)."""

    summary: dict = {"variables": sorted(ds.data_vars)}
    for name in ("time", "level", "latitude", "longitude"):
        if name in ds.coords:
            values = ds[name].values
            if np.asarray(values).dtype.kind == "M":
                summary[name] = {
                    "n": int(values.size),
                    "start": str(np.datetime_as_string(values.min(), unit="s")),
                    "end": str(np.datetime_as_string(values.max(), unit="s")),
                }
            else:
                summary[name] = {
                    "n": int(values.size),
                    "range": [float(values.min()), float(values.max())],
                }
    return summary


__all__ = ["SubsetError", "select_subset", "subset_summary"]
