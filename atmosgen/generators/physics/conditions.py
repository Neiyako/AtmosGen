"""Initial / boundary conditions for the 2D advection-diffusion solver (plan §4.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import xarray as xr

METERS_PER_DEGREE = 111_320.0


class ConditionError(ValueError):
    """Raised when an initial/boundary condition cannot be realised."""


@dataclass(frozen=True)
class GridGeometry:
    """Physical geometry of the solver grid plus its lat/lon labels."""

    nx: int
    ny: int
    dx_m: float
    dy_m: float
    latitude: np.ndarray
    longitude: np.ndarray
    center_lat: float
    center_lon: float

    @property
    def dlat_deg(self) -> float:
        return self.dy_m / METERS_PER_DEGREE

    @property
    def dlon_deg(self) -> float:
        return self.dx_m / (METERS_PER_DEGREE * np.cos(np.radians(self.center_lat)))

    def x_m(self) -> np.ndarray:
        return (self.longitude - self.center_lon) * METERS_PER_DEGREE * np.cos(
            np.radians(self.center_lat)
        )

    def y_m(self) -> np.ndarray:
        return (self.latitude - self.center_lat) * METERS_PER_DEGREE


def build_geometry(
    *,
    nx: int,
    ny: int,
    dx_m: float,
    dy_m: float,
    center: tuple[float, float] | None = None,
    real: xr.Dataset | None = None,
) -> GridGeometry:
    """Build the lat/lon labels of the solver grid.

    ``gaussian_plume`` conditions supply a centre; ``from_real`` conditions take
    the real dataset's lat/lon extent instead.
    """

    if real is not None and "latitude" in real.coords and "longitude" in real.coords:
        lats = np.asarray(real["latitude"].values, dtype=float)
        lons = np.asarray(real["longitude"].values, dtype=float)
        latitude = np.sort(np.linspace(lats.min(), lats.max(), ny))
        longitude = np.sort(np.linspace(lons.min(), lons.max(), nx))
        return GridGeometry(
            nx=nx,
            ny=ny,
            dx_m=dx_m,
            dy_m=dy_m,
            latitude=latitude,
            longitude=longitude,
            center_lat=float(np.mean(latitude)),
            center_lon=float(np.mean(longitude)),
        )
    if center is None:
        raise ConditionError("initial_condition.kind=gaussian_plume 需要 center=[lat, lon]")
    center_lat, center_lon = float(center[0]), float(center[1])
    dlat = dy_m / METERS_PER_DEGREE
    dlon = dx_m / (METERS_PER_DEGREE * np.cos(np.radians(center_lat)))
    latitude = center_lat + (np.arange(ny) - (ny - 1) / 2.0) * dlat
    longitude = center_lon + (np.arange(nx) - (nx - 1) / 2.0) * dlon
    return GridGeometry(
        nx=nx,
        ny=ny,
        dx_m=dx_m,
        dy_m=dy_m,
        latitude=latitude,
        longitude=longitude,
        center_lat=center_lat,
        center_lon=center_lon,
    )


def gaussian_plume(
    geometry: GridGeometry,
    *,
    sigma_deg: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Analytic Gaussian plume centred on the grid centre, shape ``(ny, nx)``."""

    if sigma_deg <= 0:
        raise ConditionError(f"sigma_deg 必须 > 0，得到 {sigma_deg}")
    sigma_m = float(sigma_deg) * METERS_PER_DEGREE
    y = geometry.y_m()[:, None]
    x = geometry.x_m()[None, :]
    return float(amplitude) * np.exp(-(x**2 + y**2) / (2.0 * sigma_m**2))


def field_from_real(
    real: xr.Dataset,
    variable: str,
    geometry: GridGeometry,
    *,
    amplitude: float = 1.0,
    time_index: int = 0,
) -> np.ndarray:
    """Interpolate a real field onto the solver grid (linear, nearest fallback)."""

    if variable not in real.data_vars:
        raise ConditionError(
            f"initial_condition.variable={variable!r} 不在真实数据中；可用: {list(real.data_vars)}"
        )
    field = real[variable]
    if "time" in field.dims:
        field = field.isel(time=time_index)
    if "level" in field.dims:
        field = field.isel(level=0)
    if "latitude" not in field.dims or "longitude" not in field.dims:
        raise ConditionError(
            f"变量 {variable!r} 需要 latitude/longitude 维度，得到 {field.dims}"
        )
    # target length check: interp handles ordering, but our labels must be sorted
    linear = field.interp(latitude=geometry.latitude, longitude=geometry.longitude, method="linear")
    values = np.asarray(linear.values, dtype=float)
    if not np.all(np.isfinite(values)):
        nearest = field.reindex(
            latitude=geometry.latitude, longitude=geometry.longitude, method="nearest"
        )
        nearest_values = np.asarray(nearest.values, dtype=float)
        values = np.where(np.isfinite(values), values, nearest_values)
    return float(amplitude) * values


def build_initial_condition(
    config: Mapping[str, Any],
    *,
    geometry: GridGeometry,
    real: xr.Dataset | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``(field, details)`` for the configured initial condition."""

    kind = config.get("kind")
    metadata: dict[str, Any] = {"kind": kind, "grid": {"nx": geometry.nx, "ny": geometry.ny}}
    if kind == "gaussian_plume":
        center = config.get("center")
        sigma = config.get("sigma_deg")
        amplitude = float(config.get("amplitude", 1.0))
        if center is None or sigma is None:
            raise ConditionError("gaussian_plume 需要 center 与 sigma_deg")
        geometry_with_center = build_geometry(
            nx=geometry.nx,
            ny=geometry.ny,
            dx_m=geometry.dx_m,
            dy_m=geometry.dy_m,
            center=(float(center[0]), float(center[1])),
        )
        field = gaussian_plume(
            geometry_with_center, sigma_deg=float(sigma), amplitude=amplitude
        )
        metadata.update(
            {
                "center": [float(center[0]), float(center[1])],
                "sigma_deg": float(sigma),
                "amplitude": amplitude,
            }
        )
        return field, metadata
    if kind == "from_real":
        variable = config.get("variable")
        if variable is None:
            raise ConditionError("from_real 需要 variable")
        if real is None:
            raise ConditionError("from_real 需要 data 段提供真实场")
        amplitude = float(config.get("amplitude", 1.0))
        field = field_from_real(real, str(variable), geometry, amplitude=amplitude)
        metadata.update({"variable": variable, "amplitude": amplitude})
        return field, metadata
    raise ConditionError(f"未知 initial_condition.kind={kind!r}（允许 gaussian_plume / from_real）")


__all__ = [
    "ConditionError",
    "GridGeometry",
    "METERS_PER_DEGREE",
    "build_geometry",
    "build_initial_condition",
    "field_from_real",
    "gaussian_plume",
]
