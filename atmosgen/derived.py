"""Derived-diagnostics engine: build the diagnostic variables from the state.

This module implements *strategy B* of the variable-system extension: derived
variables are **not** generated independently — they are computed from the
already-available state variables with explicit physical relations, so the
delivered dataset is self-consistent by construction.

Relations implemented (all of them are the definitions declared in
:mod:`atmosgen.variables`):

============================  ==============================================
variable                      relation
============================  ==============================================
``tcwv``                      ``(1/g) * integral(q dp)``
``tclw`` / ``cwp``            ``(1/g) * integral(clwc dp)``
``tciw`` / ``iwp``            ``(1/g) * integral(ciwc dp)``
``twp``                       ``cwp + iwp``
``rh`` / ``r``                ``100 * q / qsat(t, p)``  (Tetens)
``qflux_u`` / ``qflux_v``     ``q * u`` / ``q * v``
``qflux_u_col``               ``(1/g) * integral(qflux_u dp)``
``qflux_v_col``               ``(1/g) * integral(qflux_v dp)``
``mfc``                       ``-div(qflux_u_col, qflux_v_col)``  (column)
``mfc_sl``                    ``-div(qflux_u, qflux_v)`` at one level
============================  ==============================================

Conventions that the engine enforces (and never mixes up):

* A **column** integral uses ``(1/g) * integral(f dp)`` on the native pressure
  levels; the result is a kg m-2 style column quantity, never a mixing ratio.
* ``mfc`` is the **column-integrated** moisture flux convergence
  (kg m-2 s-1) and ``mfc_sl`` the **single-level** diagnostic (kg kg-1 s-1).
  They are separate variables with separate units, so the two physical
  meanings can never be confused.
* Supersaturation is **reported, not removed**: the relative humidity is
  returned exactly as computed.

Nothing here invents data. When a dependency is missing the corresponding
variable is reported as ``NOT_AVAILABLE`` with the reason — it is never
defaulted to zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import xarray as xr

from .variables import VAR_SPECS, spec_for, transitive_dependencies

__all__ = [
    "DerivedError",
    "GRAVITY",
    "column_integral",
    "compute_derived",
    "horizontal_divergence",
    "level_pressure_pa",
    "moisture_flux_convergence",
    "relative_humidity",
    "saturation_specific_humidity",
]

#: Standard gravity used for every column integral (m s-2).
GRAVITY = 9.80665

#: 1 degree of latitude in metres (same constant as the physical validator).
_DEG_M = 111320.0

#: Candidate names of the vertical coordinate.
_LEVEL_NAMES = ("level", "pressure_level", "plev", "isobaricInhPa", "isobaricinhpa")

#: Recognised pressure-level units and their factor to Pa.
_HPA_UNITS = ("hpa", "hectopascal", "hectopascals", "mb", "mbar", "millibar", "millibars")
_PA_UNITS = ("pa", "pascal", "pascals")

#: Tetens coefficients (identical to the ones used by the physical validator,
#: so the saturation check and the relative humidity agree exactly).
_TETENS_A = 6.112  # hPa
_TETENS_B = 17.67
_TETENS_C = 29.65
_KELVIN_OFFSET = 273.15

#: Order in which the derived variables are computed (dependencies first).
_COMPUTE_ORDER: tuple[str, ...] = (
    "tcwv",
    "tclw",
    "tciw",
    "cwp",
    "iwp",
    "twp",
    "rh",
    "r",
    "qflux_u",
    "qflux_v",
    "qflux_u_col",
    "qflux_v_col",
    "mfc",
    "mfc_sl",
)


class DerivedError(ValueError):
    """Raised when a derived variable cannot be computed at all."""


# --------------------------------------------------------------------------- #
# Level handling
# --------------------------------------------------------------------------- #
def level_pressure_pa(ds: xr.Dataset) -> tuple[np.ndarray, str]:
    """Return ``(levels_pa, coord_name)`` of the vertical coordinate.

    Raises :class:`DerivedError` when the dataset has no usable pressure
    coordinate — the caller turns that into a ``NOT_AVAILABLE`` report instead
    of inventing a value.
    """

    name = next((n for n in _LEVEL_NAMES if n in ds.coords or n in ds.variables), None)
    if name is None:
        raise DerivedError(
            "数据集无 level 坐标（已查 level/pressure_level/plev），"
            "无法进行垂直积分或按气压计算相对湿度"
        )
    units = ""
    if name in ds.coords:
        units = str(ds.coords[name].attrs.get("units") or "")
    if not units:
        units = str(ds.attrs.get("level_units") or "")
    norm = units.strip().lower().replace(" ", "")
    if norm in _HPA_UNITS or "hpa" in norm or "hectopascal" in norm:
        factor = 100.0
    elif norm in _PA_UNITS or norm.startswith("pa") or "pascal" in norm:
        factor = 1.0
    else:
        raise DerivedError(
            f"level 单位 {units!r} 无法识别为气压层（需 hPa/mb/millibars/Pa）"
        )
    raw = np.asarray(ds[name].values, dtype=float).ravel()
    if raw.size < 2:
        raise DerivedError(f"level 坐标只有 {raw.size} 个值，无法进行垂直积分")
    if not np.all(np.isfinite(raw)):
        raise DerivedError("level 坐标含非有限值")
    levels_pa = raw * factor
    if np.any(levels_pa <= 0):
        raise DerivedError("level 坐标换算后存在 ≤0 的气压值")
    return levels_pa, name


def _surface_level_index(levels_pa: np.ndarray) -> int:
    """Index of the level closest to the surface (largest pressure)."""

    return int(np.argmax(levels_pa))


# --------------------------------------------------------------------------- #
# Physics helpers
# --------------------------------------------------------------------------- #
def saturation_specific_humidity(
    temperature: np.ndarray, pressure_hpa: np.ndarray | float
) -> np.ndarray:
    """Saturation specific humidity (kg/kg) with the Tetens formula.

    ``temperature`` is in K and ``pressure_hpa`` in hPa. Where the denominator
    ``p - 0.378 e_s`` is not positive the result is NaN (undefined), never a
    clipped substitute.
    """

    t = np.asarray(temperature, dtype=float)
    p = np.asarray(pressure_hpa, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        es = _TETENS_A * np.exp(_TETENS_B * (t - _KELVIN_OFFSET) / (t - _TETENS_C))
        denom = p - 0.378 * es
        ok = np.isfinite(es) & np.isfinite(denom) & (denom > 0)
        qsat = np.where(ok, 0.622 * es / np.where(ok, denom, 1.0), np.nan)
    return qsat


def relative_humidity(
    temperature: xr.DataArray,
    specific_humidity: xr.DataArray,
    levels_pa: np.ndarray,
    *,
    gravity: float = GRAVITY,
) -> xr.DataArray:
    """Relative humidity in % from ``t``, ``q`` and the pressure levels.

    The value is reported exactly as computed — supersaturation is **not**
    clipped. The level axis is broadcast over the ``level`` dimension only.
    """

    t = temperature
    q = specific_humidity
    if "level" not in t.dims or "level" not in q.dims:
        raise DerivedError("相对湿度需要 t 与 q 的 level 维")
    if t.sizes["level"] != q.sizes["level"]:
        raise DerivedError("相对湿度需要 t 与 q 的 level 维长度一致")
    level_axis = t.get_axis_num("level")
    shape = [1] * t.ndim
    shape[level_axis] = levels_pa.size
    p_hpa = (levels_pa / 100.0).reshape(shape)
    qsat = xr.DataArray(
        saturation_specific_humidity(np.asarray(t.values, dtype=float), p_hpa),
        dims=t.dims,
        coords={d: t.coords[d] for d in t.dims if d in t.coords},
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        rh = 100.0 * q / qsat
    rh = rh.where(np.isfinite(rh))
    out = rh.transpose(*t.dims)
    out.attrs.update(
        {
            "units": "%",
            "long_name": "Relative humidity",
            "standard_name": "relative_humidity",
            "atmosgen_derived": "rh = 100 * q / qsat(t, p)  [Tetens]",
            "atmosgen_supersaturation_policy": (
                "reported as computed; values > 100 % are supersaturation "
                "(physically possible in cloud) and are never clipped"
            ),
        }
    )
    return out


def column_integral(
    field: xr.DataArray,
    levels_pa: np.ndarray,
    *,
    gravity: float = GRAVITY,
) -> xr.DataArray:
    """Column integral ``(1/g) * integral(f dp)`` of a level-dependent field.

    The trapezoidal rule is applied on the pressure levels sorted ascending,
    which is the same convention as ``atmosgen.data.preprocess.derive_tcwv``
    (the result is a positive column quantity).
    """

    if "level" not in field.dims:
        raise DerivedError("垂直积分需要 level 维")
    n_levels = int(field.sizes["level"])
    if levels_pa.size != n_levels:
        raise DerivedError(
            f"level 坐标长度 {levels_pa.size} 与场的 level 维 {n_levels} 不一致"
        )
    order = np.argsort(levels_pa)
    levels_sorted = levels_pa[order]
    thickness = np.diff(levels_sorted)
    if np.any(thickness <= 0):
        raise DerivedError("气压层必须互不相同才能做梯形积分")

    values = np.asarray(field.values, dtype=float)
    axis = field.get_axis_num("level")
    values = np.moveaxis(values, axis, 0)[order]
    weight = thickness.reshape((-1,) + (1,) * (values.ndim - 1))
    with np.errstate(invalid="ignore"):
        integral = (0.5 * (values[:-1] + values[1:]) * weight).sum(axis=0)
    column = integral / gravity
    out_dims = tuple(dim for dim in field.dims if dim != "level")
    coords = {dim: field.coords[dim] for dim in out_dims if dim in field.coords}
    return xr.DataArray(column, dims=out_dims, coords=coords)


def _as_latlon(array: xr.DataArray) -> tuple[np.ndarray, bool]:
    """Return ``(values with lat/lon last, had_time)``."""

    dims = list(array.dims)
    if "latitude" not in dims or "longitude" not in dims:
        raise DerivedError("散度计算需要 latitude/longitude 维")
    order = [d for d in dims if d not in ("latitude", "longitude")]
    order += ["latitude", "longitude"]
    transposed = array.transpose(*order)
    return np.asarray(transposed.values, dtype=float), "time" in dims


def _d_dx(array: np.ndarray, dx_row: np.ndarray) -> np.ndarray:
    """∂/∂x with a per-latitude spacing (metres); central + one-sided edges.

    ``array`` has the latitude and longitude axes last; ``dx_row`` holds the
    zonal spacing in metres for every latitude row.
    """

    out = np.empty_like(array)
    spacing = dx_row[None, :, None]  # (1, nlat, 1) — broadcasts over longitude
    out[..., :, 1:-1] = (array[..., :, 2:] - array[..., :, :-2]) / (2.0 * spacing)
    # the edge slices drop the longitude axis, so they use the (nlat,) spacing
    out[..., :, 0] = (array[..., :, 1] - array[..., :, 0]) / dx_row
    out[..., :, -1] = (array[..., :, -1] - array[..., :, -2]) / dx_row
    return out


def _d_dy(array: np.ndarray, dy_m: float) -> np.ndarray:
    """∂/∂y with a constant spacing (metres); central + one-sided edges."""

    out = np.empty_like(array)
    out[..., 1:-1, :] = (array[..., 2:, :] - array[..., :-2, :]) / (2.0 * dy_m)
    out[..., 0, :] = (array[..., 1, :] - array[..., 0, :]) / dy_m
    out[..., -1, :] = (array[..., -1, :] - array[..., -2, :]) / dy_m
    return out


def _grid_metric(ds: xr.Dataset) -> tuple[np.ndarray, float]:
    """Return ``(dx_row_m, dy_m)`` from the latitude/longitude coordinates."""

    lat = np.asarray(ds["latitude"].values, dtype=float).ravel()
    lon = np.asarray(ds["longitude"].values, dtype=float).ravel()
    if lat.size < 2 or lon.size < 2:
        raise DerivedError("经纬度网格至少需要 2×2 个格点")
    dlat = float(np.mean(np.abs(np.diff(lat))))
    dlon = float(np.mean(np.abs(np.diff(lon))))
    if not (dlat > 0 and dlon > 0):
        raise DerivedError("经纬度网格间距为 0，无法换算米制")
    dy_m = _DEG_M * dlat
    dx_row = _DEG_M * np.cos(np.deg2rad(lat)) * dlon
    dx_row = np.maximum(dx_row, 1e-6)
    return dx_row, dy_m


def horizontal_divergence(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Horizontal divergence of two 2-D fields, in s**-1.

    ``u`` and ``v`` must have ``latitude``/``longitude`` dimensions; the
    latitude/longitude coordinates of ``u`` define the metric
    (``dx = 111320 * cos(lat) * dlon`` m, ``dy = 111320 * dlat`` m). Central
    differences are used in the interior and one-sided differences on the
    edges.
    """

    if set(u.dims) != set(v.dims):
        raise DerivedError(
            f"散度计算需要 u 与 v 的维度一致：{sorted(u.dims)} vs {sorted(v.dims)}"
        )
    u_values, had_time = _as_latlon(u)
    v_values, _ = _as_latlon(v)
    ds = u._to_temp_dataset() if hasattr(u, "_to_temp_dataset") else None  # pragma: no cover
    lat = np.asarray(u.coords["latitude"].values, dtype=float).ravel()
    lon = np.asarray(u.coords["longitude"].values, dtype=float).ravel()
    if lat.size < 2 or lon.size < 2:
        raise DerivedError("经纬度网格至少需要 2×2 个格点")
    dlat = float(np.mean(np.abs(np.diff(lat))))
    dlon = float(np.mean(np.abs(np.diff(lon))))
    if not (dlat > 0 and dlon > 0):
        raise DerivedError("经纬度网格间距为 0，无法换算米制")
    dy_m = _DEG_M * dlat
    dx_row = np.maximum(_DEG_M * np.cos(np.deg2rad(lat)) * dlon, 1e-6)
    divergence = _d_dx(u_values, dx_row) + _d_dy(v_values, dy_m)
    dims = [d for d in u.dims if d not in ("latitude", "longitude")]
    dims += ["latitude", "longitude"]
    coords = {d: u.coords[d] for d in dims if d in u.coords}
    out = xr.DataArray(divergence, dims=dims, coords=coords)
    out.attrs.update({"units": "s**-1", "long_name": "Horizontal divergence"})
    return out


def moisture_flux_convergence(
    flux_u: xr.DataArray, flux_v: xr.DataArray, *, ds: xr.Dataset | None = None
) -> xr.DataArray:
    """``-div(flux_u, flux_v)`` — the moisture flux convergence of a flux pair."""

    divergence = horizontal_divergence(flux_u, flux_v)
    out = -divergence
    out.attrs.update({"long_name": "Moisture flux convergence"})
    return out


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
@dataclass
class _Workspace:
    """Mutable working set used while the derived variables are computed."""

    dataset: xr.Dataset
    gravity: float = GRAVITY
    cache: dict[str, xr.DataArray] = field(default_factory=dict)
    levels_pa: np.ndarray | None = None
    level_name: str | None = None
    level_error: str | None = None
    notes: list[str] = field(default_factory=list)

    # -- levels ---------------------------------------------------------- #
    def levels(self) -> tuple[np.ndarray, str]:
        if self.levels_pa is None and self.level_error is None:
            try:
                self.levels_pa, self.level_name = level_pressure_pa(self.dataset)
            except DerivedError as exc:
                self.level_error = str(exc)
        if self.levels_pa is None:
            raise DerivedError(self.level_error or "level 坐标不可用")
        return self.levels_pa, self.level_name or "level"

    # -- variables ------------------------------------------------------- #
    def has(self, name: str) -> bool:
        return name in self.dataset.data_vars or name in self.cache

    def get(self, name: str) -> xr.DataArray:
        if name in self.cache:
            return self.cache[name]
        if name in self.dataset.data_vars:
            return self.dataset[name]
        raise DerivedError(f"变量 {name!r} 不可用")

    def put(self, name: str, array: xr.DataArray) -> None:
        self.cache[name] = array


def _column_of(
    workspace: _Workspace,
    source: str,
    *,
    units: str,
    long_name: str,
    standard_name: str | None,
    formula: str,
    source_variable: str,
) -> xr.DataArray:
    """Compute ``(1/g) * integral(source dp)`` with the registry metadata."""

    levels_pa, _ = workspace.levels()
    field = workspace.get(source)
    column = column_integral(field, levels_pa, gravity=workspace.gravity)
    spec = spec_for(source_variable)
    attrs: dict[str, Any] = {
            "units": units,
            "long_name": long_name,
            "atmosgen_derived": formula,
            "atmosgen_dependencies": source,
            "levels_used": (
                f"{levels_pa.min() / 100:.0f}..{levels_pa.max() / 100:.0f} hPa "
                f"({levels_pa.size} levels)"
            ),
            "atmosgen_integration": "(1/g) * trapezoidal integral over pressure levels",
    }
    if standard_name:
        attrs["standard_name"] = standard_name
    column.attrs.update(attrs)
    _ = spec  # the registry entry documents the dependency chain
    return column


def _compute_one(workspace: _Workspace, name: str) -> xr.DataArray:
    """Compute a single derived variable from the workspace."""

    spec = spec_for(name)
    if name in ("tcwv", "tclw", "tciw", "cwp", "iwp"):
        source = {"tcwv": "q", "tclw": "clwc", "tciw": "ciwc", "cwp": "clwc", "iwp": "ciwc"}[name]
        return _column_of(
            workspace,
            source,
            units=spec.units,
            long_name=spec.long_name,
            standard_name=spec.standard_name,
            formula=spec.formula,
            source_variable=name,
        )
    if name == "twp":
        cwp = workspace.get("cwp")
        iwp = workspace.get("iwp")
        total = cwp + iwp
        total.attrs.update(
            {
                "units": spec.units,
                "long_name": spec.long_name,
                "standard_name": spec.standard_name,
                "atmosgen_derived": spec.formula,
                "atmosgen_dependencies": "cwp, iwp",
                "atmosgen_note": spec.comment,
            }
        )
        return total.transpose(*cwp.dims)
    if name in ("rh", "r"):
        levels_pa, _ = workspace.levels()
        t = workspace.get("t")
        q = workspace.get("q")
        rh = relative_humidity(t, q, levels_pa, gravity=workspace.gravity)
        rh.attrs["long_name"] = spec.long_name
        rh.attrs["atmosgen_dependencies"] = "t, q"
        return rh
    if name in ("qflux_u", "qflux_v"):
        source = "u" if name == "qflux_u" else "v"
        q = workspace.get("q")
        wind = workspace.get(source)
        flux = q * wind
        flux = flux.transpose(*q.dims)
        flux.attrs.update(
            {
                "units": spec.units,
                "long_name": spec.long_name,
                "standard_name": spec.standard_name,
                "atmosgen_derived": spec.formula,
                "atmosgen_dependencies": f"q, {source}",
                "atmosgen_note": spec.comment,
            }
        )
        return flux
    if name in ("qflux_u_col", "qflux_v_col"):
        source = "qflux_u" if name == "qflux_u_col" else "qflux_v"
        if not workspace.has(source):
            # compute the per-level flux on the fly
            workspace.put(source, _compute_one(workspace, source))
        return _column_of(
            workspace,
            source,
            units=spec.units,
            long_name=spec.long_name,
            standard_name=spec.standard_name,
            formula=spec.formula,
            source_variable=name,
        )
    if name == "mfc":
        for source in ("qflux_u_col", "qflux_v_col"):
            if not workspace.has(source):
                workspace.put(source, _compute_one(workspace, source))
        flux_u = workspace.get("qflux_u_col")
        flux_v = workspace.get("qflux_v_col")
        mfc = moisture_flux_convergence(flux_u, flux_v)
        mfc.attrs.update(
            {
                "units": spec.units,
                "long_name": spec.long_name,
                "standard_name": spec.standard_name,
                "atmosgen_derived": spec.formula,
                "atmosgen_dependencies": "qflux_u_col, qflux_v_col",
                "atmosgen_mfc_type": "column_integrated",
                "atmosgen_note": spec.comment,
            }
        )
        return mfc
    if name == "mfc_sl":
        for source in ("qflux_u", "qflux_v"):
            if not workspace.has(source):
                workspace.put(source, _compute_one(workspace, source))
        levels_pa, _ = workspace.levels()
        index = _surface_level_index(levels_pa)
        flux_u = workspace.get("qflux_u").isel(level=index)
        flux_v = workspace.get("qflux_v").isel(level=index)
        mfc = moisture_flux_convergence(flux_u, flux_v)
        mfc.attrs.update(
            {
                "units": spec.units,
                "long_name": spec.long_name,
                "standard_name": spec.standard_name,
                "atmosgen_derived": spec.formula,
                "atmosgen_dependencies": "qflux_u, qflux_v",
                "atmosgen_mfc_type": "single_level",
                "atmosgen_level_used": f"{levels_pa[index] / 100:.0f} hPa",
                "atmosgen_note": spec.comment,
            }
        )
        return mfc
    raise DerivedError(f"没有为变量 {name!r} 实现派生公式")


def _missing_dependencies(name: str, workspace: _Workspace) -> list[str]:
    """Return the *primary* dependencies of ``name`` missing from the workspace.

    Derived dependencies (e.g. ``qflux_u_col`` for ``mfc``) are deliberately not
    reported: the engine computes them on demand from the state variables, so a
    missing intermediate is not a hard blocker. Only a missing *state* variable
    (t/q/u/v/clwc/...) is.
    """

    closure = transitive_dependencies([name])
    missing: list[str] = []
    for dependency in closure.get(name, ()):
        if spec_for(dependency).is_derived:
            continue  # computed on demand, never a blocker
        if not workspace.has(dependency):
            missing.append(dependency)
    return missing


def compute_derived(
    dataset: xr.Dataset,
    *,
    variables: Sequence[str] | None = None,
    gravity: float = GRAVITY,
    force: bool = False,
) -> tuple[xr.Dataset, dict[str, Any]]:
    """Compute the requested derived variables and return ``(dataset, report)``.

    Parameters
    ----------
    dataset:
        The canonical dataset holding the state variables.
    variables:
        The derived variables to compute. ``None`` means "every derived
        variable in the registry". Variables that are already present in the
        dataset are **kept** (an ingested or jointly generated value is
        authoritative) unless ``force=True``.
    force:
        Recompute variables that are already present.

    The report maps every requested variable to a status:

    ``COMPUTED``
        the variable was derived from its dependencies;
    ``KEPT``
        the variable was already present and was not overwritten;
    ``NOT_AVAILABLE``
        the dependencies are missing — the reason is given, and the variable
        is **not** written with a placeholder value.
    """

    wanted = list(variables) if variables is not None else list(_COMPUTE_ORDER)
    unknown = [name for name in wanted if name not in VAR_SPECS]
    if unknown:
        raise DerivedError(f"未知的派生变量 {unknown}")
    not_derived = [
        name for name in wanted if name in VAR_SPECS and not spec_for(name).is_derived
    ]
    if not_derived:
        raise DerivedError(
            f"compute_derived 只能计算派生变量，收到主变量 {not_derived}"
        )

    workspace = _Workspace(dataset=dataset, gravity=gravity)
    status: dict[str, str] = {}
    computed: dict[str, str] = {}
    kept: dict[str, str] = {}
    unavailable: dict[str, str] = {}

    # Process in dependency order so that a derived dependency (e.g. twp needs
    # cwp/iwp) is available when it is needed.
    ordered = [name for name in _COMPUTE_ORDER if name in set(wanted)]
    ordered += [name for name in wanted if name not in ordered]

    for name in ordered:
        if not force and name in dataset.data_vars:
            status[name] = "KEPT"
            kept[name] = "变量已存在于数据集中，未覆盖（ingested/联合生成的值具有权威性）"
            continue
        missing = _missing_dependencies(name, workspace)
        if missing:
            status[name] = "NOT_AVAILABLE"
            unavailable[name] = (
                f"缺少依赖变量 {missing}；未写入占位值（不以 0 代替缺失量）"
            )
            continue
        try:
            array = _compute_one(workspace, name)
        except DerivedError as exc:
            status[name] = "NOT_AVAILABLE"
            unavailable[name] = str(exc)
            continue
        spec = spec_for(name)
        array = array.transpose(*spec.dims)
        attrs = dict(array.attrs)
        attrs["units"] = spec.units
        attrs.setdefault("long_name", spec.long_name)
        if spec.standard_name:
            attrs.setdefault("standard_name", spec.standard_name)
        if "standard_name" in attrs and attrs["standard_name"] is None:
            del attrs["standard_name"]  # None is not serialisable to NetCDF
        array.attrs = attrs
        workspace.put(name, array)
        status[name] = "COMPUTED"
        computed[name] = spec.formula or spec.long_name

    out = dataset
    for name, array in workspace.cache.items():
        if name in dataset.data_vars:
            continue
        out = out.assign(**{name: array})

    report: dict[str, Any] = {
        "status": status,
        "computed": computed,
        "kept": kept,
        "unavailable": unavailable,
        "formulas": {
            name: spec_for(name).formula for name in ordered if spec_for(name).formula
        },
        "notes": list(workspace.notes),
    }
    if workspace.level_name:
        report["level_coordinate"] = workspace.level_name
    if workspace.levels_pa is not None:
        report["levels_used"] = (
            f"{workspace.levels_pa.min() / 100:.0f}..{workspace.levels_pa.max() / 100:.0f} hPa "
            f"({workspace.levels_pa.size} levels)"
        )
    return out, report


def dependency_report(
    requested: Iterable[str], available: Iterable[str]
) -> dict[str, list[str]]:
    """Return, per requested variable, the dependencies missing from ``available``."""

    have = set(available)
    closure = transitive_dependencies(list(requested))
    missing: dict[str, list[str]] = {}
    for name in requested:
        if name in have:
            continue
        absent = [dep for dep in closure.get(name, ()) if dep not in have]
        if absent:
            missing[name] = absent
    return missing


__all__ += ["dependency_report"]
