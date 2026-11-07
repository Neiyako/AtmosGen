"""Coupled multi-variable atmospheric field generator (``physics`` mode).

Strategy "C — simplified physical field generation": the base meteorological
fields (``t``, ``sp``, ``q``, ``u``, ``v``, ``w``, ``z``), the cloud water
species (``clwc``, ``ciwc``, ``crwc``, ``cswc``) and the accumulated
precipitation (``tp``) are built on a pressure-level grid from **two smooth,
independently advected and diffused pattern drivers** that come from the
*existing* self-written 2-D advection-diffusion solver
(:func:`atmosgen.generators.physics.advection_diffusion.solve_advection_diffusion`).
No field is drawn from an independent random number generator: every field
inherits the solver's spatial continuity and temporal evolution.

This is **not** a full NWP model: there is no radiation, no surface flux, no
full microphysics and no data assimilation.  It is a self-consistent
*simplified* column/field generator.

Conventions (fixed by :mod:`atmosgen.variables`, never re-invented here)
-------------------------------------------------------------------------
``w``   pressure-coordinate vertical velocity in **Pa s-1** (ERA5 style),
        positive = upward motion (pressure decreasing).  Never m/s.
``z``   geopotential in **m2 s-2** (ERA5 style), *not* geopotential height.
``tp``  **accumulated** total precipitation (liquid + solid, melted
        equivalent) over the window that *ends* at the time stamp, in
        kg m-2 (1 kg m-2 = 1 mm).  It is not a rate.

Formulas (all quantities carry the unit written after the colon)
----------------------------------------------------------------
**Step 0 — pattern drivers.**  The solver is run twice with the same grid,
time stepping and tracer parameters (``u0``, ``v0``, ``kappa``) and periodic
boundary conditions, with a Gaussian plume as the initial field:

``driver A``  centre ``tracer.plume_center``            (rotational part)
``driver B``  centre ``tracer.divergence_plume_center`` (divergent part)

Both drivers live on **one shared canonical grid** whose centre is the
mid-point of the two plume centres, so that the two patterns can be combined
without regridding.  The solver's ``gaussian_plume`` initial condition always
places the plume on the *grid* centre; an off-centre plume is therefore
supplied as an explicit initial field: one in-memory dataset carries both
plumes (``pattern_a`` / ``pattern_b``) and is handed to the solver through
its ``initial_condition.kind="from_real"`` interface (the same Gaussian
``A*exp(-(x^2+y^2)/(2*sigma^2))`` as
:func:`atmosgen.generators.physics.conditions.gaussian_plume`, only with an
explicit centre and the grid's own metric).  Because that in-memory dataset
carries the ``latitude``/``longitude`` coordinates, the solver rebuilds
*exactly* the canonical geometry for both runs, and both drivers therefore
share one grid and one time axis.

**Time axis.**  The in-memory initial dataset carries a ``time`` coordinate
equal to ``DEFAULT_START_TIME`` (2000-01-01T00:00:00, the synthetic origin of
the Mode-2 solver), so the solver uses it as the clock origin and the final
time axis is ``DEFAULT_START_TIME + save_index * dt_s`` — a synthetic axis, not
wall-clock data.

Each snapshot is normalised per time step::

    pattern = c / max(c)            (zeros when max(c) < 1e-12)
    anom   = 2*pattern - 1          -> [-1, 1]

``anomA`` drives the rotational/temperature/pressure parts, ``anomB`` drives
the divergent part — the two drivers stay independent, which is what makes the
horizontal divergence (and therefore the moisture-flux convergence diagnostic)
non-zero.

**Step 1 — temperature** ``t`` [K]::

    T_bg(p) = T_s - (T_s - 220 K) * ln(p_s/p) / ln(p_s/p_top)
    shape(p)= 1 - 0.6 * ln(p_s/p) / ln(p_s/p_top)          (1 at surface, 0.4 top)
    t       = T_bg(p) + A_T * shape(p) * anomA              -> [200 K, 320 K]

**Step 2 — surface pressure** ``sp`` [Pa]::

    sp = p_s * (1 + A_p * anomA)                           -> [95 kPa, 107 kPa]

**Step 3 — horizontal wind** ``u``, ``v`` [m s-1].  With the rotational
stream function ``psi`` and the divergent velocity potential ``chi`` [m2 s-1]::

    psi = -U * L * anomA * shape_w(p)
    chi =  D * L * anomB * shape_w(p)
    u   = -dpsi/dy + dchi/dx
    v   =  dpsi/dx + dchi/dy

``L = plume_sigma_deg * 111320 m`` is the pattern length scale, so that
``psi``/``chi`` carry m2 s-1 and ``u``/``v`` come out with the requested
m s-1 magnitude (``U = pattern.wind_speed_scale``,
``D = pattern.divergence_scale``).  ``shape_w(p)`` is the same vertical shape
as the temperature perturbation (1 at the surface, 0.4 at the top), so the
wind has vertical shear.  Derivatives use central differences in the interior
and one-sided differences at the edges; the meridional spacing is ``dy_m``
and the zonal spacing is ``dx_row(lat) = 111320 * cos(lat) * dlon_deg``.
``|u|, |v| <= 60 m s-1`` is asserted (no silent clipping).

**Step 4 — vertical velocity** ``w`` [Pa s-1].  Horizontal divergence
``D = du/dx + dv/dy`` [s-1] (in the continuum the ``psi`` part cancels
analytically, so ``D`` is the Laplacian of ``chi``; the discrete operators
used here are the same ones the derived diagnostics use for the moisture-flux
convergence, so ``w`` and ``mfc`` stay consistent).  Integrating upward from
``w = 0`` at the surface level::

    w[k] = w[k-1] - 0.5*(D[k-1] + D[k]) * (p[k] - p[k-1])     (p in Pa)

Convergence (``D < 0``) therefore gives ``w < 0`` aloft, i.e. rising motion
(``w = dp/dt``).  If ``|w|`` exceeds 10 Pa s-1 the *divergent* wind amplitude
is rescaled once, globally, and the factor is reported in the diagnostics
(no silent clipping).

**Step 5 — specific humidity** ``q`` [kg kg-1].  Tetens saturation, the same
formula the physical validator uses::

    es(T)   = 6.112 * exp(17.67*(T-273.15)/(T-29.65))            [hPa]
    qsat(T,p)= 0.622 * es(T) / (p_hPa - 0.378*es(T))              [kg/kg]

    rh_base(p)  = background_rh + cloud_rh_boost * band(p)
    band(p)     = 1 inside [cloud.top_pressure_hpa, cloud.base_pressure_hpa],
                  tapering linearly to 0 in ln p over 0.3 (trapezoid)
    rh_target   = clip(rh_base(p) + humidity_amplitude*(0.7*anomA + 0.3*anomB),
                       0.02, 1.25)
    q_total     = rh_target * qsat
    q           = min(q_total, qsat)          (the physical saturation limit)
    qc          = q_total - q >= 0            (the condensed excess)

The excess is **not** thrown away: it becomes the cloud condensate of step 6.

**Step 6 — cloud species** [kg kg-1] (all >= 0).  With the ice fraction::

    ice_fraction = clip((freezing_level_k + ice_ramp_k/2 - T)/ice_ramp_k, 0, 1)
    clwc = qc * (1 - ice_fraction)
    ciwc = qc * ice_fraction

(condensate is confined to the cloud band; the masked mass is reported in the
diagnostics).  Precipitation fallout below the cloud base.  The layer
thickness attributed to a level is

    dp(p) = p(p-1) - p(p)            [Pa, downwards positive]
            (for the lowest level the surface layer p_s - p(0) is used, so
             sum(dp) = p_s - p_top)

::

    qc_col     = sum_over_levels(qc * dp/g)                      [kg/m2]
    precip_col = precipitation_fraction * qc_col                  [kg/m2]
    prof(p)    = exp(-(p - p_base)/H),  H = 150 hPa, normalised on the
                 levels below the cloud base so that sum(prof*dp) = 1
                 (prof is dimensionless; when no level lies below the cloud
                 base the whole profile mass sits on the lowest level,
                 prof = 1/dp there)
    x(p)       = g * precip_col * prof(p)                         [kg/kg]
    crwc       = x(p) * (1 - ice_fraction(p))
    cswc       = x(p) * ice_fraction(p)

The falling mass is conserved exactly::

    (1/g) * sum((crwc+cswc) * dp) == precip_col                  (checked)

**Step 7 — accumulated precipitation** ``tp`` [kg m-2]::

    precip_rate_col = (crwc_col + cswc_col) / fallout_timescale_s   [kg/m2/s]
    tp              = precip_rate_col * accumulation_window

``accumulation_window`` is ``precipitation.accumulation_seconds`` when > 0
and ``time.dt_s`` otherwise.  ``tp`` at the first time stamp is the
accumulation over the **first** interval.

**Step 8 — geopotential** ``z`` [m2 s-2] (hydrostatic)::

    Phi_sfc = R_d * T_s * ln(sp/p0)                       (p0 = 101325 Pa)
    I[0]   = 0
    I[k]   = I[k-1] + 0.5*(R_d*T[k-1]/p[k-1] + R_d*T[k]/p[k]) * (p[k]-p[k-1])
    z[k]   = Phi_sfc - I[k]

with ``R_d = 287.058 J kg-1 K-1``.  ``z`` at 100 hPa is ~1.6e5 m2 s-2
(~16 km geopotential height).

**Step 9 — assembly.**  The dataset carries the twelve canonical primary
variables with the exact dimensions enforced by the contract; every variable
carries ``units``/``long_name`` (and ``standard_name`` when the registry has
one) copied verbatim from :data:`atmosgen.variables.VAR_SPECS`, plus an
``atmosgen_construction`` sentence describing how it was built.  The derived
diagnostics (tcwv, cwp, iwp, twp, rh, r, qflux_*, mfc, mfc_sl, tclw, tciw) are
**not** produced here: they belong to the derived-diagnostics engine.

**Step 10 — contract.**  Built and attached exactly like
``AdvectionDiffusionGenerator.sample()`` does.

Dependencies: numpy + xarray only.  No ML, no new third-party packages.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import xarray as xr

from ... import __version__
from ...contract import attach_contract, build_contract
from ...runner import ResourceTracker, set_global_seed
from ...variables import VAR_SPECS
from ..base import BaseGenerator, GenerationResult
from .advection_diffusion import (
    DEFAULT_START_TIME,
    PhysicsError,
    SolverDiagnostics,
    solve_advection_diffusion,
)
from .conditions import METERS_PER_DEGREE, GridGeometry, build_geometry

__all__ = ["MultifieldGenerator", "PhysicsError"]


# --------------------------------------------------------------------------- #
# Physical constants and hard limits
# --------------------------------------------------------------------------- #
G = 9.80665            # m s-2, gravitational acceleration
R_D = 287.058          # J kg-1 K-1, dry-air gas constant
P0 = 101325.0          # Pa, reference surface pressure of the geopotential
T_TOP_BACKGROUND = 220.0   # K, background temperature at the model top
EPSILON = 0.622        # -, Mw/Md
TETENS_A = 6.112       # hPa
TETENS_B = 17.67       # K^-1
TETENS_C = 29.65       # K

#: Hard physical ranges; a violation is an error, never a silent clip.
T_MIN_K, T_MAX_K = 200.0, 320.0
SP_MIN_PA, SP_MAX_PA = 95000.0, 107000.0
WIND_MAX_MS = 60.0
W_MAX_PA_S = 10.0
Z_TOP_MIN, Z_TOP_MAX = 1.4e5, 1.9e5
#: Relative tolerance of the precipitation mass-conservation check.
MASS_CONSERVATION_RTOL = 1e-6

#: Trapezoid taper of the cloud band, expressed in ln p (dimensionless).
CLOUD_BAND_TAPER = 0.3
#: E-folding depth of the below-cloud precipitation profile [hPa].
PRECIP_PROFILE_H = 150.0

#: Defaults for the two optional knobs the lead may add to the config.
DEFAULT_CLOUD_RH_BOOST = 0.35
DEFAULT_FALLOUT_TIMESCALE_S = 3600.0

#: One short sentence per variable: how the field was constructed.
_CONSTRUCTION: dict[str, str] = {
    "t": (
        "Log-pressure background temperature profile (T_surface -> 220 K at the "
        "model top) plus the advection-diffusion pattern anomaly scaled by the "
        "vertical shape function."
    ),
    "q": (
        "Saturation-limited construction: q = min(rh_target*qsat(T,p), qsat(T,p)); "
        "the excess over saturation is condensed into the cloud condensate."
    ),
    "u": (
        "Rotational part from the stream function of pattern driver A plus the "
        "divergent part from the velocity potential of pattern driver B "
        "(central differences in the interior, one-sided at the edges)."
    ),
    "v": (
        "Rotational part from the stream function of pattern driver A plus the "
        "divergent part from the velocity potential of pattern driver B "
        "(central differences in the interior, one-sided at the edges)."
    ),
    "w": (
        "Pressure-coordinate vertical velocity (Pa/s) integrated upward from "
        "w=0 at the surface level with the horizontal divergence of the wind."
    ),
    "z": (
        "Hydrostatic geopotential: Phi_sfc - integral(R_d*T/p dp) on the pressure "
        "levels (m2/s2)."
    ),
    "sp": (
        "Surface pressure p_s*(1 + pressure_amplitude * pattern anomaly), driven by "
        "pattern driver A."
    ),
    "clwc": (
        "Liquid part of the condensed excess (qc) inside the cloud band, "
        "partitioned by the ice fraction."
    ),
    "ciwc": (
        "Ice part of the condensed excess (qc) inside the cloud band, partitioned "
        "by the ice fraction."
    ),
    "crwc": (
        "Precipitation fallout below the cloud base: rain fraction of the "
        "column condensate on a mass-conserving exponential profile "
        "(x = g*precip_col*prof, sum(prof*dp)=1)."
    ),
    "cswc": (
        "Precipitation fallout below the cloud base: snow fraction of the "
        "column condensate on a mass-conserving exponential profile "
        "(x = g*precip_col*prof, sum(prof*dp)=1)."
    ),
    "tp": (
        "Accumulated total precipitation: (rain column + snow column) / fallout "
        "timescale * accumulation window (kg/m2)."
    ),
}


# --------------------------------------------------------------------------- #
# Small numerical helpers
# --------------------------------------------------------------------------- #
def _gaussian_plume_at(
    geometry: GridGeometry,
    center: tuple[float, float],
    *,
    sigma_deg: float,
    amplitude: float,
) -> np.ndarray:
    """Gaussian plume of ``conditions.gaussian_plume`` with an explicit centre.

    Same functional form (``A*exp(-(x^2+y^2)/(2*sigma^2))``) but

    - the centre can differ from the grid centre (the solver's
      ``gaussian_plume`` initial condition can only centre the plume on the
      grid centre), and the x metric uses the *domain* centre latitude so that
      the plume is not stretched by the shifted geometry;
    - the distance to the plume centre is the **minimum-image** distance of
      the periodic torus spanned by the domain.  The solver runs with
      ``boundary_condition.kind="periodic"``, so the initial field must be
      continuous across the wrap-around: with a plain Euclidean distance the
      plume would jump from ``A`` to its far-field value over one cell and the
      resulting pattern would have an artificial gradient (and hence an
      artificial wind of > 60 m/s).

    Returns an array of shape ``(ny, nx)``.
    """

    if sigma_deg <= 0:
        raise PhysicsError(f"羽流 sigma_deg 必须 > 0，得到 {sigma_deg}")
    lat = geometry.latitude[:, None]
    lon = geometry.longitude[None, :]
    cos_lat = float(np.cos(np.radians(geometry.center_lat)))
    x = (lon - geometry.center_lon) * METERS_PER_DEGREE * cos_lat
    y = (lat - geometry.center_lat) * METERS_PER_DEGREE
    x_center = (float(center[1]) - geometry.center_lon) * METERS_PER_DEGREE * cos_lat
    y_center = (float(center[0]) - geometry.center_lat) * METERS_PER_DEGREE
    # Minimum image on the periodic torus (the periodic cell is nx*dx_m by
    # ny*dy_m wide).
    length_x = geometry.nx * geometry.dx_m
    length_y = geometry.ny * geometry.dy_m
    dx = x - x_center
    dx = dx - length_x * np.round(dx / length_x)
    dy = y - y_center
    dy = dy - length_y * np.round(dy / length_y)
    sigma_m = float(sigma_deg) * METERS_PER_DEGREE
    return float(amplitude) * np.exp(-(dx**2 + dy**2) / (2.0 * sigma_m**2))


def _tetens_es(t: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure [hPa] (Tetens, the validator's formula)."""

    return TETENS_A * np.exp(TETENS_B * (t - 273.15) / (t - TETENS_C))


def _qsat(t: np.ndarray, p_hpa: np.ndarray) -> np.ndarray:
    """Saturation specific humidity [kg/kg] (``p_hpa`` broadcasts against ``t``)."""

    es = _tetens_es(t)
    return EPSILON * es / (p_hpa - 0.378 * es)


def _d_dy(field: np.ndarray, dy_m: float) -> np.ndarray:
    """Meridional derivative: central inside, one-sided at the edges.

    ``field`` is indexed ``[..., latitude, longitude]``.
    """

    out = np.empty_like(field)
    out[..., 1:-1, :] = (field[..., 2:, :] - field[..., :-2, :]) / (2.0 * dy_m)
    out[..., 0, :] = (field[..., 1, :] - field[..., 0, :]) / dy_m
    out[..., -1, :] = (field[..., -1, :] - field[..., -2, :]) / dy_m
    return out


def _d_dx(field: np.ndarray, dx_row: np.ndarray) -> np.ndarray:
    """Zonal derivative with a latitude-dependent spacing ``dx_row`` [m].

    ``field`` is indexed ``[..., latitude, longitude]``; the longitude axis is
    the last one.
    """

    out = np.empty_like(field)
    shape = (1,) * (field.ndim - 2) + (dx_row.size, 1)
    dx = dx_row.reshape(shape)
    out[..., 1:-1] = (field[..., 2:] - field[..., :-2]) / (2.0 * dx)
    out[..., 0:1] = (field[..., 1:2] - field[..., 0:1]) / dx
    out[..., -1:] = (field[..., -1:] - field[..., -2:-1]) / dx
    return out


def _layer_thickness(p_pa: np.ndarray, p_surface_pa: float) -> np.ndarray:
    """Layer thickness attributed to every level ``dp(p)`` [Pa].

    ``dp(p) = p(p-1) - p(p)`` (downwards positive); for the lowest level the
    surface layer ``p_surface - p(0)`` is used, so that
    ``sum(dp) == p_surface - p_top``.
    """

    n = p_pa.size
    dp = np.empty(n, dtype=float)
    dp[0] = p_surface_pa - p_pa[0]
    dp[1:] = p_pa[:-1] - p_pa[1:]
    return dp


def _cloud_band(
    levels_hpa: np.ndarray, top_hpa: float, base_hpa: float
) -> np.ndarray:
    """Trapezoid cloud band in ln p: 1 inside, tapering to 0 outside."""

    ln_p = np.log(levels_hpa)
    ln_top = float(np.log(top_hpa))
    ln_base = float(np.log(base_hpa))
    band = np.ones_like(ln_p)
    above = ln_p < ln_top
    below = ln_p > ln_base
    band[above] = np.clip(1.0 - (ln_top - ln_p[above]) / CLOUD_BAND_TAPER, 0.0, 1.0)
    band[below] = np.clip(1.0 - (ln_p[below] - ln_base) / CLOUD_BAND_TAPER, 0.0, 1.0)
    return band


def _normalise_driver(c: np.ndarray) -> np.ndarray:
    """``2*c/max(c) - 1`` per time step (zeros when the field is degenerate)."""

    peak = c.max(axis=(1, 2), keepdims=True)
    safe = np.where(peak >= 1e-12, peak, 1.0)
    pattern = np.divide(c, safe, out=np.zeros_like(c), where=peak >= 1e-12)
    return 2.0 * pattern - 1.0


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
class MultifieldGenerator(BaseGenerator):
    """Coupled multi-variable physics field generator (Phase 7)."""

    name = "multifield"
    mode = "physics"
    description = "耦合多变量物理场生成器（平流-扩散模式驱动）"
    implemented_phase = 7

    def __init__(
        self,
        config: Any = None,
        *,
        seed: int = 42,
        config_path: str | None = None,
        config_sha256: str | None = None,
        trained_from: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(config, seed=seed)
        set_global_seed(seed)
        self.config_path = config_path
        self.config_sha256 = config_sha256
        self.trained_from = dict(trained_from) if trained_from else None
        self.fit_resources: dict[str, Any] = {}
        self.sample_resources: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Availability / life cycle
    # ------------------------------------------------------------------ #
    @classmethod
    def dependency_problems(cls) -> list[str]:
        problems: list[str] = []
        for module in ("numpy", "xarray"):
            try:
                __import__(module)
            except Exception as exc:  # pragma: no cover - depends on environment
                problems.append(f"missing module {module!r}: {exc}")
        return problems

    def fit(self, dataset: xr.Dataset | None = None) -> "MultifieldGenerator":
        """This generator needs no real data; ``dataset`` is accepted and ignored."""

        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Configuration helpers
    # ------------------------------------------------------------------ #
    def _multifield_config(self) -> Any:
        """Return the ``MultifieldConfig`` of the run."""

        config = self.config
        if config is None:
            raise PhysicsError("multifield 生成器需要 multifield 配置段（config.multifield）")
        section = getattr(config, "multifield", None)
        if section is not None:
            return section
        if hasattr(config, "levels_hpa"):  # a bare MultifieldConfig was passed
            return config
        raise PhysicsError(
            "multifield 生成器需要 multifield 配置段（config.multifield）；"
            f"当前配置对象 {type(config).__name__} 不含该段"
        )

    def _requested_fields(self) -> int | None:
        samples = getattr(getattr(self.config, "generation", None), "samples", None)
        fields = getattr(samples, "fields", None) if samples is not None else None
        if fields is None:
            return None
        return int(fields)

    # ------------------------------------------------------------------ #
    # Pattern drivers (step 0)
    # ------------------------------------------------------------------ #
    def _domain_center(self, cfg: Any) -> tuple[float, float]:
        """Centre of the shared canonical domain (mid-point of both plumes)."""

        a = tuple(float(v) for v in cfg.tracer.plume_center)
        b = tuple(float(v) for v in cfg.tracer.divergence_plume_center)
        return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))

    def _initial_dataset(self, cfg: Any, geometry: GridGeometry) -> xr.Dataset:
        """In-memory dataset with both Gaussian plumes on the canonical grid.

        It carries the ``latitude``/``longitude`` coordinates (so the solver
        rebuilds exactly this geometry) and a ``time`` coordinate equal to
        ``DEFAULT_START_TIME`` (the synthetic clock origin of the Mode-2
        solver), and it is handed to *both* driver runs so that the two
        patterns share one grid and one time axis.
        """

        pattern_a = _gaussian_plume_at(
            geometry,
            tuple(float(v) for v in cfg.tracer.plume_center),
            sigma_deg=float(cfg.tracer.plume_sigma_deg),
            amplitude=float(cfg.tracer.plume_amplitude),
        )
        pattern_b = _gaussian_plume_at(
            geometry,
            tuple(float(v) for v in cfg.tracer.divergence_plume_center),
            sigma_deg=float(cfg.tracer.divergence_plume_sigma_deg),
            amplitude=float(cfg.tracer.plume_amplitude),
        )
        for name, field in (("pattern_a", pattern_a), ("pattern_b", pattern_b)):
            if not np.all(np.isfinite(field)):
                raise PhysicsError(f"初始羽流场 {name} 含 NaN/Inf；检查 plume_center/sigma_deg")
        return xr.Dataset(
            data_vars={
                "pattern_a": (("latitude", "longitude"), pattern_a),
                "pattern_b": (("latitude", "longitude"), pattern_b),
            },
            coords={
                "latitude": geometry.latitude,
                "longitude": geometry.longitude,
                "time": np.array([DEFAULT_START_TIME], dtype="datetime64[ns]"),
            },
        )

    def _run_driver(
        self,
        cfg: Any,
        initial_dataset: xr.Dataset,
        *,
        variable: str,
        fields: int | None,
    ) -> tuple[np.ndarray, SolverDiagnostics, np.ndarray]:
        """Run the advection-diffusion solver for one pattern driver.

        The Gaussian plume is supplied as an explicit initial field (the
        solver's ``gaussian_plume`` initial condition can only centre the
        plume on the grid centre).

        Returns ``(c, diagnostics, time_axis)``.
        """

        dataset, diagnostics = solve_advection_diffusion(
            grid={
                "nx": int(cfg.grid.nx),
                "ny": int(cfg.grid.ny),
                "dx_m": float(cfg.grid.dx_m),
                "dy_m": float(cfg.grid.dy_m),
            },
            time={"dt_s": float(cfg.time.dt_s), "steps": int(cfg.time.steps)},
            parameters={
                "u0": float(cfg.tracer.u0),
                "v0": float(cfg.tracer.v0),
                "kappa": float(cfg.tracer.kappa),
            },
            initial_condition={
                "kind": "from_real",
                "variable": variable,
                "amplitude": 1.0,
            },
            boundary_condition={"kind": "periodic"},
            fields=fields,
            real=initial_dataset,
        )
        values = np.asarray(dataset["c"].values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise PhysicsError("模式驱动场含 NaN/Inf；检查平流-扩散参数稳定性")
        time_axis = np.asarray(dataset["time"].values)
        return values, diagnostics, time_axis

    # ------------------------------------------------------------------ #
    # Main generation
    # ------------------------------------------------------------------ #
    def _generate(self, cfg: Any) -> tuple[xr.Dataset, dict[str, Any]]:
        grid_cfg = cfg.grid
        levels_hpa = np.asarray([float(v) for v in cfg.levels_hpa], dtype=float)
        if levels_hpa.size < 2:
            raise PhysicsError("multifield.levels_hpa 至少需要 2 层")
        if np.any(np.diff(levels_hpa) >= 0):
            raise PhysicsError("multifield.levels_hpa 必须严格从地面到顶降序")
        nlev = int(levels_hpa.size)

        p_hpa = levels_hpa
        p_pa = p_hpa * 100.0
        p_surface_hpa = float(cfg.surface.pressure_hpa)
        p_surface_pa = p_surface_hpa * 100.0
        p_top_hpa = float(p_hpa[-1])
        t_surface = float(cfg.surface.temperature_k)

        dx_m = float(grid_cfg.dx_m)
        dy_m = float(grid_cfg.dy_m)
        dt_s = float(cfg.time.dt_s)
        steps = int(cfg.time.steps)

        # ---- step 0: the two pattern drivers -----------------------------
        domain_center = self._domain_center(cfg)
        geometry = build_geometry(
            nx=int(grid_cfg.nx),
            ny=int(grid_cfg.ny),
            dx_m=dx_m,
            dy_m=dy_m,
            center=domain_center,
        )
        fields = self._requested_fields()
        initial_dataset = self._initial_dataset(cfg, geometry)
        driver_a, diag_a, time_axis = self._run_driver(
            cfg, initial_dataset, variable="pattern_a", fields=fields
        )
        driver_b, diag_b, time_axis_b = self._run_driver(
            cfg, initial_dataset, variable="pattern_b", fields=fields
        )
        if driver_a.shape != driver_b.shape:
            raise PhysicsError(
                f"两个模式驱动的形状不一致: {driver_a.shape} vs {driver_b.shape}"
            )
        if not np.array_equal(time_axis, time_axis_b):
            raise PhysicsError("两个模式驱动的时间轴不一致，无法逐时次组合")
        anom_a = _normalise_driver(driver_a)   # (n_time, ny, nx)
        anom_b = _normalise_driver(driver_b)
        n_time = int(anom_a.shape[0])

        # ---- step 1: temperature ------------------------------------------
        log_ratio = np.log(p_surface_hpa / p_hpa)
        log_span = float(np.log(p_surface_hpa / p_top_hpa))
        if log_span <= 0:
            raise PhysicsError("levels_hpa 的顶层必须低于 surface.pressure_hpa")
        t_background = t_surface - (t_surface - T_TOP_BACKGROUND) * log_ratio / log_span
        shape = 1.0 - 0.6 * log_ratio / log_span
        amp_t = float(cfg.pattern.temperature_amplitude_k)
        temperature = (
            t_background[None, :, None, None]
            + amp_t * shape[None, :, None, None] * anom_a[:, None, :, :]
        )
        t_min = float(np.min(temperature))
        t_max = float(np.max(temperature))
        if t_min < T_MIN_K or t_max > T_MAX_K:
            raise PhysicsError(
                f"温度越界: t ∈ [{t_min:.3f}, {t_max:.3f}] K 不在 "
                f"[{T_MIN_K}, {T_MAX_K}] K；减小 pattern.temperature_amplitude_k"
            )

        # ---- step 2: surface pressure --------------------------------------
        amp_p = float(cfg.pattern.pressure_amplitude)
        surface_pressure = p_surface_pa * (1.0 + amp_p * anom_a)
        sp_min = float(np.min(surface_pressure))
        sp_max = float(np.max(surface_pressure))
        if sp_min < SP_MIN_PA or sp_max > SP_MAX_PA:
            raise PhysicsError(
                f"地面气压越界: sp ∈ [{sp_min:.1f}, {sp_max:.1f}] Pa 不在 "
                f"[{SP_MIN_PA}, {SP_MAX_PA}] Pa；减小 pattern.pressure_amplitude"
            )

        # ---- step 3: wind (rotational + divergent) -------------------------
        length_scale = float(cfg.tracer.plume_sigma_deg) * METERS_PER_DEGREE
        wind_scale = float(cfg.pattern.wind_speed_scale)
        divergence_scale = float(cfg.pattern.divergence_scale)
        dx_row = (
            METERS_PER_DEGREE
            * np.cos(np.radians(geometry.latitude))
            * geometry.dlon_deg
        )

        psi = (
            -wind_scale
            * length_scale
            * shape[None, :, None, None]
            * anom_a[:, None, :, :]
        )
        chi = (
            divergence_scale
            * length_scale
            * shape[None, :, None, None]
            * anom_b[:, None, :, :]
        )

        def _wind(chi_field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            u = -_d_dy(psi, dy_m) + _d_dx(chi_field, dx_row)
            v = _d_dx(psi, dx_row) + _d_dy(chi_field, dy_m)
            return u, v

        u, v = _wind(chi)
        wind_absmax = float(max(np.max(np.abs(u)), np.max(np.abs(v))))
        if wind_absmax > WIND_MAX_MS:
            raise PhysicsError(
                f"风速越界: max|u|,|v| = {wind_absmax:.3f} m/s > {WIND_MAX_MS} m/s；"
                "减小 pattern.wind_speed_scale / pattern.divergence_scale"
            )

        # ---- step 4: vertical velocity (Pa/s) ------------------------------
        def _divergence(u_field: np.ndarray, v_field: np.ndarray) -> np.ndarray:
            return _d_dx(u_field, dx_row) + _d_dy(v_field, dy_m)

        def _integrate_w(divergence: np.ndarray) -> np.ndarray:
            w = np.zeros_like(divergence)
            for k in range(1, nlev):
                w[:, k] = w[:, k - 1] - 0.5 * (
                    divergence[:, k - 1] + divergence[:, k]
                ) * (p_pa[k] - p_pa[k - 1])
            return w

        divergence = _divergence(u, v)
        w = _integrate_w(divergence)
        w_absmax = float(np.max(np.abs(w)))
        w_rescale = 1.0
        if w_absmax > W_MAX_PA_S:
            # Rescale the *divergent* wind amplitude once, globally: w and the
            # divergence are linear in chi, so this is exact (no clipping).
            w_rescale = W_MAX_PA_S / w_absmax
            chi = chi * w_rescale
            u, v = _wind(chi)
            wind_absmax = float(max(np.max(np.abs(u)), np.max(np.abs(v))))
            if wind_absmax > WIND_MAX_MS:
                raise PhysicsError(
                    f"风速越界（重标度后）: max|u|,|v| = {wind_absmax:.3f} m/s > "
                    f"{WIND_MAX_MS} m/s"
                )
            divergence = _divergence(u, v)
            w = _integrate_w(divergence)
            w_absmax = float(np.max(np.abs(w)))

        # ---- step 5: specific humidity (saturation limited) -----------------
        rh_boost = float(getattr(cfg.cloud, "cloud_rh_boost", DEFAULT_CLOUD_RH_BOOST))
        background_rh = float(cfg.cloud.background_rh)
        humidity_amplitude = float(cfg.pattern.humidity_amplitude)
        band = _cloud_band(
            p_hpa,
            float(cfg.cloud.top_pressure_hpa),
            float(cfg.cloud.base_pressure_hpa),
        )
        rh_base = background_rh + rh_boost * band
        rh_target = np.clip(
            rh_base[None, :, None, None]
            + humidity_amplitude
            * (0.7 * anom_a[:, None, :, :] + 0.3 * anom_b[:, None, :, :]),
            0.02,
            1.25,
        )
        q_sat = _qsat(temperature, p_hpa[None, :, None, None])
        if not np.all(np.isfinite(q_sat)):
            raise PhysicsError("饱和比湿 qsat 含 NaN/Inf；检查温度与气压层设置")
        q_total = rh_target * q_sat
        humidity = np.minimum(q_total, q_sat)
        condensate = q_total - humidity
        if float(np.min(condensate)) < -1e-15:
            raise PhysicsError("凝结量出现负值；检查 relative-humidity 构造")
        condensate = np.maximum(condensate, 0.0)

        # The condensate lives inside the cloud band; anything outside is
        # masked away (and the masked mass is reported).
        layer_dp = _layer_thickness(p_pa, p_surface_pa)
        in_band = (band > 0.0).astype(float)
        condensate_band = condensate * in_band[None, :, None, None]
        masked_condensate = float(
            np.max(
                np.sum(
                    (condensate - condensate_band) * layer_dp[None, :, None, None] / G,
                    axis=1,
                )
            )
        )

        # ---- step 6: cloud species -----------------------------------------
        freezing = float(cfg.cloud.freezing_level_k)
        ice_ramp = float(cfg.cloud.ice_ramp_k)
        ice_fraction = np.clip(
            (freezing + 0.5 * ice_ramp - temperature) / ice_ramp, 0.0, 1.0
        )
        clwc = condensate_band * (1.0 - ice_fraction)
        ciwc = condensate_band * ice_fraction

        qc_column = np.sum(
            condensate_band * layer_dp[None, :, None, None] / G, axis=1
        )
        precip_column = float(cfg.cloud.precipitation_fraction) * qc_column

        # Precipitation fallout profile below the cloud base: exponential in
        # pressure, normalised on the levels below the cloud base so that
        # sum(prof*dp) == 1 (prof is dimensionless).
        below_base = p_hpa > float(cfg.cloud.base_pressure_hpa)
        profile = np.zeros(nlev, dtype=float)
        if bool(np.any(below_base)):
            raw = np.where(
                below_base,
                np.exp(
                    -(p_hpa - float(cfg.cloud.base_pressure_hpa)) / PRECIP_PROFILE_H
                ),
                0.0,
            )
            total = float(np.sum(raw * layer_dp))
            if total > 0:
                profile = raw / total
        if float(np.sum(profile * layer_dp)) <= 0:
            # No level below the cloud base: the whole profile mass sits on
            # the lowest level (prof = 1/dp keeps sum(prof*dp) == 1).
            profile = np.zeros(nlev, dtype=float)
            profile[0] = 1.0 / layer_dp[0]

        # x(p) = g * precip_col * prof(p)   [kg/kg]
        mixing_ratio = G * precip_column[:, None, :, :] * profile[None, :, None, None]
        rain = mixing_ratio * (1.0 - ice_fraction)
        snow = mixing_ratio * ice_fraction
        rain_column = np.sum(rain * layer_dp[None, :, None, None] / G, axis=1)
        snow_column = np.sum(snow * layer_dp[None, :, None, None] / G, axis=1)

        # Mass conservation: (1/g)*sum((crwc+cswc)*dp) == precip_col
        conserved = rain_column + snow_column
        scale = max(float(np.max(np.abs(precip_column))), 1e-30)
        conservation_error = float(
            np.max(np.abs(conserved - precip_column)) / scale
        )
        if conservation_error > MASS_CONSERVATION_RTOL:
            raise PhysicsError(
                f"降水质量不守恒: 相对误差 {conservation_error:.3e} > "
                f"{MASS_CONSERVATION_RTOL:.1e}（检查降水廓线归一化）"
            )

        # ---- step 7: accumulated precipitation ------------------------------
        fallout_timescale = float(
            getattr(
                cfg.precipitation, "fallout_timescale_s", DEFAULT_FALLOUT_TIMESCALE_S
            )
        )
        if fallout_timescale <= 0:
            raise PhysicsError(
                f"precipitation.fallout_timescale_s 必须 > 0，得到 {fallout_timescale}"
            )
        accumulation_window = float(
            getattr(cfg.precipitation, "accumulation_seconds", 0.0)
        )
        if accumulation_window <= 0:
            accumulation_window = dt_s
        precip_rate_column = conserved / fallout_timescale
        total_precip = precip_rate_column * accumulation_window
        if float(np.min(total_precip)) < 0:
            raise PhysicsError("累计降水出现负值；检查降水构造")

        # ---- step 8: geopotential ------------------------------------------
        phi_surface = R_D * t_surface * np.log(surface_pressure / P0)
        integration = np.zeros_like(temperature)
        for k in range(1, nlev):
            integration[:, k] = integration[:, k - 1] + 0.5 * (
                R_D * temperature[:, k - 1] / p_pa[k - 1]
                + R_D * temperature[:, k] / p_pa[k]
            ) * (p_pa[k] - p_pa[k - 1])
        geopotential = phi_surface[:, None, :, :] - integration

        # ---- step 9: assemble ------------------------------------------------
        data_vars: dict[str, Any] = {}
        for name, values in (
            ("t", temperature),
            ("q", humidity),
            ("u", u),
            ("v", v),
            ("w", w),
            ("z", geopotential),
            ("clwc", clwc),
            ("ciwc", ciwc),
            ("crwc", rain),
            ("cswc", snow),
        ):
            data_vars[name] = (("time", "level", "latitude", "longitude"), values)
        data_vars["sp"] = (("time", "latitude", "longitude"), surface_pressure)
        data_vars["tp"] = (("time", "latitude", "longitude"), total_precip)

        dataset = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": time_axis.astype("datetime64[ns]"),
                "level": p_hpa.astype("float64"),
                "latitude": geometry.latitude.astype("float64"),
                "longitude": geometry.longitude.astype("float64"),
            },
        )
        dataset["level"].attrs.update(
            {"units": "hPa", "standard_name": "air_pressure"}
        )
        dataset["latitude"].attrs.update({"units": "degrees_north"})
        dataset["longitude"].attrs.update({"units": "degrees_east"})

        rescale_note = (
            f" The divergent wind amplitude was rescaled globally by "
            f"{w_rescale:.6g} to keep |w| within {W_MAX_PA_S} Pa/s."
            if w_rescale != 1.0
            else " No amplitude rescaling was required."
        )
        for name in data_vars:
            spec = VAR_SPECS[name]
            attrs = {
                "units": spec.units,
                "long_name": spec.long_name,
                "atmosgen_construction": _CONSTRUCTION[name],
            }
            if spec.standard_name:
                attrs["standard_name"] = spec.standard_name
            if name in {"u", "v"}:
                attrs["atmosgen_construction"] += rescale_note
            elif name == "w":
                attrs["atmosgen_construction"] += rescale_note
            dataset[name].attrs.update(attrs)

        # ---- guards ----------------------------------------------------------
        for name in data_vars:
            values = np.asarray(dataset[name].values, dtype=float)
            if not np.all(np.isfinite(values)):
                raise PhysicsError(f"变量 {name} 含 NaN/Inf；生成结果不是有限场")
        z_top = float(np.min(geopotential[:, -1, :, :]))
        z_top_max = float(np.max(geopotential[:, -1, :, :]))
        if z_top < Z_TOP_MIN or z_top_max > Z_TOP_MAX:
            raise PhysicsError(
                f"顶层位势超出合理范围: z(top) ∈ [{z_top:.4g}, {z_top_max:.4g}] m2/s2 "
                f"不在 [{Z_TOP_MIN:.4g}, {Z_TOP_MAX:.4g}] m2/s2"
            )

        diagnostics: dict[str, Any] = {
            # Flat keys first: the CLI summary (``atmosgen generate``) reads
            # cfl_x / cfl_y / diffusion_number / mass_drift_max from the
            # diagnostics block, exactly like the advection-diffusion generator.
            "cfl_x": float(diag_a.as_dict()["cfl_x"]),
            "cfl_y": float(diag_a.as_dict()["cfl_y"]),
            "diffusion_number": float(diag_a.as_dict()["diffusion_number"]),
            "mass_drift_max": float(
                max(
                    float(diag_a.as_dict()["mass_drift_max"]),
                    float(diag_b.as_dict()["mass_drift_max"]),
                )
            ),
            "drivers": {
                "driver_a": diag_a.as_dict(),
                "driver_b": diag_b.as_dict(),
                "domain_center": [float(domain_center[0]), float(domain_center[1])],
                "length_scale_m": float(length_scale),
                "n_saved_fields": int(n_time),
                "requested_fields": fields,
                "dt_s": float(dt_s),
                "steps": int(steps),
                "time_origin": str(np.datetime_as_string(DEFAULT_START_TIME, unit="s")),
            },
            "ranges": {
                "t_min": t_min,
                "t_max": t_max,
                "sp_min": sp_min,
                "sp_max": sp_max,
                "u_abs_max": float(np.max(np.abs(u))),
                "v_abs_max": float(np.max(np.abs(v))),
                "w_abs_max": w_absmax,
                "z_top_min": z_top,
                "z_top_max": z_top_max,
                "q_min": float(np.min(humidity)),
                "q_max": float(np.max(humidity)),
                "tp_min": float(np.min(total_precip)),
                "tp_max": float(np.max(total_precip)),
                "crwc_max": float(np.max(rain)),
                "cswc_max": float(np.max(snow)),
            },
            "rescaling": {
                "w_rescale_factor": float(w_rescale),
                "wind_rescale_applied": bool(w_rescale != 1.0),
                "wind_abs_max": wind_absmax,
            },
            "cloud": {
                "band": [float(value) for value in band],
                "cloud_rh_boost": rh_boost,
                "layer_dp_pa": [float(value) for value in layer_dp],
                "condensate_column_max": float(np.max(qc_column)),
                "precip_column_max": float(np.max(precip_column)),
                "masked_condensate_column": masked_condensate,
                "fallout_timescale_s": fallout_timescale,
                "accumulation_window_s": accumulation_window,
                "mass_conservation_rel_error": conservation_error,
            },
            "grid": {
                "nx": int(grid_cfg.nx),
                "ny": int(grid_cfg.ny),
                "dx_m": dx_m,
                "dy_m": dy_m,
                "dlon_deg": float(geometry.dlon_deg),
                "dlat_deg": float(geometry.dlat_deg),
            },
        }
        return dataset, diagnostics

    # ------------------------------------------------------------------ #
    # Public sampling entry point
    # ------------------------------------------------------------------ #
    def sample(self) -> GenerationResult:
        self.require_fitted()
        cfg = self._multifield_config()

        tracker = ResourceTracker(label=f"physics:{self.name}").start()
        try:
            dataset, diagnostics = self._generate(cfg)
        finally:
            self.sample_resources = tracker.stop()

        multifield_config = {
            "grid": cfg.grid.model_dump(),
            "time": cfg.time.model_dump(),
            "levels_hpa": [float(value) for value in cfg.levels_hpa],
            "surface": cfg.surface.model_dump(),
            "pattern": cfg.pattern.model_dump(),
            "tracer": cfg.tracer.model_dump(),
            "cloud": cfg.cloud.model_dump(),
            "precipitation": cfg.precipitation.model_dump(),
        }
        n_saved = int(dataset.sizes["time"])
        contract = build_contract(
            dataset,
            generation_method="physics",
            source={
                "kind": "synthetic",
                "dataset": "physics.multifield",
                "uri": None,
                "trained_from": self.trained_from,
                "access_date": None,
                "readonly": True,
                "license": "AtmosGen self-written solver (no third-party code vendored)",
            },
            model_name="physics.multifield",
            model_version=__version__,
            random_seed=int(self.seed),
            initial_conditions={
                "kind": "gaussian_plume",
                "details": {
                    "driver_a": {
                        "center": [float(v) for v in cfg.tracer.plume_center],
                        "sigma_deg": float(cfg.tracer.plume_sigma_deg),
                        "amplitude": float(cfg.tracer.plume_amplitude),
                        "role": "rotational / temperature / pressure pattern",
                    },
                    "driver_b": {
                        "center": [
                            float(v) for v in cfg.tracer.divergence_plume_center
                        ],
                        "sigma_deg": float(cfg.tracer.divergence_plume_sigma_deg),
                        "amplitude": float(cfg.tracer.plume_amplitude),
                        "role": "divergent wind pattern (independent from driver A)",
                    },
                    "domain_center": diagnostics["drivers"]["domain_center"],
                    "note": (
                        "两个驱动共用同一规范网格（中心取两羽流中心的中点）；"
                        "两个高斯羽流作为显式初始场（求解器 from_real 初始场接口）"
                        "交给平流-扩散求解器演化，周期边界；时间轴起点为 "
                        "DEFAULT_START_TIME（2000-01-01T00:00:00，合成时间轴）。"
                    ),
                },
            },
            boundary_conditions={"kind": "periodic"},
            physical_parameters={
                "grid": multifield_config["grid"],
                "dt_s": multifield_config["time"]["dt_s"],
                "steps": multifield_config["time"]["steps"],
                "levels_hpa": multifield_config["levels_hpa"],
                "surface": multifield_config["surface"],
                "pattern": multifield_config["pattern"],
                "tracer": multifield_config["tracer"],
                "cloud": multifield_config["cloud"],
                "precipitation": multifield_config["precipitation"],
                "fallout_timescale_s": diagnostics["cloud"]["fallout_timescale_s"],
                "cloud_rh_boost": diagnostics["cloud"]["cloud_rh_boost"],
                "length_scale_m": diagnostics["drivers"]["length_scale_m"],
            },
            generation_config={
                "config_path": self.config_path,
                "config_sha256": self.config_sha256,
                "samples": {"unit": "field", "fields": n_saved},
                "discretisation": {
                    "driver": "explicit_euler_upwind_central_diffusion",
                    "horizontal_derivatives": "central interior / one-sided edges",
                    "vertical_w": "upward integration from w=0 at the surface level",
                    "geopotential": "hydrostatic",
                    "precipitation_profile": "exp(-(p-p_base)/150hPa), sum(prof*dp)=1",
                },
                "diagnostics": {
                    "cfl_x": diagnostics["drivers"]["driver_a"]["cfl_x"],
                    "cfl_y": diagnostics["drivers"]["driver_a"]["cfl_y"],
                    "diffusion_number": diagnostics["drivers"]["driver_a"][
                        "diffusion_number"
                    ],
                    "w_rescale_factor": diagnostics["rescaling"]["w_rescale_factor"],
                    "mass_conservation_rel_error": diagnostics["cloud"][
                        "mass_conservation_rel_error"
                    ],
                },
            },
            metadata={
                "notes": (
                    "Mode 2 physics generation, strategy C (simplified physical "
                    "field generation): the pressure-level multi-variable dataset "
                    "is driven by two independent advection-diffusion pattern "
                    "drivers. It is NOT a full NWP model (no radiation, no surface "
                    "fluxes, no full microphysics, no data assimilation). The "
                    "derived diagnostics (tcwv/cwp/iwp/twp/rh/r/qflux_*/mfc/...) "
                    "are produced by the derived-diagnostics engine, not here."
                ),
                "elapsed_seconds": self.sample_resources.get("elapsed_seconds"),
                "peak_rss_mb": self.sample_resources.get("peak_rss_mb"),
            },
        )
        dataset = attach_contract(dataset, contract)
        return GenerationResult(
            dataset=dataset,
            contract=contract,
            metadata={
                "diagnostics": diagnostics,
                "resources": self.sample_resources,
                "multifield_config": multifield_config,
            },
        )
