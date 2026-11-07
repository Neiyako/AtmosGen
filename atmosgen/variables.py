"""Canonical AtmosGen variable registry — the single source of truth.

Every variable that may appear in an AtmosGen dataset is declared once, here,
with the metadata that the contract, the IO layer, the generators, the derived
diagnostics engine and the physical validator all read:

===========================  ====================================================
field                       meaning
===========================  ====================================================
``name``                    canonical variable name (the key used everywhere)
``dims``                    exact dimension tuple the contract enforces
``units``                   canonical unit string (CF-style, ``**`` exponents)
``long_name``               human-readable description
``standard_name``           CF standard name when one exists, else ``None``
``category``                grouping used by the reports (``thermodynamic``, ...)
``dtype``                   on-disk data type
``source``                  ``primary`` (generated/ingested) or ``derived``
``requires``                direct variable dependencies (``source`` = derived)
``requires_levels``         the computation needs a vertical (pressure) axis
``formula``                 the defining relation, written out
``comment``                 convention notes (units, sign, accumulation, ...)
===========================  ====================================================

Conventions that the task statement leaves open are fixed **here** and must not
be re-invented downstream:

``w``
    Vertical velocity in **Pa s-1** (pressure coordinates, ERA5 convention):
    positive = upward motion (pressure decreasing). Never m/s.
``z``
    Geopotential in **m2 s-2** (ERA5 convention). Geopotential *height* in m is
    ``z / g``.
``tp``
    **Accumulated** total precipitation over the interval that ends at the time
    stamp, in kg m-2 (1 kg m-2 = 1 mm liquid equivalent). It is *not* a rate;
    the instantaneous rate is ``tp / dt``.
``clwc`` / ``ciwc`` / ``crwc`` / ``cswc``
    Mass mixing ratios in kg kg-1 (not column quantities).
``tcwv`` / ``tclw`` / ``tciw`` / ``cwp`` / ``iwp`` / ``twp``
    Column integrals in kg m-2 — never to be confused with the per-level
    mixing ratios above.
``tclw`` vs ``cwp`` / ``tciw`` vs ``iwp``
    Identical quantities under two names (``tclw`` follows the ERA5 naming,
    ``cwp``/``iwp`` the diagnostic naming). ``twp = cwp + iwp`` is the
    condensate-only total; water vapour is reported separately as ``tcwv``.
``mfc`` vs ``mfc_sl``
    ``mfc`` is the **column-integrated** moisture flux convergence
    (kg m-2 s-1); ``mfc_sl`` is a **single-level** diagnostic (kg kg-1 s-1).
    They are separate variables so the two physical meanings are never mixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

__all__ = [
    "DERIVED_VARIABLES",
    "PRIMARY_VARIABLES",
    "VAR_SPECS",
    "VariableError",
    "VariableSpec",
    "dependencies_of",
    "known_variables",
    "missing_dependencies",
    "resolve_dependencies",
    "spec_for",
    "transitive_dependencies",
    "validate_variable_names",
]


class VariableError(ValueError):
    """Raised for an unknown variable or an unsatisfiable dependency set."""


@dataclass(frozen=True)
class VariableSpec:
    """Canonical definition of one AtmosGen variable."""

    name: str
    dims: tuple[str, ...]
    units: str
    long_name: str
    standard_name: str | None
    category: str
    dtype: str = "float64"
    #: ``primary`` = generated or ingested directly; ``derived`` = computed
    #: from other canonical variables by the derived-diagnostics engine.
    source: str = "primary"
    #: direct dependencies (only meaningful for ``source == "derived"``)
    requires: tuple[str, ...] = ()
    #: the computation needs a vertical (pressure) coordinate
    requires_levels: bool = False
    #: the defining relation, written out
    formula: str = ""
    #: convention notes (units, sign, accumulation semantics, ...)
    comment: str = ""

    @property
    def is_derived(self) -> bool:
        return self.source == "derived"

    @property
    def is_level_field(self) -> bool:
        return "level" in self.dims


#: 4-D (level) field template used by the per-level state variables.
_L4 = ("time", "level", "latitude", "longitude")
#: 3-D (column / surface) field template.
_L3 = ("time", "latitude", "longitude")


_VAR_SPECS: tuple[VariableSpec, ...] = (
    # ------------------------------------------------------------------ #
    # Primary state variables
    # ------------------------------------------------------------------ #
    VariableSpec(
        "t", _L4, "K", "Temperature", "air_temperature", "thermodynamic",
        comment="Absolute temperature; the saturation check uses the Tetens formula.",
    ),
    VariableSpec(
        "q", _L4, "kg kg**-1", "Specific humidity", "specific_humidity", "thermodynamic",
        comment="Mass mixing ratio of water vapour (not a column quantity).",
    ),
    VariableSpec(
        "u", _L4, "m s**-1", "U component of wind", "eastward_wind", "dynamic",
        comment="Eastward (zonal) wind speed; positive towards the east.",
    ),
    VariableSpec(
        "v", _L4, "m s**-1", "V component of wind", "northward_wind", "dynamic",
        comment="Meridional wind speed; positive towards the north.",
    ),
    VariableSpec(
        "w", _L4, "Pa s**-1", "Vertical velocity (pressure coordinate)", "omega", "dynamic",
        comment=(
            "CONVENTION: Pa/s (ERA5 style), positive = upward motion (pressure "
            "decreasing). It is NOT a velocity in m/s; w[m/s] = w[Pa/s] / (rho*g)."
        ),
    ),
    VariableSpec(
        "z", _L4, "m**2 s**-2", "Geopotential", "geopotential", "dynamic",
        comment=(
            "CONVENTION: geopotential in m2/s2 (ERA5 style). Geopotential height "
            "in metres is z/g with g = 9.80665 m/s2."
        ),
    ),
    VariableSpec(
        "sp", _L3, "Pa", "Surface pressure", "surface_air_pressure", "surface",
        comment="Surface (or model bottom) pressure; 1 hPa = 100 Pa.",
    ),
    VariableSpec(
        "clwc", _L4, "kg kg**-1", "Specific cloud liquid water content",
        "specific_cloud_liquid_water_content", "cloud",
        comment="Mass mixing ratio of cloud liquid water (>= 0).",
    ),
    VariableSpec(
        "ciwc", _L4, "kg kg**-1", "Specific cloud ice water content",
        "specific_cloud_ice_water_content", "cloud",
        comment="Mass mixing ratio of cloud ice water (>= 0).",
    ),
    VariableSpec(
        "crwc", _L4, "kg kg**-1", "Specific rain water content",
        "specific_rain_water_content", "cloud",
        comment="Mass mixing ratio of rain water (>= 0).",
    ),
    VariableSpec(
        "cswc", _L4, "kg kg**-1", "Specific snow water content",
        "specific_snow_water_content", "cloud",
        comment="Mass mixing ratio of snow water (>= 0).",
    ),
    VariableSpec(
        "tp", _L3, "kg m**-2", "Total precipitation (accumulated)", 
        "lwe_thickness_of_precipitation_amount", "precipitation",
        comment=(
            "CONVENTION: accumulated total precipitation (liquid + solid, melted "
            "equivalent) over the interval ENDING at this time stamp; 1 kg/m2 = 1 mm. "
            "It is NOT an instantaneous rate: rate = tp / dt."
        ),
    ),
    VariableSpec(
        "c", _L3, "1", "Tracer concentration (advection-diffusion)", None, "tracer",
        comment="Passive tracer used by the Mode-2 advection-diffusion solver.",
    ),
    # ------------------------------------------------------------------ #
    # Derived variables
    # ------------------------------------------------------------------ #
    VariableSpec(
        "tcwv", _L3, "kg m**-2", "Total column water vapour",
        "atmosphere_mass_content_of_water_vapor", "column",
        source="derived", requires=("q",), requires_levels=True,
        formula="tcwv = (1/g) * integral(q dp)",
        comment="Column integral over the available pressure levels; kg/m2.",
    ),
    VariableSpec(
        "tclw", _L3, "kg m**-2", "Total column cloud liquid water",
        "atmosphere_mass_content_of_cloud_liquid_water", "column",
        source="derived", requires=("clwc",), requires_levels=True,
        formula="tclw = (1/g) * integral(clwc dp)",
        comment="ERA5-style name; by definition identical to cwp.",
    ),
    VariableSpec(
        "tciw", _L3, "kg m**-2", "Total column cloud ice water",
        "atmosphere_mass_content_of_cloud_ice_water", "column",
        source="derived", requires=("ciwc",), requires_levels=True,
        formula="tciw = (1/g) * integral(ciwc dp)",
        comment="ERA5-style name; by definition identical to iwp.",
    ),
    VariableSpec(
        "rh", _L4, "%", "Relative humidity", "relative_humidity", "derived",
        source="derived", requires=("t", "q"), requires_levels=True,
        formula="rh = 100 * q / qsat(t, p)",
        comment=(
            "Supersaturation policy: values are reported as computed and are never "
            "clipped. rh in (100, 120] % is flagged as supersaturated (physically "
            "possible inside cloud); rh > 200 % is treated as a hard failure."
        ),
    ),
    VariableSpec(
        "r", _L4, "%", "Relative humidity", "relative_humidity", "derived",
        source="derived", requires=("t", "q"), requires_levels=True,
        formula="r = 100 * q / qsat(t, p)",
        comment="Short alias of `rh`: same formula, same values, same units.",
    ),
    VariableSpec(
        "qflux_u", _L4, "kg kg**-1 m s**-1", "Eastward moisture flux", None, "derived",
        source="derived", requires=("q", "u"),
        formula="qflux_u = q * u",
        comment="Per-level moisture flux; a single-level diagnostic, not a column quantity.",
    ),
    VariableSpec(
        "qflux_v", _L4, "kg kg**-1 m s**-1", "Northward moisture flux", None, "derived",
        source="derived", requires=("q", "v"),
        formula="qflux_v = q * v",
        comment="Per-level moisture flux; a single-level diagnostic, not a column quantity.",
    ),
    VariableSpec(
        "qflux_u_col", _L3, "kg m**-1 s**-1", "Column-integrated eastward moisture flux",
        None, "derived",
        source="derived", requires=("qflux_u",), requires_levels=True,
        formula="qflux_u_col = (1/g) * integral(qflux_u dp)",
        comment="Column-integrated moisture flux (kg/m/s); the divergence of this field is `mfc`.",
    ),
    VariableSpec(
        "qflux_v_col", _L3, "kg m**-1 s**-1", "Column-integrated northward moisture flux",
        None, "derived",
        source="derived", requires=("qflux_v",), requires_levels=True,
        formula="qflux_v_col = (1/g) * integral(qflux_v dp)",
        comment="Column-integrated moisture flux (kg/m/s); the divergence of this field is `mfc`.",
    ),
    VariableSpec(
        "mfc", _L3, "kg m**-2 s**-1", "Moisture flux convergence (column-integrated)",
        None, "derived",
        source="derived", requires=("qflux_u_col", "qflux_v_col"), requires_levels=True,
        formula="mfc = -div(qflux_u_col, qflux_v_col)",
        comment=(
            "COLUMN-INTEGRATED moisture flux convergence (kg/m2/s). Requires a "
            "vertical axis; a single-level diagnostic is provided separately as `mfc_sl`."
        ),
    ),
    VariableSpec(
        "mfc_sl", _L3, "kg kg**-1 s**-1", "Moisture flux convergence (single level)",
        None, "derived",
        source="derived", requires=("qflux_u", "qflux_v"),
        formula="mfc_sl = -div(qflux_u, qflux_v) at one level",
        comment=(
            "SINGLE-LEVEL diagnostic (kg/kg/s) evaluated on the lowest available "
            "level. Deliberately kept apart from the column-integrated `mfc`."
        ),
    ),
    VariableSpec(
        "cwp", _L3, "kg m**-2", "Cloud water path",
        "atmosphere_mass_content_of_cloud_liquid_water", "derived",
        source="derived", requires=("clwc",), requires_levels=True,
        formula="cwp = (1/g) * integral(clwc dp)",
        comment="Diagnostic name for the liquid cloud water path; identical to tclw.",
    ),
    VariableSpec(
        "iwp", _L3, "kg m**-2", "Ice water path",
        "atmosphere_mass_content_of_cloud_ice_water", "derived",
        source="derived", requires=("ciwc",), requires_levels=True,
        formula="iwp = (1/g) * integral(ciwc dp)",
        comment="Diagnostic name for the ice water path; identical to tciw.",
    ),
    VariableSpec(
        "twp", _L3, "kg m**-2", "Total water path (cloud liquid + ice)", None, "derived",
        source="derived", requires=("cwp", "iwp"),
        formula="twp = cwp + iwp",
        comment=(
            "Condensate-only total water path. Water vapour is NOT included; it is "
            "reported separately as tcwv."
        ),
    ),
)

VAR_SPECS: dict[str, VariableSpec] = {spec.name: spec for spec in _VAR_SPECS}

#: Variables that are generated or ingested directly.
PRIMARY_VARIABLES: tuple[str, ...] = tuple(
    spec.name for spec in _VAR_SPECS if not spec.is_derived
)
#: Variables that are computed from other canonical variables.
DERIVED_VARIABLES: tuple[str, ...] = tuple(
    spec.name for spec in _VAR_SPECS if spec.is_derived
)


# --------------------------------------------------------------------------- #
# Lookups and dependency helpers
# --------------------------------------------------------------------------- #
def known_variables() -> list[str]:
    """Return the canonical variable names in schema order."""

    return [spec.name for spec in _VAR_SPECS]


def spec_for(name: str) -> VariableSpec:
    """Return the :class:`VariableSpec` for a canonical variable name."""

    try:
        return VAR_SPECS[name]
    except KeyError:
        raise VariableError(
            f"unknown variable {name!r}; canonical variables are {known_variables()}"
        ) from None


def dependencies_of(name: str) -> tuple[str, ...]:
    """Direct dependencies of ``name`` (empty for primary variables)."""

    return spec_for(name).requires


def transitive_dependencies(names: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Return the transitive dependency closure of ``names``.

    The result maps every variable that must exist (directly or indirectly) to
    the variables it directly depends on. Primary variables map to an empty
    tuple.
    """

    closure: dict[str, tuple[str, ...]] = {}
    stack = list(names)
    while stack:
        current = stack.pop()
        if current in closure:
            continue
        spec = spec_for(current)
        closure[current] = spec.requires
        for dependency in spec.requires:
            if dependency not in closure:
                stack.append(dependency)
    return closure


def resolve_dependencies(names: Iterable[str]) -> tuple[str, ...]:
    """Return the ordered transitive closure of ``names`` (dependencies first)."""

    closure = transitive_dependencies(names)
    ordered = [name for name in known_variables() if name in closure]
    return tuple(ordered)


def validate_variable_names(names: Iterable[str]) -> list[str]:
    """Return the subset of ``names`` that is not a canonical variable."""

    return [name for name in names if name not in VAR_SPECS]


def missing_dependencies(
    names: Iterable[str], available: Iterable[str]
) -> dict[str, tuple[str, ...]]:
    """Return, per requested variable, the dependencies missing from ``available``.

    Only variables whose *whole* dependency chain is unsatisfiable are
    reported; a variable that is itself present in ``available`` is never
    reported as missing (an ingested value is authoritative).
    """

    have = set(available)
    closure = transitive_dependencies(names)
    missing: dict[str, tuple[str, ...]] = {}
    for name in names:
        if name in have:
            continue
        chain = closure.get(name, ())
        absent = tuple(dep for dep in chain if dep not in have)
        if absent:
            missing[name] = absent
    return missing


def variable_definitions(names: Iterable[str]) -> dict[str, dict[str, object]]:
    """Serialisable metadata block for the requested variables."""

    block: dict[str, dict[str, object]] = {}
    for name in names:
        spec = VAR_SPECS.get(name)
        if spec is None:
            continue
        block[name] = {
            "units": spec.units,
            "dims": list(spec.dims),
            "long_name": spec.long_name,
            "standard_name": spec.standard_name,
            "category": spec.category,
            "dtype": spec.dtype,
            "source": spec.source,
            "requires": list(spec.requires),
            "requires_levels": spec.requires_levels,
            "formula": spec.formula,
            "comment": spec.comment,
        }
    return block
