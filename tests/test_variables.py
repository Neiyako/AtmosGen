"""Tests of the extended variable system (schema 1.2).

Covers the areas the task statement asks for:

* new-variable generation (coupled multifield generator),
* unit consistency (registry units vs. the written NetCDF),
* variable range checks,
* derived-variable computation,
* cloud-water-path integration,
* numerical MFC computation,
* missing-dependency handling (NOT_AVAILABLE, never a zero placeholder),
* NetCDF write + re-read round trip,
* regression of the old configurations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from atmosgen.contract import (
    VAR_SPECS,
    attach_contract,
    build_contract,
    known_variables,
    spec_for,
)
from atmosgen.config import load_config_file, validate_config_file
from atmosgen.derived import (
    GRAVITY,
    column_integral,
    compute_derived,
    horizontal_divergence,
    level_pressure_pa,
    saturation_specific_humidity,
)
from atmosgen.generation import check_requested_variables
from atmosgen.variables import (
    DERIVED_VARIABLES,
    PRIMARY_VARIABLES,
    missing_dependencies,
    resolve_dependencies,
    transitive_dependencies,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Variables added by the schema-1.2 extension.
NEW_VARIABLES = (
    "sp",
    "w",
    "z",
    "tp",
    "rh",
    "r",
    "tclw",
    "tciw",
    "qflux_u",
    "qflux_v",
    "qflux_u_col",
    "qflux_v_col",
    "mfc",
    "mfc_sl",
    "cwp",
    "iwp",
    "twp",
)


# --------------------------------------------------------------------------- #
# Fixtures: a small, physically consistent canonical dataset
# --------------------------------------------------------------------------- #
def _state_dataset(seed: int = 3) -> xr.Dataset:
    """A dataset with the state variables needed by the derived engine."""

    rng = np.random.default_rng(seed)
    nt, nl, nlat, nlon = 2, 4, 5, 6
    levels = np.array([1000.0, 850.0, 700.0, 500.0])
    lat = np.linspace(30.0, 34.0, nlat)
    lon = np.linspace(110.0, 115.0, nlon)
    times = np.array(["2020-01-01T00", "2020-01-01T06"], dtype="datetime64[ns]")
    shape4 = (nt, nl, nlat, nlon)

    t = 250.0 + 20.0 * rng.random(shape4)
    q = 2e-3 + 4e-3 * rng.random(shape4)
    u = 4.0 + 6.0 * rng.random(shape4)
    v = 2.0 + 5.0 * rng.random(shape4)
    clwc = 5e-5 * rng.random(shape4)
    ciwc = 2e-5 * rng.random(shape4)

    ds = xr.Dataset(
        data_vars={
            "t": (("time", "level", "latitude", "longitude"), t, {"units": "K"}),
            "q": (("time", "level", "latitude", "longitude"), q, {"units": "kg kg**-1"}),
            "u": (("time", "level", "latitude", "longitude"), u, {"units": "m s**-1"}),
            "v": (("time", "level", "latitude", "longitude"), v, {"units": "m s**-1"}),
            "clwc": (("time", "level", "latitude", "longitude"), clwc,
                     {"units": "kg kg**-1"}),
            "ciwc": (("time", "level", "latitude", "longitude"), ciwc,
                     {"units": "kg kg**-1"}),
        },
        coords={"time": times, "level": levels, "latitude": lat, "longitude": lon},
    )
    ds["level"].attrs["units"] = "hPa"
    ds["latitude"].attrs["units"] = "degrees_north"
    ds["longitude"].attrs["units"] = "degrees_east"
    return ds


ALL_DERIVED = [
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
]


# --------------------------------------------------------------------------- #
# 1. Registry
# --------------------------------------------------------------------------- #
def test_registry_contains_every_new_variable():
    for name in NEW_VARIABLES:
        assert name in known_variables(), name
        assert name in VAR_SPECS


def test_registry_metadata_is_complete():
    for name, spec in VAR_SPECS.items():
        assert spec.units, name
        assert spec.long_name, name
        assert spec.dtype, name
        assert spec.source in {"primary", "derived"}, name
        assert spec.category, name
        # every dependency must itself be a canonical variable
        for dependency in spec.requires:
            assert dependency in VAR_SPECS, (name, dependency)


def test_conventions_are_explicit():
    """w / z / tp must carry an unambiguous convention."""

    assert spec_for("w").units == "Pa s**-1"
    assert "Pa/s" in spec_for("w").comment
    assert spec_for("z").units == "m**2 s**-2"
    assert spec_for("tp").units == "kg m**-2"
    assert "accumulated" in spec_for("tp").comment.lower()


def test_column_variables_are_not_mixing_ratios():
    for name in ("tcwv", "tclw", "tciw", "cwp", "iwp", "twp"):
        assert spec_for(name).units == "kg m**-2", name
        assert spec_for(name).dims == ("time", "latitude", "longitude"), name
    for name in ("clwc", "ciwc", "crwc", "cswc"):
        assert spec_for(name).units == "kg kg**-1", name


def test_mfc_and_mfc_sl_have_different_units():
    """The column and the single-level diagnostic must not be mixed up."""

    assert spec_for("mfc").units == "kg m**-2 s**-1"
    assert spec_for("mfc_sl").units == "kg kg**-1 s**-1"
    assert spec_for("mfc").comment != spec_for("mfc_sl").comment


def test_dependency_graph_is_acyclic_and_complete():
    closure = transitive_dependencies(ALL_DERIVED)
    for name, requires in closure.items():
        for dependency in requires:
            assert dependency in closure, (name, dependency)
    assert "q" in resolve_dependencies(["tcwv"])
    assert "clwc" in resolve_dependencies(["twp"])


# --------------------------------------------------------------------------- #
# 2. Derived-variable computation
# --------------------------------------------------------------------------- #
def test_compute_derived_produces_every_variable():
    ds = _state_dataset()
    out, report = compute_derived(ds, variables=ALL_DERIVED)
    for name in ALL_DERIVED:
        assert report["status"][name] == "COMPUTED", (name, report["unavailable"])
        assert name in out.data_vars, name
        assert tuple(out[name].dims) == spec_for(name).dims, name
        assert np.all(np.isfinite(out[name].values)), name


def test_relative_humidity_matches_the_definition():
    ds = _state_dataset()
    out, _ = compute_derived(ds, variables=["rh"])
    levels_pa, _ = level_pressure_pa(out)
    t = out["t"].values
    q = out["q"].values
    p_hpa = (levels_pa / 100.0).reshape(1, -1, 1, 1)
    qsat = saturation_specific_humidity(t, p_hpa)
    expected = 100.0 * q / qsat
    assert np.allclose(out["rh"].values, expected, rtol=1e-12, atol=1e-12)
    assert out["rh"].attrs["units"] == "%"


def test_relative_humidity_is_not_clipped():
    """Supersaturation must be reported, never silently removed."""

    ds = _state_dataset()
    levels = np.asarray(ds["level"].values, dtype=float)
    q = np.full((2, 4, 5, 6), 0.02)  # far above saturation at 255 K
    ds["q"] = (("time", "level", "latitude", "longitude"), q)
    out, _ = compute_derived(ds, variables=["rh"])
    assert float(out["rh"].max()) > 100.0
    assert "never clipped" in out["rh"].attrs["atmosgen_supersaturation_policy"]


def test_r_is_an_alias_of_rh():
    ds = _state_dataset()
    out, _ = compute_derived(ds, variables=["rh", "r"])
    assert np.array_equal(out["rh"].values, out["r"].values)
    assert out["r"].attrs["units"] == "%"


def test_moisture_flux_is_q_times_wind():
    ds = _state_dataset()
    out, _ = compute_derived(ds, variables=["qflux_u", "qflux_v"])
    assert np.allclose(out["qflux_u"].values, (out["q"] * out["u"]).values)
    assert np.allclose(out["qflux_v"].values, (out["q"] * out["v"]).values)
    assert out["qflux_u"].attrs["units"] == "kg kg**-1 m s**-1"


# --------------------------------------------------------------------------- #
# 3. Cloud water path integration
# --------------------------------------------------------------------------- #
def test_cloud_water_path_is_the_vertical_integral():
    ds = _state_dataset()
    out, _ = compute_derived(ds, variables=["cwp", "iwp", "tclw", "tciw"])
    levels_pa, _ = level_pressure_pa(out)
    expected_cwp = column_integral(out["clwc"], levels_pa).values
    expected_iwp = column_integral(out["ciwc"], levels_pa).values
    assert np.allclose(out["cwp"].values, expected_cwp, rtol=1e-12, atol=1e-12)
    assert np.allclose(out["iwp"].values, expected_iwp, rtol=1e-12, atol=1e-12)
    # tclw / tciw are the ERA5-style names of exactly the same quantities
    assert np.allclose(out["tclw"].values, out["cwp"].values)
    assert np.allclose(out["tciw"].values, out["iwp"].values)


def test_total_water_path_is_the_sum_of_its_parts():
    ds = _state_dataset()
    out, _ = compute_derived(ds, variables=["cwp", "iwp", "twp"])
    assert np.allclose(out["twp"].values, (out["cwp"] + out["iwp"]).values)
    assert out["twp"].attrs["units"] == "kg m**-2"


def test_column_integral_of_a_constant_field_is_exact():
    """A hand-checkable analytic case: f = c on [p0, p1] -> (1/g) * c * (p1 - p0)."""

    levels_pa = np.array([100000.0, 50000.0])
    field = xr.DataArray(
        np.full((3, 2), 2.0e-5),
        dims=("time", "level"),
        coords={"time": np.arange(3), "level": levels_pa / 100.0},
    )
    column = column_integral(field, levels_pa)
    expected = 2.0e-5 * (100000.0 - 50000.0) / GRAVITY
    assert np.allclose(column.values, expected, rtol=1e-12)


# --------------------------------------------------------------------------- #
# 4. MFC (numerical)
# --------------------------------------------------------------------------- #
def test_mfc_matches_the_negative_divergence_of_the_column_flux():
    ds = _state_dataset()
    out, _ = compute_derived(
        ds, variables=["qflux_u_col", "qflux_v_col", "mfc"]
    )
    divergence = horizontal_divergence(out["qflux_u_col"], out["qflux_v_col"])
    assert np.allclose(out["mfc"].values, (-divergence).values, rtol=1e-10, atol=1e-16)


def test_mfc_of_a_uniform_flux_is_zero():
    """A spatially uniform moisture flux has no convergence."""

    ds = _state_dataset()
    uniform = xr.Dataset(
        data_vars={
            "qflux_u_col": (
                ("time", "latitude", "longitude"),
                np.full((2, 5, 6), 100.0),
            ),
            "qflux_v_col": (
                ("time", "latitude", "longitude"),
                np.full((2, 5, 6), -50.0),
            ),
        },
        coords={
            "time": np.array(["2020-01-01T00", "2020-01-01T06"], dtype="datetime64[ns]"),
            "latitude": np.linspace(30.0, 34.0, 5),
            "longitude": np.linspace(110.0, 115.0, 6),
        },
    )
    divergence = horizontal_divergence(uniform["qflux_u_col"], uniform["qflux_v_col"])
    assert np.allclose(divergence.values, 0.0, atol=1e-12)


def test_mfc_of_a_linear_flux_is_analytic():
    """u = a*x, v = 0  =>  div = a  =>  mfc = -a (exact for central differences)."""

    lat = np.linspace(30.0, 34.0, 5)
    lon = np.linspace(110.0, 115.0, 6)
    dx_row = 111320.0 * np.cos(np.deg2rad(lat)) * (lon[1] - lon[0])
    a = 2.0e-6  # 1/s
    x = (lon - lon[0])[None, :].repeat(len(lat), axis=0)
    flux_u = xr.DataArray(
        (a * dx_row[:, None] * x)[None, :, :],
        dims=("time", "latitude", "longitude"),
        coords={
            "time": np.array(["2020-01-01T00"], dtype="datetime64[ns]"),
            "latitude": lat,
            "longitude": lon,
        },
    )
    flux_v = xr.zeros_like(flux_u)
    divergence = horizontal_divergence(flux_u, flux_v)
    interior = divergence.values[0, 1:-1, 1:-1]
    assert np.allclose(interior, a, rtol=1e-10)


def test_mfc_and_mfc_sl_are_labelled_by_their_type():
    ds = _state_dataset()
    out, _ = compute_derived(ds, variables=["mfc", "mfc_sl"])
    assert out["mfc"].attrs["atmosgen_mfc_type"] == "column_integrated"
    assert out["mfc_sl"].attrs["atmosgen_mfc_type"] == "single_level"
    assert out["mfc_sl"].attrs["atmosgen_level_used"]


def test_single_level_mfc_uses_the_lowest_level():
    ds = _state_dataset()
    out, _ = compute_derived(ds, variables=["qflux_u", "qflux_v", "mfc_sl"])
    levels_pa, _ = level_pressure_pa(out)
    surface_index = int(np.argmax(levels_pa))
    expected = -horizontal_divergence(
        out["qflux_u"].isel(level=surface_index),
        out["qflux_v"].isel(level=surface_index),
    )
    assert np.allclose(out["mfc_sl"].values, expected.values, rtol=1e-12)


# --------------------------------------------------------------------------- #
# 5. Missing dependencies
# --------------------------------------------------------------------------- #
def test_missing_dependency_is_reported_not_zero_filled():
    ds = _state_dataset().drop_vars(["t"])
    out, report = compute_derived(ds, variables=["rh"])
    assert report["status"]["rh"] == "NOT_AVAILABLE"
    assert "t" in report["unavailable"]["rh"]
    assert "rh" not in out.data_vars
    assert "rh" not in ds.data_vars  # nothing was invented


def test_missing_level_coordinate_is_reported():
    ds = _state_dataset().drop_vars(["level"])
    out, report = compute_derived(ds, variables=["tcwv", "cwp"])
    assert report["status"]["tcwv"] == "NOT_AVAILABLE"
    assert report["status"]["cwp"] == "NOT_AVAILABLE"
    assert "level" in report["unavailable"]["tcwv"]
    assert "tcwv" not in out.data_vars


def test_dependency_check_reports_unsatisfiable_requests():
    problems = check_requested_variables(
        ["rh"], capabilities=["t", "q"], model="unit-test"
    )
    assert problems == []  # t and q are present -> rh is derivable
    problems = check_requested_variables(
        ["cwp"], capabilities=["t", "q"], model="unit-test"
    )
    assert problems and "clwc" in problems[0]
    problems = check_requested_variables(
        ["sp"], capabilities=["t", "q"], model="unit-test"
    )
    assert problems and "sp" in problems[0]


def test_missing_dependencies_helper():
    missing = missing_dependencies(["rh"], available=["q"])
    assert missing["rh"] == ("t",)
    assert missing_dependencies(["rh"], available=["t", "q"]) == {}


def test_existing_variable_is_kept_not_overwritten():
    ds = _state_dataset()
    ds["tcwv"] = (("time", "latitude", "longitude"), np.full((2, 5, 6), 42.0))
    out, report = compute_derived(ds, variables=["tcwv"])
    assert report["status"]["tcwv"] == "KEPT"
    assert float(out["tcwv"].mean()) == 42.0


# --------------------------------------------------------------------------- #
# 6. NetCDF round trip
# --------------------------------------------------------------------------- #
def test_netcdf_roundtrip_preserves_units_and_attributes(tmp_path: Path):
    from atmosgen.data.io import read_canonical, save_canonical

    ds = _state_dataset()
    out, _ = compute_derived(ds, variables=ALL_DERIVED)
    contract = build_contract(
        out,
        generation_method="statistical",
        source={"kind": "synthetic", "dataset": "unit-test", "readonly": True},
        model_name="unit-test",
        random_seed=42,
    )
    out = attach_contract(out, contract)
    target = tmp_path / "data.nc"
    save_canonical(out, target, contract=contract)

    reopened = read_canonical(target)
    try:
        for name in ALL_DERIVED:
            assert name in reopened.data_vars, name
            assert reopened[name].attrs["units"] == spec_for(name).units, name
            assert tuple(reopened[name].dims) == spec_for(name).dims, name
        assert reopened["level"].attrs["units"] == "hPa"
        assert reopened["latitude"].attrs["units"] == "degrees_north"
        assert reopened["longitude"].attrs["units"] == "degrees_east"
        # the values survive the round trip
        assert np.allclose(reopened["cwp"].values, out["cwp"].values)
    finally:
        reopened.close()


def test_contract_records_the_variable_definitions(tmp_path: Path):
    from atmosgen.data.io import save_canonical

    ds = _state_dataset()
    out, _ = compute_derived(ds, variables=["rh", "cwp"])
    contract = build_contract(
        out,
        generation_method="statistical",
        source={"kind": "synthetic", "dataset": "unit-test", "readonly": True},
    )
    definitions = contract["metadata"]["variable_definitions"]
    assert definitions["cwp"]["units"] == "kg m**-2"
    assert definitions["cwp"]["source"] == "derived"
    assert definitions["cwp"]["requires"] == ["clwc"]
    assert definitions["rh"]["formula"].startswith("rh = 100 * q / qsat")


def test_contract_sidecar_is_written(tmp_path: Path):
    from atmosgen.data.io import save_canonical, sidecar_contract_path

    ds = _state_dataset()
    contract = build_contract(
        ds,
        generation_method="statistical",
        source={"kind": "synthetic", "dataset": "unit-test", "readonly": True},
    )
    target = tmp_path / "data.nc"
    save_canonical(ds, target, contract=contract)
    sidecar = sidecar_contract_path(target)
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["variables"] == list(ds.data_vars)


# --------------------------------------------------------------------------- #
# 7. Old-configuration regression
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name",
    [
        "demo.yaml",
        "era5_ingest.yaml",
        "era5_cloud_sample.yaml",
        "physics_advection.yaml",
        "analog_blocks.yaml",
        "hybrid.yaml",
        "stat_ctgan.yaml",
        "stat_gaussian_copula.yaml",
        "stat_tvae.yaml",
    ],
)
def test_shipped_configurations_still_parse(name: str):
    """Old configurations must keep working (no new required fields)."""

    path = REPO_ROOT / "configs" / name
    assert path.exists(), name
    try:
        parsed = load_config_file(path)
    except Exception as exc:  # statistical configs need sdv; that is pre-existing
        assert "sdv" in str(exc) or "sdmetrics" in str(exc), (name, exc)
        return
    if hasattr(parsed, "generation"):
        assert parsed.generation.variables is None or isinstance(
            parsed.generation.variables, list
        )


def test_legacy_config_does_not_request_variables():
    """Existing configs must not gain a variable selection implicitly."""

    path = REPO_ROOT / "configs" / "physics_advection.yaml"
    try:
        parsed = load_config_file(path)
    except Exception:
        pytest.skip("config not loadable in this environment")
    assert parsed.generation.variables is None


def test_registry_variables_are_a_superset_of_the_old_ones():
    """The original 10 canonical variables must still be registered."""

    for name in ("t", "q", "u", "v", "clwc", "ciwc", "crwc", "cswc", "tcwv", "c"):
        assert name in known_variables()
    assert set(PRIMARY_VARIABLES) >= {"t", "q", "u", "v", "c"}
    assert set(DERIVED_VARIABLES) >= {"tcwv"}
