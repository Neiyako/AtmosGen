"""Configuration models: valid examples, cross-field rejection, dataset checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from atmosgen.config import (
    ConfigError,
    GenerationConfigFile,
    IngestConfigFile,
    check_config_against_dataset,
    load_config_file,
    validate_config_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO = REPO_ROOT / "configs" / "demo.yaml"
INGEST = REPO_ROOT / "configs" / "era5_ingest.yaml"
PHYSICS = REPO_ROOT / "configs" / "physics_advection.yaml"


def _base_generation(**overrides):
    payload = {
        "project": {"name": "demo", "seed": 42},
        "data": {
            "input": "./data/processed/era5_local.nc",
            "variables": ["t", "q", "u", "v", "tcwv"],
        },
        "generation": {
            "mode": "statistical",
            "model": "gaussian_copula",
            "samples": {"unit": "cell_time", "rows": 1000},
        },
        "validation": {"statistical": True},
        "output": {"format": "netcdf", "directory": "./output"},
    }
    payload.update(overrides)
    return payload


def test_shipped_configs_are_structurally_valid():
    for path in (DEMO, INGEST, PHYSICS):
        check = validate_config_file(path, check_data=False)
        assert check.valid, check.errors
        assert check.model is not None


def test_ingest_config_loads_as_ingest(write_yaml):
    path = write_yaml(
        "ingest.yaml",
        {
            "project": {"name": "ing", "seed": 7},
            "source": {
                "kind": "local_netcdf",
                "input": "./source.nc",
                "variable_map": {"clwc": "clwc"},
            },
            "output": {"processed_path": "./out.nc"},
        },
    )
    model = load_config_file(path)
    assert isinstance(model, IngestConfigFile)
    assert model.source.kind == "local_netcdf"


def test_field_unit_with_tabular_model_is_rejected(write_yaml):
    payload = _base_generation()
    payload["generation"]["samples"] = {"unit": "field", "fields": 10}
    path = write_yaml("bad_field.yaml", payload)
    check = validate_config_file(path, check_data=False)
    assert not check.valid
    assert any("field" in error for error in check.errors)


def test_unknown_model_is_rejected_and_lists_choices(write_yaml):
    payload = _base_generation()
    payload["generation"]["model"] = "magic_net"
    path = write_yaml("bad_model.yaml", payload)
    check = validate_config_file(path, check_data=False)
    assert not check.valid
    assert any("gaussian_copula" in error for error in check.errors)


def test_physics_mode_requires_physics_section(write_yaml):
    payload = _base_generation()
    payload["generation"] = {"mode": "physics", "model": "advection_diffusion", "samples": {"unit": "field", "fields": 5}}
    path = write_yaml("bad_physics.yaml", payload)
    check = validate_config_file(path, check_data=False)
    assert not check.valid
    assert any("physics" in error for error in check.errors)


def test_cfl_violation_is_rejected(write_yaml):
    payload = {
        "project": {"name": "phys", "seed": 1},
        "generation": {
            "mode": "physics",
            "model": "advection_diffusion",
            "samples": {"unit": "field", "fields": 10},
        },
        "physics": {
            "equation": "advection_diffusion",
            "grid": {"nx": 64, "ny": 64, "dx_m": 1000.0, "dy_m": 1000.0},
            "time": {"dt_s": 600.0, "steps": 10},
            "parameters": {"u0": 50.0, "v0": 0.0, "kappa": 0.0},
            "initial_condition": {"kind": "gaussian_plume", "center": [30, 110], "sigma_deg": 1.0},
            "boundary_condition": {"kind": "periodic"},
        },
    }
    path = write_yaml("bad_cfl.yaml", payload)
    check = validate_config_file(path, check_data=False)
    assert not check.valid
    assert any("CFL" in error for error in check.errors)


def test_threshold_out_of_range_is_rejected(write_yaml):
    payload = _base_generation()
    payload["validation"] = {"thresholds_overrides": {"ks_max": 1.5}}
    path = write_yaml("bad_threshold.yaml", payload)
    check = validate_config_file(path, check_data=False)
    assert not check.valid
    assert any("ks_max" in error for error in check.errors)


def test_ingest_config_requires_directory_for_local_era5(write_yaml):
    path = write_yaml(
        "bad_ingest.yaml",
        {
            "project": {"name": "ing"},
            "source": {"kind": "local_era5", "file_pattern": "x.nc"},
            "output": {"processed_path": "./out.nc"},
        },
    )
    check = validate_config_file(path, check_data=False)
    assert not check.valid
    assert any("root" in error or "date_range" in error for error in check.errors)


def test_local_netcdf_derive_is_rejected_not_ignored(write_yaml):
    path = write_yaml(
        "cloud_derive.yaml",
        {
            "project": {"name": "cloud"},
            "source": {
                "kind": "local_netcdf",
                "input": "./cloud.nc",
                "variable_map": {"clwc": "clwc"},
                "derive": {"tcwv": {"method": "vertical_integral_q"}},
            },
            "output": {"processed_path": "./out.nc"},
        },
    )
    check = validate_config_file(path, check_data=False)
    assert not check.valid
    assert any("derive.tcwv" in error for error in check.errors)


def test_local_netcdf_rejects_archive_only_fields(write_yaml):
    path = write_yaml(
        "cloud_archive_fields.yaml",
        {
            "project": {"name": "cloud"},
            "source": {
                "kind": "local_netcdf",
                "input": "./cloud.nc",
                "root": "/Volumes/Untitled",
                "years": [2024],
                "variable_map": {"clwc": "clwc"},
            },
            "output": {"processed_path": "./out.nc"},
        },
    )
    check = validate_config_file(path, check_data=False)
    assert not check.valid
    joined = " ".join(check.errors)
    assert "source.root" in joined and "source.years" in joined


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config_file(tmp_path / "missing.yaml")


def test_check_config_against_dataset_detects_mismatch(tmp_path, small_ds, saved_dataset):
    path = saved_dataset(small_ds, "input.nc")
    payload = _base_generation()
    payload["data"]["input"] = str(path)
    payload["data"]["variables"] = ["t", "q", "u", "v", "tcwv", "clwc"]
    model = GenerationConfigFile.model_validate(payload)
    problems = check_config_against_dataset(model)
    assert any("clwc" in problem for problem in problems)


def test_check_config_against_dataset_accepts_matching_request(tmp_path, small_ds, saved_dataset):
    path = saved_dataset(small_ds, "input.nc")
    payload = _base_generation()
    payload["data"]["input"] = str(path)
    payload["data"]["levels"] = [500.0, 700.0]
    model = GenerationConfigFile.model_validate(payload)
    assert check_config_against_dataset(model) == []
