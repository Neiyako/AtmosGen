"""CLI behaviour: exit codes, help, validate-config, ingest dispatch, NOT_RUN stubs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from typer.testing import CliRunner

from atmosgen.cli import app

REPO_ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def _generation_yaml(path: Path, model: str = "gaussian_copula") -> Path:
    path.write_text(
        "\n".join(
            [
                "project: {name: cli_demo, seed: 42}",
                "data:",
                "  input: ./data/processed/era5_local.nc",
                "  variables: [t, q, u, v, tcwv]",
                "generation:",
                "  mode: statistical",
                f"  model: {model}",
                "  samples: {unit: cell_time, rows: 100}",
                "output: {format: netcdf, directory: ./output}",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "atmosgen 0.1.0" in result.stdout


def test_list_models():
    result = runner.invoke(app, ["list-models"])
    assert result.exit_code == 0
    assert "gaussian_copula" in result.stdout
    assert "advection_diffusion" in result.stdout


def test_validate_config_pass(tmp_path):
    config = _generation_yaml(tmp_path / "demo.yaml")
    result = runner.invoke(app, ["validate-config", str(config), "--no-check-data"])
    assert result.exit_code == 0
    assert "PASS" in result.stdout


def test_validate_config_fail(tmp_path):
    config = _generation_yaml(tmp_path / "bad.yaml", model="magic_net")
    result = runner.invoke(app, ["validate-config", str(config), "--no-check-data"])
    assert result.exit_code == 2
    assert "FAIL" in result.stdout
    assert "gaussian_copula" in result.stdout


def test_validate_shipped_configs():
    for name in ("era5_ingest.yaml", "era5_cloud_sample.yaml", "physics_advection.yaml", "hybrid.yaml"):
        result = runner.invoke(
            app, ["validate-config", str(REPO_ROOT / "configs" / name), "--no-check-data"]
        )
        assert result.exit_code == 0, (name, result.stdout)


def test_ingest_rejects_generation_config(tmp_path):
    config = _generation_yaml(tmp_path / "gen.yaml")
    result = runner.invoke(app, ["ingest", "--config", str(config)])
    assert result.exit_code == 2
    assert "不是 ingest 配置" in result.stdout + result.stderr


def test_generate_is_explicit_not_run(tmp_path):
    config = _generation_yaml(tmp_path / "gen.yaml")
    result = runner.invoke(app, ["generate", "--config", str(config)])
    assert result.exit_code == 3
    assert "NOT_RUN" in result.stdout


def test_evaluate_and_report_are_not_run():
    assert runner.invoke(app, ["evaluate"]).exit_code == 3
    assert runner.invoke(app, ["report"]).exit_code == 3


def test_cli_ingest_end_to_end_local_netcdf(tmp_path, write_yaml):
    source = tmp_path / "sample.nc"
    ds = xr.Dataset(
        {
            "clwc": (
                ("valid_time", "pressure_level", "latitude", "longitude"),
                np.full((1, 2, 2, 2), 1e-4),
                {"units": "kg kg**-1", "long_name": "Specific cloud liquid water content"},
            )
        },
        coords={
            "valid_time": (("valid_time",), np.array(["2024-01-01T00"], dtype="datetime64[ns]")),
            "pressure_level": (("pressure_level",), [500.0, 700.0], {"units": "hPa"}),
            "latitude": (("latitude",), [25.0, 30.0], {"units": "degrees_north"}),
            "longitude": (("longitude",), [70.0, 75.0], {"units": "degrees_east"}),
        },
    )
    ds.to_netcdf(source, engine="netcdf4")
    config = write_yaml(
        "cloud.yaml",
        {
            "project": {"name": "cli_cloud", "seed": 1, "output_dir": str(tmp_path / "output")},
            "source": {"kind": "local_netcdf", "input": str(source), "variable_map": {"clwc": "clwc"}},
            "output": {"processed_path": str(tmp_path / "processed" / "cloud.nc")},
        },
    )
    result = runner.invoke(app, ["ingest", "--config", str(config)])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "status: PASS" in result.stdout
    assert (tmp_path / "processed" / "cloud.nc").exists()
    assert list((tmp_path / "output").iterdir()), "run directory with the report should exist"
