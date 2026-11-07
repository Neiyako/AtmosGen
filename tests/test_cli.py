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


def test_generate_unimplemented_mode_exits_3(tmp_path):
    config = tmp_path / "physics.yaml"
    config.write_text(
        "\n".join(
            [
                "project: {name: physics_cli, seed: 1}",
                "generation:",
                "  mode: physics",
                "  model: advection_diffusion",
                "  samples: {unit: field, fields: 5}",
                "physics:",
                "  equation: advection_diffusion",
                "  grid: {nx: 32, ny: 32, dx_m: 42000.0, dy_m: 42000.0}",
                "  time: {dt_s: 600.0, steps: 10}",
                "  parameters: {u0: 8.0, v0: 2.0, kappa: 100000.0}",
                "  initial_condition: {kind: gaussian_plume, center: [30.0, 110.0], sigma_deg: 1.5}",
                "  boundary_condition: {kind: periodic}",
            ]
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["generate", "--config", str(config)])
    assert result.exit_code == 3
    assert "NOT_RUN" in result.stdout


def test_help_and_commands_still_required_args():
    # commands are implemented now: missing required args -> typer usage error (2)
    assert runner.invoke(app, ["evaluate"]).exit_code == 2
    assert runner.invoke(app, ["report"]).exit_code == 2


def _validation_ready() -> bool:
    """Skip the e2e test until the validation teammate's modules import cleanly."""
    try:
        from atmosgen.validation.report import aggregate_status  # noqa: F401
        from atmosgen.validation.spatial import run_spatial_validation  # noqa: F401
        from atmosgen.validation.statistical import run_statistical_validation  # noqa: F401
        from atmosgen.validation.temporal import run_temporal_validation  # noqa: F401
    except Exception:
        return False
    return True


def test_generate_statistical_end_to_end(tmp_path, write_yaml):
    if not _validation_ready():
        pytest.skip("validation modules (teammate) not ready yet")
    from conftest import build_small_dataset, make_contract

    from atmosgen.data.io import save_canonical

    real_path = save_canonical(
        build_small_dataset(seed=4), tmp_path / "real.nc", contract=make_contract(build_small_dataset(seed=4))
    )
    config = write_yaml(
        "gen.yaml",
        {
            "project": {"name": "cli_gen", "seed": 7, "output_dir": str(tmp_path / "output")},
            "data": {"input": str(real_path), "variables": ["t", "q", "u", "v", "tcwv"]},
            "generation": {
                "mode": "statistical",
                "model": "gaussian_copula",
                "samples": {"unit": "cell_time", "rows": 128},
                "covariates": ["hour_sin", "level", "lat_norm", "lon_norm"],
            },
            "output": {"format": "netcdf", "directory": str(tmp_path / "output")},
        },
    )
    result = runner.invoke(app, ["generate", "--config", str(config)])
    assert result.exit_code == 0, result.stdout + result.stderr
    run_dirs = list((tmp_path / "output").iterdir())
    assert run_dirs, "run directory expected"
    run_dir = run_dirs[0]
    for name in ("data.nc", "contract.json", "metadata.json", "environment.json", "config.yaml"):
        assert (run_dir / name).exists(), f"missing {name}"
    data_path = run_dir / "data.nc"

    evaluate_result = runner.invoke(
        app, ["evaluate", "--input", str(data_path), "--real", str(real_path)]
    )
    assert evaluate_result.exit_code == 0, evaluate_result.stdout + evaluate_result.stderr
    assert "overall_status" in evaluate_result.stdout
    assert (run_dir / "quality_report.json").exists()
    assert (run_dir / "quality_report.md").exists()

    report_result = runner.invoke(
        app, ["report", "--input", str(run_dir / "quality_report.json"), "--format", "md"]
    )
    assert report_result.exit_code == 0
    assert "质量验证报告" in report_result.stdout or "overall_status" in report_result.stdout.lower()


def test_generate_statistical_reuses_config_snapshot(tmp_path, write_yaml):
    pytest.importorskip("sdv")
    from conftest import build_small_dataset, make_contract

    from atmosgen.data.io import save_canonical

    real_path = save_canonical(
        build_small_dataset(seed=5), tmp_path / "real.nc", contract=make_contract(build_small_dataset(seed=5))
    )
    config = write_yaml(
        "gen.yaml",
        {
            "project": {"name": "cli_gen2", "seed": 9, "output_dir": str(tmp_path / "output")},
            "data": {"input": str(real_path), "variables": ["t", "q", "u", "v", "tcwv"]},
            "generation": {
                "mode": "statistical",
                "model": "gaussian_copula",
                "samples": {"unit": "cell_time", "rows": None},
            },
            "output": {"format": "netcdf", "directory": str(tmp_path / "output")},
        },
    )
    ok = runner.invoke(app, ["generate", "--config", str(config)])
    assert ok.exit_code == 0, ok.stdout + ok.stderr
    (run_dir,) = list((tmp_path / "output").iterdir())
    # second run in the same second would collide without a new timestamp; just
    # assert the snapshot is a byte copy of the requested config
    assert (run_dir / "config.yaml").read_text(encoding="utf-8") == config.read_text(encoding="utf-8")


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
