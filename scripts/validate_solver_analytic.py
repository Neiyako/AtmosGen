#!/usr/bin/env python
"""Validate the self-written advection-diffusion solver against analytic solutions.

Covers plan §4.2 validation list and Test C:

1. pure diffusion vs the analytic Gaussian solution at 3 resolutions
   (records L2 relative error and observed convergence order);
2. constant-wind advection: plume centroid displacement == u*T (error < 2%);
3. periodic mass conservation (max relative drift);
4. stability at the maximum CFL (no NaN/Inf, finite fields);
5. Test C: changing IC/parameters produces measurable output differences.

Usage
-----
    export PATH=/opt/miniconda3/envs/atmosgen/bin:$PATH
    python scripts/validate_solver_analytic.py [--out output/solver_validation_<ts>]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atmosgen.generators.physics.advection_diffusion import (  # noqa: E402
    PhysicsError,
    check_stability,
    solve_advection_diffusion,
)
from atmosgen.generators.physics.conditions import METERS_PER_DEGREE  # noqa: E402

CENTER = [30.0, 110.0]


def _gridprofile(nx: int, ny: int, dx_m: float, dy_m: float):
    return {"nx": nx, "ny": ny, "dx_m": dx_m, "dy_m": dy_m}


def _plume_ic(center=CENTER, sigma_deg: float = 0.45, amplitude: float = 1.0):
    return {
        "kind": "gaussian_plume",
        "center": list(center),
        "sigma_deg": sigma_deg,
        "amplitude": amplitude,
    }


def _xy(dataset):
    lat = np.asarray(dataset["latitude"].values, dtype=float)
    lon = np.asarray(dataset["longitude"].values, dtype=float)
    center_lat, center_lon = CENTER
    x = (lon - center_lon) * METERS_PER_DEGREE * np.cos(np.radians(center_lat))
    y = (lat - center_lat) * METERS_PER_DEGREE
    return x, y


def analytic_diffusion(dataset, *, sigma0_deg: float, kappa: float, t_seconds: float, amplitude: float = 1.0):
    """2-D Gaussian diffusion analytic solution on the dataset grid."""

    x, y = _xy(dataset)
    sigma0 = sigma0_deg * METERS_PER_DEGREE
    variance = sigma0**2 + 2.0 * kappa * t_seconds
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return amplitude * (sigma0**2 / variance) * np.exp(-(xx**2 + yy**2) / (2.0 * variance))


def test_diffusion_convergence() -> dict:
    """Pure diffusion vs analytic, 3 resolutions; return errors + order."""

    sigma0_deg = 0.45
    sigma0 = sigma0_deg * METERS_PER_DEGREE
    kappa = 1.0e5
    t_end = 20_000.0
    domain_m = 1_000_000.0
    rows = []
    for nx, ny in ((50, 50), (100, 100), (200, 200)):
        dx = domain_m / nx
        dy = domain_m / ny
        # 2-D stability: kappa*dt*(1/dx^2+1/dy^2) <= 0.5; keep 0.4 headroom
        dt = min(0.4 * 0.5 * dx**2 / (2.0 * kappa), 2000.0)  # square grid
        steps = int(round(t_end / dt))
        dataset, diagnostics = solve_advection_diffusion(
            grid=_gridprofile(nx, ny, dx, dy),
            time={"dt_s": dt, "steps": steps},
            parameters={"u0": 0.0, "v0": 0.0, "kappa": kappa},
            initial_condition=_plume_ic(sigma_deg=sigma0_deg),
            boundary_condition={"kind": "periodic"},
        )
        numeric = np.asarray(dataset["c"].values[-1], dtype=float)
        analytic = analytic_diffusion(
            dataset, sigma0_deg=sigma0_deg, kappa=kappa, t_seconds=steps * dt
        )
        denom = np.linalg.norm(analytic)
        error = float(np.linalg.norm(numeric - analytic) / denom) if denom else float("nan")
        rows.append(
            {
                "nx": nx,
                "dx_m": dx,
                "dt_s": dt,
                "steps": steps,
                "t_end_s": steps * dt,
                "l2_relative_error": error,
                "mass_drift_max": diagnostics.mass_drift_max,
            }
        )
    order = [
        float(math.log(rows[i]["l2_relative_error"] / rows[i + 1]["l2_relative_error"])
              / math.log(rows[i]["dx_m"] / rows[i + 1]["dx_m"]))
        for i in range(len(rows) - 1)
    ]
    return {"cases": rows, "observed_order": order, "pass": all(r["l2_relative_error"] < 0.05 for r in rows)}


def test_advection_displacement() -> dict:
    """Constant wind: measured centroid displacement vs u*T (error < 2%)."""

    dx = 2000.0
    nx = ny = 256  # 512 km domain
    dt = 100.0
    u0 = 10.0
    v0 = 2.0
    steps = 120  # T = 12 000 s -> 120 km, well inside half the domain
    dataset, diagnostics = solve_advection_diffusion(
        grid=_gridprofile(nx, ny, dx, dx),
        time={"dt_s": dt, "steps": steps},
        parameters={"u0": u0, "v0": v0, "kappa": 0.0},
        initial_condition=_plume_ic(sigma_deg=0.45),
        boundary_condition={"kind": "periodic"},
    )
    field = np.asarray(dataset["c"].values[-1], dtype=float)
    x, y = _xy(dataset)
    total = field.sum()
    centroid_x = float((field * x[None, :]).sum() / total)
    centroid_y = float((field * y[:, None]).sum() / total)
    expected_x = u0 * dt * steps
    expected_y = v0 * dt * steps
    error_x = abs(centroid_x - expected_x) / expected_x
    error_y = abs(centroid_y - expected_y) / abs(expected_y) if expected_y else 0.0
    return {
        "u0": u0,
        "v0": v0,
        "t_end_s": dt * steps,
        "expected_displacement_x_m": expected_x,
        "measured_displacement_x_m": centroid_x,
        "relative_error_x": error_x,
        "expected_displacement_y_m": expected_y,
        "measured_displacement_y_m": centroid_y,
        "relative_error_y": error_y,
        "mass_drift_max": diagnostics.mass_drift_max,
        "pass": error_x < 0.02 and error_y < 0.02,
    }


def test_mass_conservation() -> dict:
    """Periodic boundary: total mass drift per step must be tiny."""

    dx = 42_000.0
    dt = 600.0
    steps = 240
    dataset, diagnostics = solve_advection_diffusion(
        grid=_gridprofile(64, 64, dx, dx),
        time={"dt_s": dt, "steps": steps},
        parameters={"u0": 8.0, "v0": 2.0, "kappa": 1.0e5},
        initial_condition=_plume_ic(sigma_deg=1.5),
        boundary_condition={"kind": "periodic"},
    )
    series = np.asarray(diagnostics.mass_series, dtype=float)
    reference = series[0]
    max_relative_drift = float(np.max(np.abs(series - reference)) / abs(reference))
    per_step = np.abs(np.diff(series)) / abs(reference)
    return {
        "steps": steps,
        "max_relative_drift": max_relative_drift,
        "max_relative_drift_per_step": float(per_step.max()),
        "pass": max_relative_drift < 1e-6,
    }


def test_stability_at_max_cfl() -> dict:
    """CFL=1.0 with zero diffusion: fields must stay finite (no blow-up)."""

    dx = 42_000.0
    dt = 600.0
    u0 = dx / dt  # CFL exactly 1.0
    steps = 100
    dataset, diagnostics = solve_advection_diffusion(
        grid=_gridprofile(64, 64, dx, dx),
        time={"dt_s": dt, "steps": steps},
        parameters={"u0": u0, "v0": 0.0, "kappa": 0.0},
        initial_condition=_plume_ic(sigma_deg=1.5),
        boundary_condition={"kind": "periodic"},
    )
    values = np.asarray(dataset["c"].values, dtype=float)
    finite = bool(np.all(np.isfinite(values)))
    peak = float(np.max(np.abs(values)))
    return {
        "cfl_x": diagnostics.cfl_x,
        "steps": steps,
        "finite": finite,
        "max_abs": peak,
        "mass_drift_max": diagnostics.mass_drift_max,
        "pass": finite and peak < 1e3,
    }


def _field_statistics(dataset) -> dict:
    field = np.asarray(dataset["c"].values[-1], dtype=float)
    x, y = _xy(dataset)
    total = field.sum()
    centroid = [
        float((field * x[None, :]).sum() / total),
        float((field * y[:, None]).sum() / total),
    ]
    variance_x = float((field * (x[None, :] - centroid[0]) ** 2).sum() / total)
    variance_y = float((field * (y[:, None] - centroid[1]) ** 2).sum() / total)
    return {
        "centroid_x_m": centroid[0],
        "centroid_y_m": centroid[1],
        "std_x_m": math.sqrt(max(variance_x, 0.0)),
        "std_y_m": math.sqrt(max(variance_y, 0.0)),
        "max": float(field.max()),
        "mass": float(total),
    }


def test_c_parameter_sensitivity() -> dict:
    """Test C: change IC/parameters -> measurable, quantified differences."""

    base = dict(
        grid=_gridprofile(64, 64, 42_000.0, 42_000.0),
        time={"dt_s": 600.0, "steps": 96},
        boundary_condition={"kind": "periodic"},
    )
    scenarios = {
        "base": dict(parameters={"u0": 8.0, "v0": 2.0, "kappa": 1.0e5}, initial_condition=_plume_ic()),
        "shifted_ic": dict(
            parameters={"u0": 8.0, "v0": 2.0, "kappa": 1.0e5},
            initial_condition=_plume_ic(center=[32.0, 112.0]),
        ),
        "strong_diffusion": dict(
            parameters={"u0": 8.0, "v0": 2.0, "kappa": 5.0e5},
            initial_condition=_plume_ic(),
        ),
        "strong_wind": dict(
            parameters={"u0": 20.0, "v0": -4.0, "kappa": 1.0e5},
            initial_condition=_plume_ic(),
        ),
    }
    results: dict[str, dict] = {}
    for name, scenario in scenarios.items():
        dataset, diagnostics = solve_advection_diffusion(
            grid=base["grid"],
            time=base["time"],
            parameters=scenario["parameters"],
            initial_condition=scenario["initial_condition"],
            boundary_condition=base["boundary_condition"],
        )
        stats = _field_statistics(dataset)
        stats.update(
            {
                "u0": scenario["parameters"]["u0"],
                "v0": scenario["parameters"]["v0"],
                "kappa": scenario["parameters"]["kappa"],
                "ic_center": scenario["initial_condition"].get("center"),
                "mass_drift_max": diagnostics.mass_drift_max,
            }
        )
        results[name] = stats
    base_stats = results["base"]
    deltas = {
        name: {
            "centroid_shift_m": math.hypot(
                stats["centroid_x_m"] - base_stats["centroid_x_m"],
                stats["centroid_y_m"] - base_stats["centroid_y_m"],
            ),
            "std_x_ratio": stats["std_x_m"] / base_stats["std_x_m"],
            "std_y_ratio": stats["std_y_m"] / base_stats["std_y_m"],
        }
        for name, stats in results.items()
        if name != "base"
    }
    changed = all(
        (delta["centroid_shift_m"] > 1.0 or abs(delta["std_x_ratio"] - 1.0) > 0.05)
        for delta in deltas.values()
    )
    return {"scenarios": results, "deltas_vs_base": deltas, "pass": changed}


def test_cfl_rejection() -> dict:
    """Unstable parameters must raise PhysicsError, not silently reduce dt."""

    try:
        check_stability(u0=100.0, v0=0.0, kappa=0.0, dt_s=600.0, dx_m=42_000.0, dy_m=42_000.0)
        return {"pass": False, "detail": "CFL=1.43 未被拒绝"}
    except PhysicsError as exc:
        return {"pass": True, "detail": str(exc)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    started = datetime.now()
    report: dict = {"started_at": started.isoformat(timespec="seconds"), "tests": {}}
    try:
        report["tests"]["diffusion_convergence"] = test_diffusion_convergence()
        report["tests"]["advection_displacement"] = test_advection_displacement()
        report["tests"]["mass_conservation"] = test_mass_conservation()
        report["tests"]["stability_max_cfl"] = test_stability_at_max_cfl()
        report["tests"]["c_parameter_sensitivity"] = test_c_parameter_sensitivity()
        report["tests"]["cfl_rejection"] = test_cfl_rejection()
    except Exception as exc:  # noqa: BLE001 - report the failure explicitly
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["status"] = "FAIL"
    else:
        report["status"] = "PASS" if all(item.get("pass") for item in report["tests"].values()) else "FAIL"
    report["finished_at"] = datetime.now().isoformat(timespec="seconds")

    out_dir = args.out or (REPO_ROOT / "output" / f"solver_validation_{started:%Y%m%d-%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nreport: {out_dir / 'report.json'}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
