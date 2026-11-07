#!/usr/bin/env python
"""Phase 5 experiment runner (plan §9): unified benchmark over the three modes.

Runs the configured experiments with identical conditions and collects
quality metrics, wall time and peak memory into one JSON summary:

* **E1** statistical model comparison (gaussian_copula / ctgan / tvae)
* **E2** physics parameter sensitivity (already covered by
  ``scripts/validate_solver_analytic.py``; referenced here for the table)
* **E3** four-way comparison (real / statistical / physics / hybrid)

Usage
-----
    export PATH=/opt/miniconda3/envs/atmosgen/bin:$PATH
    python scripts/run_experiments.py --experiments e1,e3 --out output/experiments
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atmosgen import __version__  # noqa: E402
from atmosgen.runner import environment_snapshot  # noqa: E402

E1_CONFIGS = (
    ("gaussian_copula", "configs/stat_gaussian_copula.yaml"),
    ("ctgan", "configs/stat_ctgan.yaml"),
    ("tvae", "configs/stat_tvae.yaml"),
)
PHYSICS_CONFIG = ("advection_diffusion", "configs/physics_advection.yaml")
HYBRID_CONFIG = ("hybrid", "configs/hybrid.yaml")
REAL_DATA = "data/processed/era5_local.nc"


def _run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> dict[str, Any]:
    """Run a CLI command and capture the outcome (exit code + stdout tail)."""

    started = time.perf_counter()
    process = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    elapsed = round(time.perf_counter() - started, 2)
    return {
        "cmd": " ".join(cmd),
        "exit_code": process.returncode,
        "elapsed_seconds": elapsed,
        "stdout_tail": process.stdout.strip().splitlines()[-12:],
        "stderr_tail": process.stderr.strip().splitlines()[-6:],
    }


def _latest_run(prefix: str, output_root: Path) -> Path | None:
    candidates = sorted(output_root.glob(f"{prefix}*"))
    return candidates[-1] if candidates else None


def _quality_report(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "quality_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics_table(report: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten the four modules into the comparable metric subset."""

    if report is None:
        return {}
    keys = (
        ("statistical", "column_shapes"),
        ("statistical", "column_pair_trends"),
        ("statistical", "ks"),
        ("statistical", "mean_ratio"),
        ("statistical", "std_ratio"),
        ("statistical", "corr_frobenius"),
        ("temporal", "acf_diff"),
        ("temporal", "diff_var_ratio"),
        ("spatial", "variogram_ratio"),
        ("spatial", "gradient_rms_ratio"),
        ("physical", "finite"),
        ("physical", "saturation"),
        ("physical", "mass_budget"),
    )
    flat: dict[str, Any] = {}
    for module, metric in keys:
        block = report.get(module) or {}
        entry = (block.get("metrics") or {}).get(metric)
        if entry is None:
            continue
        flat[f"{module}.{metric}"] = {
            "value": entry.get("value"),
            "status": entry.get("status"),
        }
    flat["_statuses"] = {
        module: (report.get(module) or {}).get("status") for module in
        ("statistical", "temporal", "spatial", "physical")
    }
    flat["_overall_status"] = report.get("overall_status")
    return flat


def experiment_e1(output_root: Path, *, evaluate: bool) -> dict[str, Any]:
    """E1: statistical model comparison under identical conditions."""

    results: list[dict[str, Any]] = []
    for model, config_path in E1_CONFIGS:
        entry: dict[str, Any] = {"model": model, "config": config_path}
        generate = _run(["atmosgen", "generate", "--config", config_path])
        entry["generate"] = generate
        if generate["exit_code"] != 0:
            results.append(entry)
            continue
        run_dir = _latest_run(f"statistical_{model}_", REPO_ROOT / "output")
        if run_dir is None:
            results.append(entry)
            continue
        entry["run_dir"] = str(run_dir)
        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            entry["fit_resources"] = metadata.get("fit_resources")
            entry["sample_resources"] = metadata.get("resources")
        if evaluate:
            evaluate_run = _run(
                [
                    "atmosgen",
                    "evaluate",
                    "--input",
                    str(run_dir / "data.nc"),
                    "--real",
                    REAL_DATA,
                    "--overwrite",
                    "--fail-on",
                    "never",
                ]
            )
            entry["evaluate"] = evaluate_run
            report = _quality_report(run_dir)
            entry["metrics"] = _metrics_table(report)
        results.append(entry)
    return {"name": "E1", "description": "统计模型对比（gaussian_copula / ctgan / tvae）", "runs": results}


def experiment_e2(output_root: Path, *, evaluate: bool) -> dict[str, Any]:
    """E2: physics parameter sensitivity (analytic validation script)."""

    target = output_root / "e2_solver_validation"
    run = _run(["python", "scripts/validate_solver_analytic.py", "--out", str(target)])
    payload: dict[str, Any] = {}
    report_path = target / "report.json"
    if report_path.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "name": "E2",
        "description": "物理参数敏感性（解析解验证 + Test C）",
        "run": run,
        "report": payload,
    }


def experiment_e3(output_root: Path, *, evaluate: bool) -> dict[str, Any]:
    """E3: four-way comparison (real / statistical / physics / hybrid)."""

    entries: list[dict[str, Any]] = []
    for label, config_path, prefix in (
        ("statistical", "configs/stat_gaussian_copula.yaml", "statistical_gaussian_copula_"),
        ("physics", "configs/physics_advection.yaml", "physics_advection_diffusion_"),
        ("hybrid", "configs/hybrid.yaml", "hybrid_hybrid_"),
    ):
        entry: dict[str, Any] = {"label": label, "config": config_path}
        generate = _run(["atmosgen", "generate", "--config", config_path])
        entry["generate"] = generate
        run_dir = _latest_run(prefix, REPO_ROOT / "output") if generate["exit_code"] == 0 else None
        if run_dir is not None:
            entry["run_dir"] = str(run_dir)
            metadata_path = run_dir / "metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                entry["fit_resources"] = metadata.get("fit_resources")
                entry["sample_resources"] = metadata.get("resources")
                entry["screening_before"] = metadata.get("screening_before")
                entry["screening_after"] = metadata.get("screening_after")
            if evaluate:
                # Mode 2 outputs a tracer with a solver clock: it has no common
                # variables/time axis with the real ERA5 dataset, so it is
                # evaluated against the physical constraints only (no --real).
                eval_cmd = [
                    "atmosgen",
                    "evaluate",
                    "--input",
                    str(run_dir / "data.nc"),
                    "--overwrite",
                    "--fail-on",
                    "never",
                ]
                if label != "physics":
                    eval_cmd.extend(["--real", REAL_DATA])
                entry["evaluate"] = _run(eval_cmd)
                entry["metrics"] = _metrics_table(_quality_report(run_dir))
        entries.append(entry)
    return {
        "name": "E3",
        "description": "真实/统计/物理/混合四组对比",
        "real_reference": REAL_DATA,
        "runs": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", default="e1,e2,e3", help="逗号分隔：e1,e2,e3")
    parser.add_argument("--out", type=Path, default=None, help="输出目录")
    parser.add_argument("--no-evaluate", action="store_true", help="只生成不评价")
    args = parser.parse_args(argv)

    output_root = args.out or (REPO_ROOT / "output" / f"experiments_{datetime.now():%Y%m%d-%H%M%S}")
    output_root.mkdir(parents=True, exist_ok=True)
    selected = {item.strip().lower() for item in args.experiments.split(",") if item.strip()}
    summary: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "atmosgen_version": __version__,
        "environment": environment_snapshot(),
        "experiments": [],
    }
    if "e1" in selected:
        summary["experiments"].append(experiment_e1(output_root, evaluate=not args.no_evaluate))
    if "e2" in selected:
        summary["experiments"].append(experiment_e2(output_root, evaluate=not args.no_evaluate))
    if "e3" in selected:
        summary["experiments"].append(experiment_e3(output_root, evaluate=not args.no_evaluate))
    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    path = output_root / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nsummary: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
