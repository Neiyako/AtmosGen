#!/usr/bin/env python
"""Synthcity isolated-environment probe (Phase 2 timebox, plan §7 Phase 2).

Runs only in the ``atmosgen-synthcity`` conda env (Python 3.10 + torch 2.2 +
numpy<2), never imported by the main AtmosGen process. The goal is a PASS /
NOT_RUN verdict with real evidence: import, plugin list, a small CTGAN fit +
generation, and marginal statistics.

Usage
-----
    /opt/miniconda3/envs/atmosgen-synthcity/bin/python scripts/synthcity_probe.py \
        --out research/env/logs/p4_synthcity_probe.log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=5, help="n_iter for the synthcity ctgan")
    parser.add_argument("--out", type=Path, default=None, help="JSON report path")
    args = parser.parse_args(argv)

    report: dict = {"status": "NOT_RUN", "steps": {}}
    started = time.time()
    try:
        import numpy
        import torch

        import synthcity
        from synthcity.plugins import Plugins
        from synthcity.plugins.core.dataloader import GenericDataLoader

        report["versions"] = {
            "python": sys.version.split()[0],
            "numpy": numpy.__version__,
            "torch": torch.__version__,
            "synthcity": getattr(synthcity, "__version__", "unknown"),
            "synthcity_path": str(Path(synthcity.__file__).parent),
        }
        plugins = [str(name) for name in Plugins().list()]
        report["steps"]["import"] = "PASS"
        report["plugin_count"] = len(plugins)
        report["plugins_sample"] = plugins[:15]
        report["steps"]["plugin_list"] = "PASS" if plugins else "FAIL"

        rng = np.random.default_rng(0)
        real = pd.DataFrame(
            {
                "t": 250.0 + 25.0 * rng.random(args.rows),
                "q": 1e-3 * rng.random(args.rows),
                "u": 20.0 * (rng.random(args.rows) - 0.5),
                "v": 20.0 * (rng.random(args.rows) - 0.5),
                "tcwv": 10.0 + 40.0 * rng.random(args.rows),
            }
        )
        loader = GenericDataLoader(real)
        t0 = time.time()
        plugin = Plugins().get("ctgan", n_iter=args.iterations)
        plugin.fit(loader)
        fit_seconds = time.time() - t0
        t1 = time.time()
        generated = plugin.generate(count=args.rows)
        sample_seconds = time.time() - t1
        frame = generated.dataframe() if hasattr(generated, "dataframe") else generated
        frame = frame.head(args.rows)

        marginal = {}
        for column in real.columns:
            if column not in frame.columns:
                marginal[column] = {"error": "column missing in generated frame"}
                continue
            marginal[column] = {
                "real_mean": float(real[column].mean()),
                "gen_mean": float(frame[column].mean()),
                "real_std": float(real[column].std()),
                "gen_std": float(frame[column].std()),
            }
        report["steps"]["fit_and_generate"] = "PASS" if len(frame) >= args.rows else "FAIL"
        report["fit_seconds"] = round(fit_seconds, 2)
        report["sample_seconds"] = round(sample_seconds, 2)
        report["generated_rows"] = int(len(frame))
        report["marginals"] = marginal
        max_rel_mean_diff = max(
            abs(info["gen_mean"] - info["real_mean"]) / max(abs(info["real_mean"]), 1e-12)
            for info in marginal.values()
            if "gen_mean" in info
        )
        report["max_relative_mean_difference"] = round(float(max_rel_mean_diff), 4)
        report["status"] = "PASS"
    except Exception as exc:  # noqa: BLE001 - probe must report its own failure
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()[-4000:]
    finally:
        report["elapsed_seconds"] = round(time.time() - started, 2)
        payload = json.dumps(report, ensure_ascii=False, indent=2, default=str)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(payload, encoding="utf-8")
        print(payload)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
