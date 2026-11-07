#!/usr/bin/env python
"""Phase 1 ingest driver: local ERA5 archive → canonical dataset (Test A).

Usage
-----
    python scripts/ingest_local_era5.py --config configs/era5_ingest.yaml
    python scripts/ingest_local_era5.py --config configs/era5_ingest.yaml --overwrite

This is a thin wrapper around ``atmosgen ingest`` so the same code path is used
by the CLI and by automation (plan §7 Phase 1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow running without `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT))

from atmosgen.cli import run_ingest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="ingest YAML 配置")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出")
    args = parser.parse_args(argv)
    return run_ingest(config_path=args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
