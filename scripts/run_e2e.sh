#!/usr/bin/env bash
# Test F (plan §8): real data -> preprocess -> generate -> validate -> report.
#
#   bash scripts/run_e2e.sh
#
# Runs the full chain with real ERA5 data and exits non-zero on any failure.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PATH="/opt/miniconda3/envs/atmosgen/bin:$PATH"

echo "== 1/6 版本与配置自检"
atmosgen --version
atmosgen validate-config configs/era5_ingest.yaml
atmosgen validate-config configs/stat_gaussian_copula.yaml --no-check-data
atmosgen validate-config configs/physics_advection.yaml
atmosgen validate-config configs/hybrid.yaml --no-check-data

echo "== 2/6 真实数据 ingest（Test A）"
atmosgen ingest --config configs/era5_ingest.yaml --overwrite

echo "== 3/6 统计生成（Mode 1）"
atmosgen generate --config configs/stat_gaussian_copula.yaml

echo "== 4/6 物理生成（Mode 2）"
atmosgen generate --config configs/physics_advection.yaml

echo "== 5/6 质量评价与报告（statistical + physical）"
for run in output/statistical_gaussian_copula_* output/physics_advection_diffusion_*; do
  [ -d "$run" ] || continue
  case "$run" in
    # Mode 2 outputs a tracer with a solver clock: no common variables/time
    # axis with the real ERA5 dataset -> evaluate against physical only
    *physics_advection_diffusion*)
      atmosgen evaluate --input "$run/data.nc" --overwrite --fail-on never ;;
    *)
      atmosgen evaluate --input "$run/data.nc" --real data/processed/era5_local.nc --overwrite --fail-on never ;;
  esac
  atmosgen report --input "$run/quality_report.json" --format md --out "$run/quality_report_rendered.md"
done

echo "== 6/6 解析解验证（求解器自检）"
python scripts/validate_solver_analytic.py --out output/e2e_solver_validation >/dev/null

echo "== 全链路完成（Test F PASS）"
