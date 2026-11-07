#!/usr/bin/env python
"""合成数据保真度诊断：面向云水资源量等**导数型**下游计算的可用性评估。

单表合成模型（SDV）可以做到"边缘分布几乎一致"，但云水资源量这类量是
``MFC = -∇·(q·V)``（水汽通量辐合）——**空间导数 + 垂直积分 + 时间累积**。
对无时空协方差的场求散度，会把独立噪声放大到与真实信号同量级：

* 图案相关（与真实 MFC）可能接近 0，而
* 量级比仍接近 1（"看起来对、逐格全错"）。

本脚本把这件事量化并可复现：边缘 KS / 时序相关 / 空间图案相关 / lag-1 ACF /
饱和违例率，以及 MFC 的图案相关、幅度比与域总量比。

用法
----
    export PATH=/opt/miniconda3/envs/atmosgen/bin:$PATH
    python scripts/quality_fidelity.py \
        --real data/processed/era5_local.nc \
        --runs output/statistical_gaussian_copula_*/data.nc \
               output/hybrid_hybrid_*/data.nc \
        --out output/quality_fidelity

输出：``report.json``（机器可读）+ ``report.md``（论文可直接用的表）+ ``figures/``（可选图）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atmosgen.data.selection import select_subset  # noqa: E402
from atmosgen.data.tabular import thin_grid  # noqa: E402

MFC_LEVELS: tuple[float, ...] = (500.0, 700.0, 850.0)
VARIABLES: tuple[str, ...] = ("t", "q")


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def load_dataset(path: str | Path) -> xr.Dataset:
    """Load a canonical dataset (contract not required for these diagnostics)."""

    return xr.open_dataset(path, chunks=None).load()


def align_real_to(reference: xr.Dataset, target: xr.Dataset) -> tuple[xr.Dataset, list[str]]:
    """Subset ``reference`` onto the coordinates of ``target``.

    Returns the aligned dataset and the list of coordinates that could not be
    aligned (e.g. the solver clock of a Mode 2 dataset, which has no counterpart
    in the ERA5 archive). Unalignable coordinates are skipped with a note rather
    than raising, so a mixed set of runs can be diagnosed in one pass.
    """

    out = reference
    skipped: list[str] = []
    for coord in ("time", "level", "latitude", "longitude"):
        if coord not in target.coords or coord not in out.coords:
            continue
        values = np.asarray(target[coord].values)
        if values.dtype.kind == "M":
            tolerance: Any = np.timedelta64(1, "ns")
        elif coord == "level" and out[coord].dtype.kind in "iu":
            tolerance = 0.51
        else:
            tolerance = 1e-6
        try:
            out = out.sel({coord: list(values)}, method="nearest", tolerance=tolerance)
        except KeyError:
            skipped.append(coord)
    return out, skipped


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic on the finite overlap of two arrays."""

    from scipy.stats import ks_2samp

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    return float(ks_2samp(a[mask], b[mask]).statistic)


def acf_lag1(values: np.ndarray) -> float:
    """Lag-1 autocorrelation of a 1-D series (NaN-safe)."""

    series = np.asarray(values, dtype=float)
    series = series[np.isfinite(series)]
    if series.size < 3:
        return float("nan")
    a = series[:-1] - series.mean()
    b = series[1:] - series.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator else float("nan")


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return float("nan")
    if np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def moisture_flux_convergence(
    ds: xr.Dataset, levels: Sequence[float] = MFC_LEVELS
) -> np.ndarray | None:
    """MFC = -∇·(q·V) using the layer-mean moisture flux (kg kg-1 · m s-1).

    Returns an array ``(time, latitude, longitude)`` or ``None`` when the
    dataset does not carry the required variables/levels.
    """

    if "q" not in ds.data_vars or "v" not in ds.data_vars or "level" not in ds.coords:
        return None
    available = {float(value) for value in np.asarray(ds["level"].values)}
    used = [level for level in levels if level in available]
    if not used:
        return None
    fluxes = []
    for level in used:
        q = np.asarray(ds["q"].sel(level=level).values, dtype=float)
        v = np.asarray(ds["v"].sel(level=level).values, dtype=float)
        fluxes.append(q * v)
    field = np.nanmean(np.stack(fluxes), axis=0)  # (time, lat, lon)

    lat = np.asarray(ds["latitude"].values, dtype=float)
    lon = np.asarray(ds["longitude"].values, dtype=float)
    if lat.size < 3 or lon.size < 3:
        return None
    dy = abs(float(np.mean(np.diff(lat)))) * 1e5  # degree -> m
    dx = abs(float(np.mean(np.diff(lon)))) * 1e5 * np.cos(np.radians(float(lat.mean())))
    if dy <= 0 or dx <= 0:
        return None
    dflux_dy = np.gradient(field, axis=1) / dy
    dflux_dx = np.gradient(field, axis=2) / dx
    return -(dflux_dy + dflux_dx)


def saturation_violation_rate(
    ds: xr.Dataset, factor: float = 1.10, level_units_hint: str | None = None
) -> dict[str, Any]:
    """Share of cells violating ``q <= factor * q_sat(t, p)`` (Tetens).

    ``level_units_hint`` supplies the pressure unit when the generated dataset
    does not carry one (the canonical contract records it, but a plain dataset
    may not).
    """

    if "q" not in ds.data_vars or "t" not in ds.data_vars:
        return {"rate": None, "reason": "缺 q 或 t"}
    if "level" not in ds.coords:
        return {"rate": None, "reason": "缺 level 坐标"}
    units = str(ds["level"].attrs.get("units", "") or level_units_hint or "").lower()
    if units in {"", "1"}:
        return {"rate": None, "reason": f"level 单位不可识别（{units!r}），不做气压换算"}
    scale = 0.01 if units == "pa" else 1.0
    pressures = np.asarray(ds["level"].values, dtype=float) * scale
    total = 0
    violations = 0
    for index, pressure in enumerate(pressures):
        q = np.asarray(ds["q"].isel(level=index).values, dtype=float)
        t = np.asarray(ds["t"].isel(level=index).values, dtype=float)
        e_s = 6.112 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
        denominator = pressure - 0.378 * e_s
        with np.errstate(divide="ignore", invalid="ignore"):
            q_sat = 0.622 * e_s / denominator
        usable = np.isfinite(q_sat) & (q_sat > 0) & np.isfinite(q) & np.isfinite(t)
        total += int(usable.sum())
        violations += int(np.count_nonzero(usable & (q > factor * q_sat)))
    if total == 0:
        return {"rate": None, "reason": "无可用配对点"}
    return {"rate": violations / total, "violations": violations, "pairs": total}


# --------------------------------------------------------------------------- #
# 单个 run 的诊断
# --------------------------------------------------------------------------- #
def diagnose_run(
    real: xr.Dataset,
    generated: xr.Dataset,
    *,
    variables: Sequence[str] = VARIABLES,
    mfc_levels: Sequence[float] = MFC_LEVELS,
) -> dict[str, Any]:
    """Compute the fidelity block for one generated dataset."""

    aligned_real, skipped = align_real_to(real, generated)
    block: dict[str, Any] = {"notes": []}
    if skipped:
        block["notes"].append(
            "无法对齐的坐标: " + ", ".join(skipped)
            + "（物理模式使用求解器合成时钟，与真实归档无对应；相关指标已跳过）"
        )

    # 1) marginal fidelity (KS per variable)
    ks: dict[str, float] = {}
    for name in variables:
        if name not in generated.data_vars or name not in aligned_real.data_vars:
            continue
        ks[name] = ks_statistic(
            np.asarray(aligned_real[name].values),
            np.asarray(generated[name].values),
        )
    block["ks"] = ks

    # 2) temporal fidelity: correlation of the level-0/mid-grid series
    if "level" in generated.coords and "time" in generated.coords and "time" not in skipped:
        index = {
            "level": 0,
            "latitude": generated.sizes["latitude"] // 2,
            "longitude": generated.sizes["longitude"] // 2,
        }
        name = "t" if "t" in generated.data_vars else (variables[0] if variables else None)
        if name:
            real_series = np.asarray(aligned_real[name].isel(**index).values, dtype=float)
            gen_series = np.asarray(generated[name].isel(**index).values, dtype=float)
            block["temporal_series_correlation"] = _safe_corr(real_series, gen_series)
            block["acf1_real"] = acf_lag1(real_series)
            block["acf1_generated"] = acf_lag1(gen_series)
    else:
        block["notes"].append("无 level/time 维，跳过时序与 ACF 指标")

    # 3) spatial fidelity: pattern correlation at the last time step
    if "level" in generated.coords and "time" in generated.coords and "time" not in skipped:
        name = "t" if "t" in generated.data_vars else (variables[0] if variables else None)
        if name:
            real_field = np.asarray(
                aligned_real[name].isel(level=0, time=-1).values, dtype=float
            )
            gen_field = np.asarray(
                generated[name].isel(level=0, time=-1).values, dtype=float
            )
            block["spatial_pattern_correlation"] = _safe_corr(real_field, gen_field)
    else:
        block["notes"].append("无 level/time 维，跳过空间图案指标")

    # 4) physical screening (saturation)
    level_units = None
    if "level" in aligned_real.coords:
        level_units = str(aligned_real["level"].attrs.get("units", "") or "") or None
    block["saturation"] = saturation_violation_rate(generated, level_units_hint=level_units)

    # 5) derivative-type quantity: moisture flux convergence
    real_mfc = moisture_flux_convergence(aligned_real, mfc_levels)
    gen_mfc = moisture_flux_convergence(generated, mfc_levels)
    if real_mfc is None or gen_mfc is None:
        block["mfc"] = {"available": False, "reason": "数据缺少 q/v 或气压层（物理模式输出的是示迹浓度）"}
    else:
        a = np.nanmean(real_mfc, axis=0).ravel()
        b = np.nanmean(gen_mfc, axis=0).ravel()
        real_total = float(np.nansum(np.nanmean(real_mfc, axis=0)))
        gen_total = float(np.nansum(np.nanmean(gen_mfc, axis=0)))
        block["mfc"] = {
            "available": True,
            "pattern_correlation": _safe_corr(a, b),
            "magnitude_ratio": float(np.nanstd(b) / np.nanstd(a)) if np.nanstd(a) else float("nan"),
            "domain_total_ratio": (gen_total / real_total) if real_total else float("nan"),
            "real_spatial_std": float(np.nanstd(a)),
            "generated_spatial_std": float(np.nanstd(b)),
        }
    return block


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    """Markdown table usable as a thesis results table."""

    lines = [
        "# 合成数据保真度诊断（云水资源量导向）",
        "",
        f"- 生成时间：{report.get('generated_at')}",
        f"- 真实参考：{report.get('real')}",
        f"- 诊断脚本：`scripts/quality_fidelity.py`",
        "",
        "| 数据集 | KS(t) | KS(q) | 时序相关 r | 空间图案 r | ACF1 真实/生成 | 饱和违例 | MFC 图案 r | MFC 幅度比 | MFC 域总量比 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in report.get("runs", []):
        block = entry["fidelity"]
        mfc = block.get("mfc", {})
        saturation = block.get("saturation", {})
        rate = saturation.get("rate")
        lines.append(
            "| {label} | {ks_t} | {ks_q} | {rt} | {rs} | {acf_real}/{acf_gen} | {sat} | {mfc_r} | {mfc_amp} | {mfc_tot} |".format(
                label=entry["label"],
                ks_t=_fmt(block.get("ks", {}).get("t")),
                ks_q=_fmt(block.get("ks", {}).get("q")),
                rt=_fmt(block.get("temporal_series_correlation")),
                rs=_fmt(block.get("spatial_pattern_correlation")),
                acf_real=_fmt(block.get("acf1_real")),
                acf_gen=_fmt(block.get("acf1_generated")),
                sat=("n/a" if rate is None else f"{rate*100:.2f}%"),
                mfc_r=_fmt(mfc.get("pattern_correlation")) if mfc.get("available") else "n/a",
                mfc_amp=_fmt(mfc.get("magnitude_ratio")) if mfc.get("available") else "n/a",
                mfc_tot=_fmt(mfc.get("domain_total_ratio")) if mfc.get("available") else "n/a",
            )
        )
    lines += [
        "",
        "## 读法",
        "",
        "- **KS 小 / MFC 图案 r≈0**：边缘分布几乎一致，但导数型量与真实无关——"
        "散度把无结构噪声放大到与信号同量级（幅度比接近 1），“看着像、逐格全错”。",
        "- **ACF1 由 0.89 变负**：生成序列独立采样，丢失天气过程持续性 → 年际变化/极端事件分析不可用。",
        "- **空间图案 r**：真实场有天气系统，统计生成没有；混合模式靠物理基底可部分恢复。",
        "",
    ]
    return "\n".join(lines)


def write_figure(report: Mapping[str, Any], out_dir: Path) -> str | None:
    """Optional figure: MFC pattern of real vs generated (best effort)."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return None
    entries = [e for e in report.get("runs", []) if e["fidelity"].get("mfc", {}).get("available")]
    if not entries:
        return None
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(entries) + 1, figsize=(4.0 * (len(entries) + 1), 3.4))
    axes = np.atleast_1d(axes)
    real_mfc = report.get("_real_mfc")
    if real_mfc is not None:
        image = axes[0].imshow(real_mfc)
        axes[0].set_title("real MFC")
        figure.colorbar(image, ax=axes[0])
    else:  # pragma: no cover
        axes[0].axis("off")
    for axis, entry in zip(axes[1:], entries):
        pattern = entry.get("_mfc_map")
        if pattern is None:
            axis.axis("off")
            continue
        image = axis.imshow(pattern)
        axis.set_title(f"{entry['label']} MFC")
        figure.colorbar(image, ax=axis)
    figure.tight_layout()
    target = fig_dir / "mfc_real_vs_generated.png"
    figure.savefig(target, dpi=120)
    plt.close(figure)
    return str(target)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", required=True, help="真实参考数据集（规范 NetCDF）")
    parser.add_argument("--runs", nargs="+", required=True, help="待诊断的生成数据集路径（可多个）")
    parser.add_argument("--out", type=Path, default=None, help="输出目录")
    parser.add_argument("--lat-step", type=float, default=1.0, help="真实数据的抽稀步长（度）")
    parser.add_argument("--lon-step", type=float, default=1.0)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    out_dir = args.out or (REPO_ROOT / "output" / f"quality_fidelity_{datetime.now():%Y%m%d-%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)

    real = load_dataset(args.real)
    real = thin_grid(real, lat_step_deg=args.lat_step, lon_step_deg=args.lon_step)
    real_mfc = moisture_flux_convergence(real)

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "real": str(args.real),
        "runs": [],
    }
    for path in args.runs:
        generated = load_dataset(path)
        label = Path(path).parent.name
        block = diagnose_run(real, generated)
        entry = {"label": label, "path": str(path), "fidelity": block}
        gen_mfc = moisture_flux_convergence(generated)
        if gen_mfc is not None:
            entry["_mfc_map"] = np.nanmean(gen_mfc, axis=0).tolist()
        report["runs"].append(entry)
        generated.close()

    if real_mfc is not None:
        report["_real_mfc"] = np.nanmean(real_mfc, axis=0).tolist()
    figure_path = None if args.no_figure else write_figure(report, out_dir)
    if figure_path:
        report["figure"] = figure_path

    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    print(f"\nreport: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
