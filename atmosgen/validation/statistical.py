"""统计分布一致性验证（实现计划 §5.1, ``validation/statistical.py``）。

口径说明
--------
- 列构造：对每个数据变量展平成列；有 ``level`` 维的变量按层拆为
  ``f"{var}@{level}"``，无 level 的变量（如 tcwv）为 ``"{var}"``；
- NaN 处理：保留不插值；real 与 synthetic 各自按列压缩 NaN 后再统计；
  任一侧缺失比例 > 50% 的列整体剔除（写进 notes），全部列被剔除 → 指标 NOT_RUN；
- 阈值：传入的 ``thresholds`` 键见 :data:`DEFAULT_THRESHOLDS`，缺省值来自计划 §5.1。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import stats

from .report import aggregate_status

__all__ = ["DEFAULT_THRESHOLDS", "run_statistical_validation"]

#: 计划 §5.1 默认阈值（键名与 atmosgen.config.THRESHOLD_LIMITS 对齐的子集）。
DEFAULT_THRESHOLDS: dict[str, float] = {
    "column_shapes_min": 0.80,
    "column_pair_trends_min": 0.80,
    "mean_ratio_min": 0.85,
    "mean_ratio_max": 1.15,
    "std_ratio_min": 0.75,
    "std_ratio_max": 1.25,
    "quantile_rmse_max": 0.10,
    "ks_max": 0.15,
    "corr_frobenius_max": 0.25,
    "extreme_ratio_min": 0.5,
    "extreme_ratio_max": 2.0,
}

_NAN_LIMIT = 0.5  # 缺失比例超过 50% → 该列不可用
_EPS = 1e-12
_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
_MIN_CORR_ROWS = 30
_DETAIL_CAP = 10  # detail 里最多列出的条目数

_RANGES = (
    ("mean_ratio", ("mean_ratio_min", "mean_ratio_max", "range")),
    ("std_ratio", ("std_ratio_min", "std_ratio_max", "range")),
    ("extreme_ratio", ("extreme_ratio_min", "extreme_ratio_max", "range")),
)


def _nan_fraction(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return 1.0
    return float(np.mean(~np.isfinite(v)))


def _finite(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float).ravel()
    return v[np.isfinite(v)]


def _metric(value: object, threshold: float | None, comparison: str, status: str,
            detail: str) -> dict[str, Any]:
    nvalue = float(value) if value is not None else float("nan")
    return {"value": nvalue, "threshold": threshold, "comparison": comparison,
            "status": status, "detail": detail}


def _short_detail(parts: list[str]) -> str:
    if len(parts) > _DETAIL_CAP:
        return "; ".join(parts[:_DETAIL_CAP]) + f"; (+{len(parts) - _DETAIL_CAP} more)"
    return "; ".join(parts)


def _columns(ds) -> dict[str, np.ndarray]:
    """把数据集展平成 1D 列（含 level 维的变量逐层拆列）。"""
    cols: dict[str, np.ndarray] = {}
    for name, da in ds.data_vars.items():
        arr = np.asarray(da.values, dtype=float)
        if "level" in da.dims:
            moved = np.moveaxis(arr, da.get_axis_num("level"), 0)
            try:
                levels = np.asarray(ds["level"].values).ravel()
            except KeyError:
                levels = np.arange(moved.shape[0])
            if levels.size != moved.shape[0]:
                levels = np.arange(moved.shape[0])
            for i, lv in enumerate(levels):
                cols[f"{name}@{lv:g}"] = moved[i].ravel()
        else:
            cols[str(name)] = arr.ravel()
    return cols


def _range_entry(per_col: dict[str, float | None], th: Mapping[str, float],
                 lo_key: str, hi_key: str, label: str,
                 demote: bool = False) -> dict[str, Any]:
    """两边阈值（range）指标的判定。

    value 取“绑定侧”的实测值：有列超上限 → value=最大值；有列破下限 →
    value=最小值；全部在界内 → value=最大值（代表测度），并列出 min/max 与
    全部逐列值。
    """
    lo = th.get(lo_key)
    hi = th.get(hi_key)
    vals = [(c, v) for c, v in per_col.items() if v is not None and np.isfinite(v)]
    skipped = [c for c, v in per_col.items() if v is None]
    if not vals:
        reason = _short_detail([f"{c}: n<2 或 mean≈0 不可计算" for c in skipped]) or "no data"
        return _metric(None, hi, "range", "NOT_RUN", f"{label}: {reason}")
    values = [v for _, v in vals]
    over = [v for v in values if hi is not None and v > hi]
    under = [v for v in values if lo is not None and v < lo]
    if over:
        value, status = max(over), "FAIL"
    elif under:
        value, status = min(under), "FAIL"
    elif lo is None or hi is None:
        value, status = float(np.median(values)), "CONDITIONAL_PASS"
    else:
        value, status = max(values), "PASS"
    has_gap = bool(skipped) or demote
    if has_gap and status == "PASS":
        status = "CONDITIONAL_PASS"  # 只覆盖了部分列 → 不是完整 PASS
    bounds = f"[{lo if lo is not None else '-'}, {hi if hi is not None else '-'}]"
    detail = (
        f"per-col={_short_detail([f'{c}={v:.4f}' for c, v in vals])}; "
        f"bounds={bounds}"
    )
    if skipped:
        detail += f"; skipped={_short_detail(skipped)}"
    return _metric(value, hi if hi is not None else lo, "range", status, detail)


def _max_entry(per_col: dict[str, float | None], th: Mapping[str, float], key: str,
               label: str, demote: bool = False) -> dict[str, Any]:
    """单边上限（max）指标：value=最差（最大）实测值，需 ≤ 阈值。"""
    limit = th.get(key)
    vals = [(c, v) for c, v in per_col.items() if v is not None and np.isfinite(v)]
    skipped = [c for c, v in per_col.items() if v is None]
    if not vals:
        reason = _short_detail(skipped) or "no data"
        return _metric(None, limit, "max", "NOT_RUN", f"{label}: {reason}")
    worst = max(v for _, v in vals)
    status = "FAIL" if limit is not None and worst > limit else "PASS"
    if (bool(skipped) or demote) and status == "PASS":
        status = "CONDITIONAL_PASS"
    detail = (f"per-col={_short_detail([f'{c}={v:.4f}' for c, v in vals])}"
              + (f"; skipped={_short_detail(skipped)}" if skipped else ""))
    return _metric(worst, limit, "max", status, detail)


def _quality_report_cls():
    """Import sdmetrics 的 single-table QualityReport。

    实测（sdmetrics 0.33.0）：统一入口 ``sdmetrics.reports.QualityReport`` 是
    multi-table 口径，metadata 需要 ``{"tables": {...}}``；展平后的单表数据
    应使用 ``sdmetrics.reports.single_table.QualityReport`` + metadata
    ``{"columns": {col: {"sdtype": "numerical"}}}``。
    不可用/版本漂移 → ImportError（上游记 NOT_RUN）。
    """
    try:
        from sdmetrics.reports.single_table import QualityReport  # noqa: F501
        return QualityReport
    except Exception as exc:
        raise ImportError(f"sdmetrics single_table QualityReport unavailable: {exc}") from exc


def _quality_property_score(cls, df_real, df_syn, meta, prop: str) -> tuple[float, int, int]:
    """跑一次 QualityReport，返回 (score, n_rows, n_rows_with_score)。"""
    report = cls()
    report.generate(df_real, df_syn, metadata=meta, verbose=False)
    details = report.get_details(prop)
    scores = details["Score"].astype(float)
    valid = scores.dropna()
    return (float(valid.mean()) if len(valid) else float("nan"),
            int(len(scores)), int(len(valid)))


def _not_run_entry(th: Mapping[str, float], key: str, comparison: str, reason: str) -> dict[str, Any]:
    return _metric(None, th.get(key), comparison, "NOT_RUN", reason)


def run_statistical_validation(real, synthetic, *,
                               thresholds: Mapping[str, float]) -> dict[str, Any]:
    """对 (real, synthetic) 数据集做统计分布一致性验证，返回统一结构。"""
    th = dict(DEFAULT_THRESHOLDS)
    for k, v in (thresholds or {}).items():
        th[k] = float(v)

    real_cols = _columns(real)
    syn_cols = _columns(synthetic)
    notes: list[str] = []
    only_real = sorted(set(real_cols) - set(syn_cols))
    only_syn = sorted(set(syn_cols) - set(real_cols))
    if only_real:
        notes.append(f"仅在 real 中的列（不参与计算）: {only_real}")
    if only_syn:
        notes.append(f"仅在 synthetic 中的列（不参与计算）: {only_syn}")
    common = sorted(set(real_cols) & set(syn_cols))

    metrics: dict[str, dict[str, Any]] = {}
    if not common:
        reason = "real 与 synthetic 没有共同变量列"
        for key, comp, th_key in (
            ("column_shapes", "min", "column_shapes_min"),
            ("column_pair_trends", "min", "column_pair_trends_min"),
            ("mean_ratio", "range", "mean_ratio_max"),
            ("std_ratio", "range", "std_ratio_max"),
            ("quantile_rmse", "max", "quantile_rmse_max"),
            ("ks", "max", "ks_max"),
            ("corr_frobenius", "max", "corr_frobenius_max"),
            ("extreme_ratio", "range", "extreme_ratio_max"),
        ):
            metrics[key] = _not_run_entry(th, th_key, comp, reason)
        notes.append(reason)
        return {"metrics": metrics,
                "status": aggregate_status(m["status"] for m in metrics.values()),
                "notes": notes}

    # NaN 处理：各列按两侧的缺失比例筛（>50% 剔除）
    excluded: dict[str, str] = {}
    for c in common:
        reasons = []
        mr_ = _nan_fraction(real_cols[c])
        mg_ = _nan_fraction(syn_cols[c])
        if mr_ > _NAN_LIMIT:
            reasons.append(f"real 缺失 {mr_:.0%}")
        if mg_ > _NAN_LIMIT:
            reasons.append(f"synthetic 缺失 {mg_:.0%}")
        if reasons:
            excluded[c] = ";".join(reasons)
    usable = [c for c in common if c not in excluded]
    if excluded:
        notes.append("因缺测 >50% 剔除的列: "
                     + "; ".join(f"{c}({r})" for c, r in sorted(excluded.items())))
    if not usable:
        reason = "所有共同列均因缺测 >50% 被剔除"
        for key, comp, th_key in (
            ("column_shapes", "min", "column_shapes_min"),
            ("column_pair_trends", "min", "column_pair_trends_min"),
            ("mean_ratio", "range", "mean_ratio_max"),
            ("std_ratio", "range", "std_ratio_max"),
            ("quantile_rmse", "max", "quantile_rmse_max"),
            ("ks", "max", "ks_max"),
            ("corr_frobenius", "max", "corr_frobenius_max"),
            ("extreme_ratio", "range", "extreme_ratio_max"),
        ):
            metrics[key] = _not_run_entry(th, th_key, comp, reason)
        notes.append(reason)
        return {"metrics": metrics,
                "status": aggregate_status(m["status"] for m in metrics.values()),
                "notes": notes}

    any_excluded = bool(excluded)

    # ---- 逐列简单统计 -------------------------------------------------------
    mean_per: dict[str, float | None] = {}
    std_per: dict[str, float | None] = {}
    quant_per: dict[str, float | None] = {}
    ks_per: dict[str, float | None] = {}
    extr_per: dict[str, float | None] = {}
    skips: dict[str, list[str]] = {}
    for c in usable:
        r = _finite(real_cols[c])
        g = _finite(syn_cols[c])
        if r.size < 3 or g.size < 3:
            skips.setdefault("n<3", []).append(c)
            continue
        mean_rr = float(r.mean())
        mean_gg = float(g.mean())
        std_r = float(r.std())
        if abs(mean_rr) > 1e-12 * max(1.0, std_r):
            mean_per[c] = mean_gg / mean_rr if mean_rr else None
        else:
            skips.setdefault("mean≈0", []).append(c)
        std_g = float(g.std())
        if std_r > _EPS:
            std_per[c] = std_g / std_r
        else:
            skips.setdefault("std≈0", []).append(c)
        if r.size >= 10 and g.size >= 10:
            q_r = np.quantile(r, _QUANTILES)
            q_g = np.quantile(g, _QUANTILES)
            span = float(q_r[-1] - q_r[0])
            if span > _EPS:
                quant_per[c] = float(np.sqrt(np.mean(((q_g - q_r) / span) ** 2)))
            else:
                skips.setdefault("q95-q05 span≈0", []).append(c)
            # 极端值：P1/P99 跨度比（span_gen/span_real），附单分位比值
            p_r = np.quantile(r, (0.01, 0.99))
            p_g = np.quantile(g, (0.01, 0.99))
            span_r = float(p_r[-1] - p_r[0])
            span_g = float(p_g[-1] - p_g[0])
            if span_r > _EPS:
                extr_per[c] = span_g / span_r
            else:
                skips.setdefault("P99-P01 span≈0", []).append(c)
        else:
            skips.setdefault("n<10", []).append(c)
        if r.size >= 1 and g.size >= 1:
            ks_per[c] = float(stats.ks_2samp(r, g).statistic)
        else:
            skips.setdefault("ks n=0", []).append(c)

    if skips:
        for reason, cols in skips.items():
            notes.append(f"{reason} 跳过: {cols}")

    metrics["mean_ratio"] = _range_entry(mean_per, th, "mean_ratio_min", "mean_ratio_max", "mean_ratio", demote=any_excluded)
    metrics["std_ratio"] = _range_entry(std_per, th, "std_ratio_min", "std_ratio_max", "std_ratio", demote=any_excluded)
    metrics["extreme_ratio"] = _range_entry(extr_per, th, "extreme_ratio_min", "extreme_ratio_max", "extreme_ratio", demote=any_excluded)
    metrics["quantile_rmse"] = _max_entry(quant_per, th, "quantile_rmse_max", "quantile_rmse", demote=any_excluded)
    ks_entry = _max_entry(ks_per, th, "ks_max", "ks", demote=any_excluded)
    metrics["ks"] = ks_entry

    # ---- 相关矩阵 Frobenius 距离 --------------------------------------------
    metrics["corr_frobenius"] = _corr_frobenius_entry(usable, real_cols, syn_cols, th, any_excluded)

    # ---- sdmetrics QualityReport --------------------------------------------
    shapes, pairs = _sdmetrics_entries(usable, real_cols, syn_cols, th,
                                       any_excluded, skips)
    metrics["column_shapes"] = shapes
    metrics["column_pair_trends"] = pairs

    status = aggregate_status(m["status"] for m in metrics.values())
    return {"metrics": metrics, "status": status, "notes": notes}


def _corr_frobenius_entry(usable, real_cols, syn_cols, th, any_excluded) -> dict[str, Any]:
    limit = th.get("corr_frobenius_max")

    def corr_matrix(cols: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
        arrays = {n: np.asarray(cols[n], dtype=float) for n in names}
        n = len(names)
        mat = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = arrays[names[i]], arrays[names[j]]
                mask = np.isfinite(a) & np.isfinite(b)
                if int(mask.sum()) < _MIN_CORR_ROWS:
                    mat[i, j] = mat[j, i] = np.nan
                    continue
                sa, sb = float(a[mask].std()), float(b[mask].std())
                if sa <= _EPS or sb <= _EPS:
                    mat[i, j] = mat[j, i] = np.nan
                    continue
                mat[i, j] = mat[j, i] = float(np.corrcoef(a[mask], b[mask])[0, 1])
        return mat

    names = [c for c in usable
             if _finite(real_cols[c]).size >= _MIN_CORR_ROWS
             and _finite(syn_cols[c]).size >= _MIN_CORR_ROWS]
    if len(names) < 2:
        reason = (f"有效列不足 2（usable={len(usable)}, corr-ready={len(names)}）"
                  f"，无法计算相关矩阵")
        return _metric(None, limit, "max", "NOT_RUN", reason)
    r_mat = corr_matrix(real_cols, names)
    g_mat = corr_matrix(syn_cols, names)
    diff = r_mat - g_mat
    mask = np.isfinite(diff)
    if not mask.any():
        return _metric(None, limit, "max", "NOT_RUN", "相关矩阵数据对不足，差值全为 NaN")
    value = float(np.linalg.norm(diff[mask]) / np.sqrt(len(names)))
    n_missing = int((~mask).sum())
    status = "FAIL" if limit is not None and value > limit else "PASS"
    if n_missing and status == "PASS":
        status = "CONDITIONAL_PASS"
    detail = f"n_vars={len(names)}; frobenius/sqrt(n)={value:.4f}"
    if n_missing:
        detail += f"; 失效相关对 {n_missing} 个（样本不足或零方差）"
    return _metric(value, limit, "max", status, detail)


def _sdmetrics_entries(usable, real_cols, syn_cols, th, any_excluded, skips):
    """column_shapes / column_pair_trends（sdmetrics QualityReport）。

    无缺测 → 整表一次 QualityReport（高效路径）；有缺测 → 按列/按列对
    分别压掉 NaN 再跑（逐列路径）。
    """
    shapes_limit = th.get("column_shapes_min")
    pair_limit = th.get("column_pair_trends_min")

    def not_run(reason: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return (_metric(None, shapes_limit, "min", "NOT_RUN", reason),
                _metric(None, pair_limit, "min", "NOT_RUN", reason))

    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - pandas 属数据栈基础件
        return not_run(f"pandas 导入失败: {type(exc).__name__}: {exc}")

    try:
        cls = _quality_report_cls()
    except Exception as exc:
        return not_run(f"sdmetrics QualityReport 不可用: {exc}")

    meta = {"columns": {c: {"sdtype": "numerical"} for c in sorted(usable)}}

    def col_real(c: str):
        return _finite(real_cols[c])

    def col_syn(c: str):
        return _finite(syn_cols[c])

    clean = (not any_excluded
             and not any(v for vs in skips.values() for v in vs))
    try:
        if clean:
            df_real = pd.DataFrame({c: real_cols[c] for c in sorted(usable)})
            df_syn = pd.DataFrame({c: syn_cols[c] for c in sorted(usable)})
            s_score, n_s, v_s = _quality_property_score(cls, df_real, df_syn, meta, "Column Shapes")
            p_score, n_p, v_p = _quality_property_score(cls, df_real, df_syn, meta, "Column Pair Trends")
            extra = ""
        else:
            # 逐列路径：每列（或列对）各自压掉 NaN 后单独跑
            s_parts: list[float] = []
            for c in sorted(usable):
                subset_meta = {"columns": {c: {"sdtype": "numerical"}}}
                df_r = pd.DataFrame({c: col_real(c)})
                df_s = pd.DataFrame({c: col_syn(c)})
                score, _, _ = _quality_property_score(cls, df_r, df_s, subset_meta,
                                                      "Column Shapes")
                if np.isfinite(score):
                    s_parts.append(score)
            s_score = float(np.mean(s_parts)) if s_parts else float("nan")
            n_s, v_s = len(usable), len(s_parts)

            from itertools import combinations
            pair_names = sorted(usable)
            pair_scores: list[float] = []
            total_pairs = len(list(combinations(pair_names, 2)))
            for a, b in combinations(pair_names, 2):
                ar, ar2 = real_cols[a], real_cols[b]
                mr = np.isfinite(ar) & np.isfinite(ar2)
                bg, bg2 = syn_cols[a], syn_cols[b]
                mg = np.isfinite(bg) & np.isfinite(bg2)
                if mr.sum() < _MIN_CORR_ROWS or mg.sum() < _MIN_CORR_ROWS:
                    continue
                df_r = pd.DataFrame({a: ar[mr], b: ar2[mr]})
                df_s = pd.DataFrame({a: bg[mg], b: bg2[mg]})
                sub_meta = {"columns": {a: {"sdtype": "numerical"},
                                        b: {"sdtype": "numerical"}}}
                score, _, _ = _quality_property_score(cls, df_r, df_s, sub_meta,
                                                      "Column Pair Trends")
                if np.isfinite(score):
                    pair_scores.append(score)
            p_score = float(np.mean(pair_scores)) if pair_scores else float("nan")
            n_p, v_p = total_pairs, len(pair_scores)
            extra = f"; 逐列/逐对压缩 NaN 路径（无缺测列 {n_s}）"
    except Exception as exc:
        return not_run(f"sdmetrics QualityReport 崩溃: {type(exc).__name__}: {exc}")

    if not np.isfinite(s_score):
        shapes = _metric(None, shapes_limit, "min", "NOT_RUN",
                         f"Column Shapes 无有效分数{extra}")
    else:
        status = "FAIL" if shapes_limit is not None and s_score < shapes_limit else "PASS"
        if status == "PASS" and (any_excluded or v_s < n_s):
            status = "CONDITIONAL_PASS"
        shapes = _metric(s_score, shapes_limit, "min", status,
                         f"sdmetrics Column Shapes 均分={s_score:.4f} ({v_s}/{n_s} 列)")

    if not np.isfinite(p_score):
        pairs = _metric(None, pair_limit, "min", "NOT_RUN",
                        f"Column Pair Trends 无有效分数（列数不足或样本不足）{extra}")
    else:
        status = "FAIL" if pair_limit is not None and p_score < pair_limit else "PASS"
        if status == "PASS" and any_excluded:
            status = "CONDITIONAL_PASS"
        pairs = _metric(p_score, pair_limit, "min", status,
                        f"sdmetrics Column Pair Trends 均分={p_score:.4f} "
                        f"(计分对 {v_p}，其中 NaN 对 {n_p - v_p} 为 sdmetrics "
                        f"0.33 设计剔除：|real r|≤0.5 不参与均分)" + extra)
    return shapes, pairs
