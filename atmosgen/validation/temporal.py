"""时间连续性验证（实现计划 §5.2, ``validation/temporal.py``）。

口径说明
--------
- 序列：沿 ``time`` 轴取每个 ``(variable, level, cell)``（tcwv 为 ``(variable, cell)``）
  的 1D 序列；real 与 synthetic 的枚举空间一致（按共同变量/层数/网格取交集），
  抽样 ≤ 200 个组合，固定种子 42；time 长度不一致时按较短截断并记录；
- NaN：保留不插值；real/synthetic 各自按序列压缩 NaN 后计算，任一侧缺测
  >50% 的序列剔除，全部被剔除 → 指标 NOT_RUN；
- 聚合：acf/slope 用均值、jump 用最差值，ratio 类 value 用中位数、状态按
  最差序列判（detail 里同时给中位数/最差值）。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .report import aggregate_status

__all__ = ["DEFAULT_THRESHOLDS", "run_temporal_validation", "timeseries_efficacy"]

#: 计划 §5.2 默认阈值
DEFAULT_THRESHOLDS: dict[str, float] = {
    "acf_diff_max": 0.20,
    "diff_var_ratio_min": 0.5,
    "diff_var_ratio_max": 2.0,
    "spectral_slope_diff_max": 0.50,
    "jump_extra_max": 0.01,
    "window_var_ratio_max": 2.0,
}

_MAX_SERIES = 200
_SERIES_SEED = 42
_LAGS = (1, 2, 3, 4, 5, 6)
_WINDOW = 8
_NAN_LIMIT = 0.5
_EPS = 1e-12
_SPECTRAL_FREQ_FRAC = 4  # 频率取 1..N/4 段
_JUMP_SIGMA = 5.0
_EFFICACY_MAX_ACC = 0.65  # 判别准确率 ≤ 0.65 ≈ 不可区分

KEY_ACF = "acf_diff"
KEY_DIFF_VAR = "diff_var_ratio"
KEY_SLOPE = "spectral_slope_diff"
KEY_JUMP = "jump_extra"
KEY_WINDOW = "window_var_ratio"
KEY_EFFICACY = "timeseries_efficacy"


def _metric(value: object, threshold: float | None, comparison: str, status: str,
            detail: str) -> dict[str, Any]:
    nvalue = float(value) if value is not None else float("nan")
    return {"value": nvalue, "threshold": threshold, "comparison": comparison,
            "status": status, "detail": detail}


def _nan_fraction(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float).ravel()
    return float(np.mean(~np.isfinite(v))) if v.size else 1.0


def _compress(values: np.ndarray) -> np.ndarray:
    """保留 NaN 不插值；按序列压缩掉非有限值。"""
    v = np.asarray(values, dtype=float).ravel()
    return v[np.isfinite(v)]


def _short(items: list[str], cap: int = 5) -> str:
    if len(items) > cap:
        return "; ".join(items[:cap]) + f"; (+{len(items) - cap} more)"
    return "; ".join(items)


Combo = tuple[str, int | None, int, int]  # (var, level_index|None, i, j)


def _catalog(real, synthetic) -> tuple[list[Combo], list[str]]:
    """枚举两侧一致的 (var, level_idx, i, j) 组合键。"""
    notes: list[str] = []
    if real.sizes["time"] != synthetic.sizes["time"]:
        notes.append(f"time 长度不一致 ({real.sizes['time']} vs "
                     f"{synthetic.sizes['time']})，按较短截断")
    combos: list[Combo] = []
    for name in sorted(real.data_vars):
        if name not in synthetic.data_vars:
            continue
        rda, sda = real[name], synthetic[name]
        if "time" not in rda.dims or "time" not in sda.dims:
            continue
        if set(rda.dims) != set(sda.dims):
            notes.append(f"{name}: 维度集合不一致 {sorted(rda.dims)} vs "
                         f"{sorted(sda.dims)}，跳过")
            continue
        has_level = "level" in rda.dims
        has_grid = "latitude" in rda.dims and "longitude" in rda.dims
        n_level = int(min(rda.shape[rda.get_axis_num("level")],
                          sda.shape[sda.get_axis_num("level")])) if has_level else 1
        n_lat = int(min(rda.shape[rda.get_axis_num("latitude")],
                        sda.shape[sda.get_axis_num("latitude")])) if has_grid else 1
        n_lon = int(min(rda.shape[rda.get_axis_num("longitude")],
                        sda.shape[sda.get_axis_num("longitude")])) if has_grid else 1
        for li in range(n_level):
            if has_grid:
                for i in range(n_lat):
                    for j in range(n_lon):
                        combos.append((name, li if has_level else None, i, j))
            else:
                combos.append((name, li if has_level else None, 0, 0))
    return combos, notes


def _series(ds, combo: Combo) -> np.ndarray | None:
    name, level_idx, i, j = combo
    if name not in ds.data_vars:
        return None
    da = ds[name]
    sel: dict[str, int] = {}
    if level_idx is not None and "level" in da.dims:
        sel.setdefault("level", min(level_idx, da.shape[da.get_axis_num("level")] - 1))
    if "latitude" in da.dims:
        sel.setdefault("latitude", min(i, da.shape[da.get_axis_num("latitude")] - 1))
    if "longitude" in da.dims:
        sel.setdefault("longitude", min(j, da.shape[da.get_axis_num("longitude")] - 1))
    vals = np.asarray(da.isel(**sel).values, dtype=float).ravel()
    return vals if vals.size else None


def _not_run_all(th: Mapping[str, float], reason: str) -> dict[str, Any]:
    metrics = {
        KEY_ACF: _metric(None, th.get("acf_diff_max"), "max", "NOT_RUN", reason),
        KEY_DIFF_VAR: _metric(None, th.get("diff_var_ratio_max"), "range", "NOT_RUN", reason),
        KEY_SLOPE: _metric(None, th.get("spectral_slope_diff_max"), "max", "NOT_RUN", reason),
        KEY_JUMP: _metric(None, th.get("jump_extra_max"), "max", "NOT_RUN", reason),
        KEY_WINDOW: _metric(None, th.get("window_var_ratio_max"), "max", "NOT_RUN", reason),
        KEY_EFFICACY: _metric(None, None, "info", "NOT_RUN", reason),
    }
    return {"metrics": metrics,
            "status": aggregate_status(m["status"] for m in metrics.values()),
            "notes": [reason]}


def run_temporal_validation(real, synthetic, *, thresholds: Mapping[str, float]) -> dict[str, Any]:
    """对 (real, synthetic) 做时间连续性验证，返回统一结构。"""
    th = dict(DEFAULT_THRESHOLDS)
    for k, v in (thresholds or {}).items():
        th[k] = float(v)
    notes: list[str] = []

    if "time" not in real.dims or "time" not in synthetic.dims:
        return _not_run_all(th, "输入缺少 time 维，所有时间连续性指标无法计算")

    combos, catalog_notes = _catalog(real, synthetic)
    notes.extend(catalog_notes)
    if not combos:
        return _not_run_all(th, "没有共同的时间序列组合（common 变量/time/网格不一致）")

    rng = np.random.default_rng(_SERIES_SEED)
    chosen = list(combos)
    if len(combos) > _MAX_SERIES:
        idx = sorted(rng.choice(len(combos), size=_MAX_SERIES, replace=False).tolist())
        chosen = [combos[i] for i in idx]
        notes.append(f"序列抽样: {len(chosen)}/{len(combos)} (seed={_SERIES_SEED})")

    n_dropped = 0
    pairs: list[tuple[Combo, np.ndarray, np.ndarray]] = []
    for combo in chosen:
        a = _series(real, combo)
        b = _series(synthetic, combo)
        if a is None or b is None or a.size != b.size:
            n_dropped += 1
            continue
        if _nan_fraction(a) > _NAN_LIMIT or _nan_fraction(b) > _NAN_LIMIT:
            n_dropped += 1
            continue
        pairs.append((combo, _compress(a), _compress(b)))
    if n_dropped:
        notes.append(f"序列剔除 {n_dropped} 个（长度不一致或缺测 >50%）")
    if not pairs:
        return _not_run_all(th, "全部序列缺测 >50% 或长度不一致，时间指标无法计算")

    metrics: dict[str, dict[str, Any]] = {
        KEY_EFFICACY: _metric(
            None, None, "info", "NOT_RUN",
            "Phase 2 默认关闭，需要显式调用 timeseries_efficacy()（sdmetrics "
            "TimeSeriesEfficacyMetric 判别，准确率 ≤ 0.65 视为不可区分）"),
    }
    metrics[KEY_ACF] = _acf_entry(pairs, th)
    metrics[KEY_DIFF_VAR] = _ratio_entry(pairs, th, KEY_DIFF_VAR,
                                         _diff_var, need=_WINDOW + 1,
                                         lo_key="diff_var_ratio_min",
                                         hi_key="diff_var_ratio_max")
    metrics[KEY_SLOPE] = _spectral_slope_entry(pairs, th)
    metrics[KEY_JUMP] = _jump_entry(pairs, th)
    metrics[KEY_WINDOW] = _ratio_entry(pairs, th, KEY_WINDOW,
                                       _window_var, need=2 * _WINDOW,
                                       lo_key=None, hi_key="window_var_ratio_max")
    # 默认关闭的 timeseries_efficacy(NOT_RUN) 不计入本节总状态，
    # 否则每次 evaluate 都会被"未启用"拖成 NOT_RUN；需要时可显式调用
    # timeseries_efficacy() 并把其状态并入重新聚合。
    statuses = [m["status"] for k, m in metrics.items() if k != KEY_EFFICACY]
    notes.append("timeseries_efficacy 默认关闭（未计入本节总状态）；"
                 "显式调用 timeseries_efficacy() 后其结果可并入聚合")
    return {"metrics": metrics,
            "status": aggregate_status(statuses),
            "notes": notes}


def _key_txt(combo: Combo) -> str:
    name, li, i, j = combo
    tag = f"[{i},{j}]" if (i, j) != (0, 0) else ""
    return f"{name}{'@' + str(li) if li is not None else ''}{tag}"


def _acf_at(x: np.ndarray, lag: int) -> float | None:
    if x.size <= lag + 2:
        return None
    a, b = x[:-lag], x[lag:]
    if a.std() <= _EPS or b.std() <= _EPS:
        return None
    val = float(np.corrcoef(a, b)[0, 1])
    return val if np.isfinite(val) else None


def _acf_entry(pairs, th) -> dict[str, Any]:
    per: dict[str, float] = {}
    skips: list[str] = []
    real_acf_vals: list[float] = []
    n_pairs = 0
    for combo, xr_, xs in pairs:
        key = _key_txt(combo)
        ar_ = [_acf_at(xr_, lag) for lag in _LAGS]
        ag = [_acf_at(xs, lag) for lag in _LAGS]
        pr = [v for v in ar_ if v is not None]
        pg = [v for v in ag if v is not None]
        if not pr or not pg:
            skips.append(f"{key}: ACF 不可算（太短或零方差）")
            continue
        n = min(len(pr), len(pg))
        per[key] = float(np.mean([abs(a - b) for a, b in zip(pr[:n], pg[:n])]))
        real_acf_vals.extend(pr[:n])
        n_pairs += 1
    if not per:
        reason = "无有效 ACF 序列（太短/零方差）"
        if skips:
            reason += f"; {_short(skips)}"
        return _metric(None, th.get("acf_diff_max"), "max", "NOT_RUN", reason)
    value = float(np.mean(list(per.values())))
    limit = th.get("acf_diff_max")
    status = "FAIL" if limit is not None and value > limit else "PASS"
    if skips and status == "PASS":
        status = "CONDITIONAL_PASS"
    worst = max(per.values())
    real_ref = float(np.mean(real_acf_vals)) if real_acf_vals else float("nan")
    detail = (f"n_series={n_pairs}; mean|Δ ACF|={value:.4f}; 最差={worst:.4f}; "
              f"real ACF(1-6) 参考={real_ref:.4f}; "
              f"口径: lag 1-6 mean|ΔACF|，逐序列差再平均")
    if skips:
        detail += f"; skipped={_short(skips)}"
    return _metric(value, limit, "max", status, detail)


def _diff_var(xr_: np.ndarray, xs: np.ndarray) -> float | None:
    dr, dg = np.diff(xr_), np.diff(xs)
    vr = float(np.var(dr, ddof=1))
    if vr <= 0.0:  # 仅常数序列剔除；量级极小（如 q）也照算相对比值
        return None
    return float(np.var(dg, ddof=1)) / vr


def _window_var(xr_: np.ndarray, xs: np.ndarray) -> float | None:
    w = _WINDOW

    def chunk_var(x: np.ndarray) -> float:
        n = (x.size // w) * w
        blocks = x[:n].reshape(n // w, w)
        return float(np.var(blocks.mean(axis=1), ddof=1))

    vr = chunk_var(xr_)
    if vr <= 0.0:  # 常数（块均值零方差）序列剔除；量级极小照算
        return None
    return chunk_var(xs) / vr


def _ratio_entry(pairs, th, label, fn, *, need: int, lo_key: str | None,
                 hi_key: str) -> dict[str, Any]:
    per: dict[str, float] = {}
    skips: list[str] = []
    for combo, xr_, xs in pairs:
        key = _key_txt(combo)
        if xr_.size < need:
            skips.append(f"{key}: 样本<{need}")
            continue
        val = fn(xr_, xs)
        if val is None:
            skips.append(f"{key}: 分母方差≈0")
            continue
        per[key] = val
    lo = th.get(lo_key) if lo_key else None
    hi = th.get(hi_key)
    comp = "max" if lo is None else "range"
    if not per:
        reason = f"{label}: 无可算序列；{_short(skips) if skips else 'no valid pairs'}"
        return _metric(None, hi if hi is not None else lo, comp, "NOT_RUN", reason)
    values = list(per.values())
    median = float(np.median(values))
    vmax, vmin = float(max(values)), float(min(values))
    if hi is not None and vmax > hi:
        value, status = vmax, "FAIL"
    elif lo is not None and vmin < lo:
        value, status = vmin, "FAIL"
    else:
        value = median
        status = "PASS" if hi is not None else "CONDITIONAL_PASS"
    if skips and status == "PASS":
        status = "CONDITIONAL_PASS"
    detail = (f"n_series={len(per)}; median={median:.4f}; min={vmin:.4f}; max={vmax:.4f}; "
              f"bounds=[{lo if lo is not None else '-'}, {hi if hi is not None else '-'}]; "
              f"口径: {label} = gen/real 逐序列比值，状态按最差序列、value 取中位数"
              f"（仅上限阈值 → 下限不判定）")
    if skips:
        detail += f"; skipped={_short(skips)}"
    return _metric(value, hi if hi is not None else lo, "range", status, detail)


def _slope(x: np.ndarray) -> float | None:
    n = x.size
    k = n // _SPECTRAL_FREQ_FRAC
    if k < 3:
        return None
    power = np.abs(np.fft.rfft(x - x.mean())) ** 2
    pk = power[1:k + 1]
    if np.any(pk <= 0):
        return None
    freqs = np.arange(1, k + 1, dtype=float)
    return float(np.polyfit(np.log(freqs), np.log(pk), 1)[0])


def _spectral_slope_entry(pairs, th) -> dict[str, Any]:
    per: dict[str, float] = {}
    skips: list[str] = []
    for combo, xr_, xs in pairs:
        key = _key_txt(combo)
        sr, sg = _slope_value(xr_), _slope_value(xs)
        if sr is None or sg is None:
            skips.append(f"{key}: 谱斜率不可算")
            continue
        per[key] = abs(sg - sr)
    if not per:
        reason = "spectral_slope_diff: 无可算序列（太短/功率含零）"
        if skips:
            reason += f"; {_short(skips)}"
        return _metric(None, th.get("spectral_slope_diff_max"), "max", "NOT_RUN", reason)
    value = float(np.mean(list(per.values())))
    limit = th.get("spectral_slope_diff_max")
    status = "FAIL" if limit is not None and value > limit else "PASS"
    if skips and status == "PASS":
        status = "CONDITIONAL_PASS"
    detail = (f"n_series={len(per)}; mean|Δslope|={value:.4f}; "
              f"最差={max(per.values()):.4f}; 口径: 沿 time rfft→log-log 拟合，频率 1..N/4")
    if skips:
        detail += f"; skipped={_short(skips)}"
    return _metric(value, limit, "max", status, detail)


def _slope_value(x: np.ndarray) -> float | None:
    return _slope(x)


def _jump(x: np.ndarray) -> float | None:
    if x.size < 4:
        return None
    s = float(x.std())
    if s <= _EPS:
        return None
    return float(np.mean(np.abs(x - x.mean()) > _JUMP_SIGMA * s))


def _jump_entry(pairs, th) -> dict[str, Any]:
    per: dict[str, float] = {}
    skips: list[str] = []
    for combo, xr_, xs in pairs:
        key = _key_txt(combo)
        jr, jg = _jump(xr_), _jump(xs)
        if jr is None or jg is None:
            skips.append(f"{key}: 突变率不可算")
            continue
        per[key] = jg - jr
    if not per:
        reason = "jump_extra: 无可算序列（太短/零方差）"
        if skips:
            reason += f"; {_short(skips)}"
        return _metric(None, th.get("jump_extra_max"), "max", "NOT_RUN", reason)
    values = list(per.values())
    limit = th.get("jump_extra_max")
    value = max(values)
    status = "FAIL" if limit is not None and value > limit else "PASS"
    if skips and status == "PASS":
        status = "CONDITIONAL_PASS"
    detail = (f"n_series={len(per)}; max extra={value:.4f}; "
              f"median extra={float(np.median(values)):+.4f}; "
              f"口径: 以 real 每序列 |x-mean|>5σ 比例为基线，gen 超阈比例-基线，取最差")
    if skips:
        detail += f"; skipped={_short(skips)}"
    return _metric(value, limit, "max", status, detail)


# ---------------------------------------------------------------------------
# 可选：sdmetrics 判别指标（默认由 run_temporal_validation 标 NOT_RUN）
# ---------------------------------------------------------------------------

def _efficacy_frames(real, synthetic, max_sequences: int):
    """按抽样好的 (level, cell) 序列，构造 {seq_index, timestep, feature...} 长表。"""
    import pandas as pd

    combos, _ = _catalog(real, synthetic)
    if not combos:
        raise ValueError("没有共同的时间序列组合")
    rng = np.random.default_rng(_SERIES_SEED)
    if len(combos) > max_sequences:
        idx = sorted(rng.choice(len(combos), size=max_sequences, replace=False).tolist())
        combos = [combos[i] for i in idx]

    n_time = int(min(real.sizes["time"], synthetic.sizes["time"]))
    frames_real: list[pd.DataFrame] = []
    frames_syn: list[pd.DataFrame] = []
    features: set[str] = set()
    for si, combo in enumerate(combos):
        feats_r: dict[str, np.ndarray] = {}
        feats_s: dict[str, np.ndarray] = {}
        ok = True
        for var in sorted(real.data_vars):
            if var not in synthetic.data_vars or "time" not in real[var].dims:
                ok = False
                break
            rv = _series(real, combo)
            sv = _series(synthetic, combo)
            if rv is None or sv is None or rv.size < n_time:
                ok = False
                break
            rv, sv = rv[:n_time], sv[:n_time]
            if _nan_fraction(rv) > _NAN_LIMIT or _nan_fraction(sv) > _NAN_LIMIT:
                ok = False
                break
            label = f"{var}" if combo[1] is None else \
                f"{var}@{float(real['level'].values[combo[1]]):g}"
            feats_r[label] = rv
            feats_s[label] = sv
            features.add(label)
        if not ok or not feats_r:
            continue
        steps = np.arange(n_time, dtype=int)
        frames_real.append(pd.DataFrame({"seq_index": si, "timestep": steps, **feats_r}))
        frames_syn.append(pd.DataFrame({"seq_index": si, "timestep": steps, **feats_s}))
    if not frames_real:
        raise ValueError("长表为空（序列全部缺测或长度不一致）")
    return frames_real, frames_syn, sorted(features)


def timeseries_efficacy(real, synthetic, *, max_sequences: int = 50) -> dict[str, Any]:
    """（可选）sdmetrics 时序判别：构造长表后调用
    ``TimeSeriesEfficacyMetric``（失败降级 ``TimeSeriesDetectionMetric``）。

    判别准确率 ≤ 0.65（≈不可区分）为 PASS。任何失败 → NOT_RUN + 原因。
    """
    try:
        frames_real, frames_syn, features = _efficacy_frames(real, synthetic, max_sequences)
        meta = {
            "seq_index": "seq_index",
            "timestep": "timestep",
            "columns": {f: {"sdtype": "numerical"} for f in features},
        }
        try:
            from sdmetrics.timeseries.efficacy import TimeSeriesEfficacyMetric

            score = float(TimeSeriesEfficacyMetric.compute(
                real_data=frames_real, synthetic_data=frames_syn, metadata=meta))
            used = "TimeSeriesEfficacyMetric"
        except Exception as first_exc:
            from sdmetrics.timeseries.detection import TimeSeriesDetectionMetric

            score = float(TimeSeriesDetectionMetric.compute(
                real_data=frames_real, synthetic_data=frames_syn, metadata=meta))
            used = f"TimeSeriesDetectionMetric (efficacy 失败: {first_exc})"
    except Exception as exc:
        reason = f"timeseries_efficacy 无法计算: {type(exc).__name__}: {exc}"
        return {"metrics": {KEY_EFFICACY: _metric(None, _EFFICACY_MAX_ACC, "max",
                                                  "NOT_RUN", reason)},
                "status": "NOT_RUN", "notes": [reason]}

    status = "PASS" if (np.isfinite(score) and score <= _EFFICACY_MAX_ACC) else "FAIL"
    detail = (f"判别准确率={score:.4f} ≤ {_EFFICACY_MAX_ACC}（≈不可区分）; "
              f"指标={used}; n_seq={len(frames_real)}; features={features}")
    return {"metrics": {KEY_EFFICACY: _metric(score, _EFFICACY_MAX_ACC, "max", status, detail)},
            "status": status, "notes": []}
