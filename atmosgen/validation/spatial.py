"""空间连续性验证（实现计划 §5.3, ``validation/spatial.py``）。

口径说明
--------
- 2D 场：4D 变量取每个 ``(time, level)`` 的 (latitude, longitude) 场；
  tcwv 取每个 ``time`` 的场；real 与 synthetic 枚举空间一致，抽样 ≤ 50 个
  ``(variable, time, level)`` 场，固定种子 123；
- NaN：保留不插值；任一侧场缺测 >50% 则该场剔除；半变异/梯度/邻域残差用
  nan-aware 计算；谱与相关长度需 FFT，先把 NaN 处替换为场均值（detail 说明）；
  全部场被剔除 → 指标 NOT_RUN；
- 聚合：每场一个值 → ratio 类状态按最差场判、value 取中位数；diff/KS 类
  value 取均值（detail 给最差值）。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scipy import stats
import numpy as np

from .report import aggregate_status

__all__ = ["DEFAULT_THRESHOLDS", "run_spatial_validation"]

#: 计划 §5.3 默认阈值
DEFAULT_THRESHOLDS: dict[str, float] = {
    "variogram_ratio_min": 0.5,
    "variogram_ratio_max": 2.0,
    "gradient_rms_ratio_min": 0.5,
    "gradient_rms_ratio_max": 2.0,
    "neighbor_ks_max": 0.20,
    "spatial_spectral_slope_diff_max": 0.30,
    "anisotropy_diff_max": 0.30,
    "corr_length_ratio_min": 0.5,
    "corr_length_ratio_max": 2.0,
}

_MAX_FIELDS = 50
_FIELDS_SEED = 123
_NAN_LIMIT = 0.5
_EPS = 1e-12
_ONE_OVER_E = float(np.exp(-1.0))
_SPECTRAL_FREQ_FRAC = 4  # 径向谱拟合的波数上限 = min(nlat,nlon)/4

KEY_VARIO = "variogram_ratio"
KEY_GRAD = "gradient_rms_ratio"
KEY_NBR_KS = "neighbor_ks"
KEY_SLOPE = "spatial_spectral_slope"
KEY_ANISO = "anisotropy_diff"
KEY_CORR_LEN = "corr_length"

_RATIO_KEYS: dict[str, tuple[str, str]] = {
    KEY_VARIO: ("variogram_ratio_min", "variogram_ratio_max"),
    KEY_GRAD: ("gradient_rms_ratio_min", "gradient_rms_ratio_max"),
    KEY_CORR_LEN: ("corr_length_ratio_min", "corr_length_ratio_max"),
}
_TH_KEYS: dict[str, str] = {
    KEY_VARIO: "variogram_ratio_max",
    KEY_GRAD: "gradient_rms_ratio_max",
    KEY_NBR_KS: "neighbor_ks_max",
    KEY_SLOPE: "spatial_spectral_slope_diff_max",
    KEY_ANISO: "anisotropy_diff_max",
    KEY_CORR_LEN: "corr_length_ratio_max",
}
CALIBER = {
    KEY_VARIO: "γ(h)=(相邻格差² 的 mean，lat/lon 双向)，h=1..5 各算 γg(h)/γr(h)，场内取中位数，跨场取中位数",
    KEY_GRAD: "‖∇(lat,lon)x‖ RMS 的 gen/real 比（np.gradient）",
    KEY_NBR_KS: "x-mean(4邻域) 残差分布的 KS 统计量（real vs gen），跨场均值",
    KEY_SLOPE: "rfft2→径向平均谱→log-log 斜率差 |Δ|（径向波数 1..min(nlat,nlon)/4），跨场均值",
    KEY_ANISO: "|log10(E_x/E_y)| 之差，E 为 x/y 方向梯度能量，跨场均值",
    KEY_CORR_LEN: "自相关沿径向降至 1/e 的距离（格），gen/real 比（rmax 内未降到的截尾场剔除）",
}

FieldKey = tuple[str, int, int | None]  # (var, time_idx, level_idx|None)


def _metric(value: object, threshold: float | None, comparison: str, status: str,
            detail: str) -> dict[str, Any]:
    nvalue = float(value) if value is not None else float("nan")
    return {"value": nvalue, "threshold": threshold, "comparison": comparison,
            "status": status, "detail": detail}


def _short(items: list[str], cap: int = 5) -> str:
    if len(items) > cap:
        return "; ".join(items[:cap]) + f"; (+{len(items) - cap} more)"
    return "; ".join(items)


def _field_key_txt(combo: FieldKey) -> str:
    name, ti, li = combo
    return f"{name}{'@L' + str(li) if li is not None else ''} t{ti}"


def _catalog(real, synthetic) -> tuple[list[FieldKey], list[str]]:
    """枚举两侧一致的 2D 场 (var, time_idx, level_idx)。"""
    notes: list[str] = []
    combos: list[FieldKey] = []
    needed = {"latitude", "longitude", "time"}
    for name in sorted(real.data_vars):
        if name not in synthetic.data_vars:
            continue
        rda, sda = real[name], synthetic[name]
        skip_var = False
        for label, da in (("real", rda), ("synthetic", sda)):
            missing = sorted(needed - set(da.dims))
            if missing:
                notes.append(f"{name}: {label} 缺 {missing} 维，该变量不参与空间指标")
                skip_var = True
        if skip_var:
            continue
        if set(rda.dims) != set(sda.dims):
            notes.append(f"{name}: 维度集合不一致，跳过")
            continue
        has_level = "level" in rda.dims
        n_time = int(min(rda.shape[rda.get_axis_num("time")],
                         sda.shape[sda.get_axis_num("time")]))
        for ti in range(n_time):
            if has_level:
                for li in range(rda.shape[rda.get_axis_num("level")]):
                    combos.append((name, ti, li))
            else:
                combos.append((name, ti, None))
    return combos, notes


def _field(ds, combo: FieldKey) -> np.ndarray | None:
    name, ti, li = combo
    if name not in ds.data_vars:
        return None
    da = ds[name]
    sel: dict[str, int] = {"time": min(ti, da.shape[da.get_axis_num("time")] - 1)}
    if li is not None and "level" in da.dims:
        sel["level"] = min(li, da.shape[da.get_axis_num("level")] - 1)
    arr = np.asarray(da.isel(**sel).values, dtype=float)
    return arr if arr.ndim == 2 else None


def _fill_nan(F: np.ndarray) -> np.ndarray:
    """FFT 类指标用：NaN 处替换为场均值（不插值，仅为 FFT 提供有限输入）。"""
    if np.isfinite(F).all():
        return F
    mean = float(np.nanmean(F)) if np.isfinite(F).any() else 0.0
    return np.where(np.isfinite(F), F, mean)


def _gamma(F: np.ndarray, h: int) -> float | None:
    """经验半变异 γ(h)：lat/lon 两方向的 (F[i+h]-F[i])² 的均值（nan-aware）。"""
    if F.shape[0] <= h or F.shape[1] <= h:
        return None
    d1 = F[h:, :] - F[:-h, :]
    d2 = F[:, h:] - F[:, :-h]
    sq = np.concatenate([d1.ravel(), d2.ravel()]) ** 2
    vals = sq[np.isfinite(sq)]
    if vals.size == 0:
        return None
    return float(np.mean(vals))


def _grad_rms(F: np.ndarray) -> float | None:
    gy, gx = np.gradient(F)  # NaN 仅向邻域扩散一圈
    gg = np.concatenate([(gy ** 2).ravel(), (gx ** 2).ravel()])
    vals = gg[np.isfinite(gg)]
    if vals.size == 0:
        return None
    return float(np.sqrt(np.mean(vals)))


def _neighbor_residual(F: np.ndarray) -> np.ndarray:
    """x - mean(4 邻域)，nan-aware；NaN 只影响其自身邻域的残差。"""
    lat_n, lon_n = F.shape
    nsum = np.zeros_like(F)
    ncnt = np.zeros(F.shape, dtype=int)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ys = slice(max(0, -dy), min(lat_n, lat_n - dy))
        xs = slice(max(0, -dx), min(lon_n, lon_n - dx))
        nys = slice(max(0, dy), min(lat_n, lat_n + dy))
        nxs = slice(max(0, dx), min(lon_n, lon_n + dx))
        valid = np.zeros(F.shape, dtype=bool)
        valid[ys, xs] = True
        neighbor = np.full_like(F, np.nan)
        neighbor[ys, xs] = F[nys, nxs]
        finite = valid & np.isfinite(neighbor)
        nsum[finite] += neighbor[finite]
        ncnt[valid] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return F - nsum / ncnt


def _neighbor_mean_valid(F: np.ndarray) -> tuple[np.ndarray, np.ndarray | bool]:
    residual = _neighbor_residual(F)
    finite = np.isfinite(residual)
    return residual[finite], finite.sum()


def _radial_autocorr_profile(F: np.ndarray) -> dict[int, float]:
    """环形自相关（irfft2 |F̂|²）按折叠半径 lag=1..rmax 的径向平均。"""
    n0, n1 = F.shape
    spec = np.fft.rfft2(F - float(np.mean(F)))
    ac = np.fft.irfft2(np.abs(spec) ** 2, s=(n0, n1))
    v0 = ac[0, 0]
    if v0 <= _EPS:
        return {}
    rmax = min(n0, n1) // 2
    bins: dict[int, list[float]] = {}
    for i in range(n0):
        for j in range(n1):
            ii = i if i <= n0 - i else n0 - i
            jj = j if j <= n1 - j else n1 - j
            d = int(round(float(np.hypot(ii, jj))))
            if d < 1 or d > rmax:
                continue
            bins.setdefault(d, []).append(float(ac[i, j] / v0))
    return {d: float(np.mean(v)) for d, v in bins.items() if v}


def _corr_length(F: np.ndarray) -> float | None:
    profile = _radial_autocorr_profile(F)
    if not profile:
        return None
    for r in sorted(profile):
        if profile[r] <= _ONE_OVER_E:
            return float(r)
    return None  # 截尾：rmax 内未降到 1/e


def _radial_spectrum_slope(F: np.ndarray) -> float | None:
    """rfft2 → 径向平均功率谱 → log-log 斜率（径向波数 1..min(nlat,nlon)/4）。"""
    n0, n1 = F.shape
    F0 = _fill_nan(F)
    F0 = F0 - F0.mean()
    power = np.abs(np.fft.rfft2(F0)) ** 2  # (n0, n1//2+1)
    ky = np.fft.fftfreq(n0) * n0
    kx = np.fft.rfftfreq(n1) * n1
    kk = np.hypot(ky[:, None], kx[None, :])
    kcap = min(n0, n1) / _SPECTRAL_FREQ_FRAC
    if kcap < 3 or not np.isfinite(power).all():
        return None
    bins: dict[int, list[float]] = {}
    for r, p in zip(kk.ravel().tolist(), power.ravel().tolist()):
        b = int(round(r))
        if b < 1 or b > kcap:
            continue  # DC/截外环不参与拟合（DC 幂恒为 0）
        if p <= 0:
            return None
        bins.setdefault(b, []).append(p)
    if len(bins) < 3:
        return None
    radii = np.array([np.log(float(b)) for b in sorted(bins)])
    powers = np.array([np.log(float(np.mean(bins[b]))) for b in sorted(bins)])
    return float(np.polyfit(radii, powers, 1)[0])


def _anisotropy_log_ratio(F: np.ndarray) -> float | None:
    gy, gx = np.gradient(F)
    ex = np.nanmean(gx ** 2)
    ey = np.nanmean(gy ** 2)
    if not (np.isfinite(ex) and np.isfinite(ey)) or ex <= _EPS or ey <= _EPS:
        return None
    return float(abs(np.log10(ex / ey)))


def _not_run_entry(th, key: str, reason: str) -> dict[str, Any]:
    comp = "range" if key in _RATIO_KEYS else "max"
    limit = th.get(_TH_KEYS[key])
    return _metric(None, limit, comp, "NOT_RUN", reason)


def run_spatial_validation(real, synthetic, *, thresholds: Mapping[str, float]) -> dict[str, Any]:
    """对 (real, synthetic) 做空间连续性验证，返回统一结构。"""
    th = dict(DEFAULT_THRESHOLDS)
    for k, v in (thresholds or {}).items():
        th[k] = float(v)
    notes: list[str] = []

    combos, catalog_notes = _catalog(real, synthetic)
    notes.extend(catalog_notes)
    if not combos:
        reason = "没有共同的空间场组合（需要 time/latitude/longitude 维）"
        metrics = {key: _not_run_entry(th, key, reason)
                   for key in (KEY_VARIO, KEY_GRAD, KEY_NBR_KS, KEY_SLOPE,
                               KEY_ANISO, KEY_CORR_LEN)}
        return {"metrics": metrics,
                "status": aggregate_status(m["status"] for m in metrics.values()),
                "notes": notes}

    rng = np.random.default_rng(_FIELDS_SEED)
    chosen = list(combos)
    if len(combos) > _MAX_FIELDS:
        idx = sorted(rng.choice(len(combos), size=_MAX_FIELDS, replace=False).tolist())
        chosen = [combos[i] for i in idx]
        notes.append(f"场抽样: {len(chosen)}/{len(combos)} (seed={_FIELDS_SEED})")

    fields: list[tuple[FieldKey, np.ndarray, np.ndarray]] = []
    n_dropped = 0
    for combo in chosen:
        fr = _field(real, combo)
        fg = _field(synthetic, combo)
        if fr is None or fg is None or fr.shape != fg.shape:
            n_dropped += 1
            continue
        miss_r = float(np.mean(~np.isfinite(fr)))
        miss_g = float(np.mean(~np.isfinite(fg)))
        if miss_r > _NAN_LIMIT or miss_g > _NAN_LIMIT:
            n_dropped += 1
            continue
        fields.append((combo, fr, fg))
    if n_dropped:
        notes.append(f"场剔除 {n_dropped} 个（形状不一致或缺测 >50%）")
    if not fields:
        reason = "全部场缺测 >50% 或形状不一致，空间指标无法计算"
        metrics = {key: _not_run_entry(th, key, reason)
                   for key in (KEY_VARIO, KEY_GRAD, KEY_NBR_KS, KEY_SLOPE,
                               KEY_ANISO, KEY_CORR_LEN)}
        return {"metrics": metrics,
                "status": aggregate_status(m["status"] for m in metrics.values()),
                "notes": notes}

    per: dict[str, dict[str, float]] = {
        key: {} for key in (KEY_VARIO, KEY_GRAD, KEY_NBR_KS, KEY_SLOPE,
                            KEY_ANISO, KEY_CORR_LEN)}
    skips: dict[str, list[str]] = {key: [] for key in per}
    for combo, fr, fg in fields:
        key = _field_key_txt(combo)
        # 1) 半变异比
        ratios: list[float] = []
        for h in range(1, 6):
            gr, gg = _gamma(fr, h), _gamma(fg, h)
            if gr is None or gg is None or gr <= _EPS:
                ratios = []
                break
            ratios.append(gg / gr)
        if ratios:
            per[KEY_VARIO][key] = float(np.median(ratios))
        else:
            skips[KEY_VARIO].append(f"{key}: γ(1..5) 不可算（场太小/全 NaN/γ≈0）")
        # 2) 梯度 RMS 比
        vr, vg = _grad_rms(fr), _grad_rms(fg)
        if vr is not None and vg is not None and vr > _EPS:
            per[KEY_GRAD][key] = vg / vr
        else:
            skips[KEY_GRAD].append(f"{key}: 梯度 RMS 不可算")
        # 3) 邻域残差 KS
        rr, n_ok_r = _neighbor_mean_valid(fr)
        rg, n_ok_g = _neighbor_mean_valid(fg)
        if n_ok_r >= 10 and n_ok_g >= 10:
            per[KEY_NBR_KS][key] = float(stats.ks_2samp(rr, rg).statistic)
        else:
            skips[KEY_NBR_KS].append(f"{key}: 残差样本不足")
        # 4) 径向谱斜率差
        sr, sg = _radial_spectrum_slope(fr), _radial_spectrum_slope(fg)
        if sr is not None and sg is not None:
            per[KEY_SLOPE][key] = abs(sg - sr)
        else:
            skips[KEY_SLOPE].append(f"{key}: 径向谱斜率不可算")
        # 5) 各向异性
        ar_, ag = _anisotropy_log_ratio(fr), _anisotropy_log_ratio(fg)
        if ar_ is not None and ag is not None:
            per[KEY_ANISO][key] = abs(ag - ar_)
        else:
            skips[KEY_ANISO].append(f"{key}: 梯度能量比≈0 或不可算")
        # 6) 相关长度比
        cr, cg = _corr_length(fr), _corr_length(fg)
        if cr is not None and cg is not None and cr > _EPS:
            per[KEY_CORR_LEN][key] = cg / cr
        else:
            skips[KEY_CORR_LEN].append(f"{key}: 自相关未在 rmax 内降到 1/e（截尾）")

    for key_, list_ in skips.items():
        if list_:
            notes.append(f"{key_}: 跳过 {_short(list_)}")

    metrics: dict[str, dict[str, Any]] = {}

    def range_result(key: str) -> dict[str, Any]:
        lo_key, hi_key = _RATIO_KEYS[key]
        lo, hi = th.get(lo_key), th.get(hi_key)
        dper = per[key]
        if not dper:
            reason = f"{key}: 无可用场；{(skips[key][0] if skips[key] else 'no fields')}"
            return _metric(None, hi, "range", "NOT_RUN", reason)
        values = list(dper.values())
        median = float(np.median(values))
        vmax, vmin = max(values), min(values)
        if hi is not None and vmax > hi:
            value, status = vmax, "FAIL"
        elif lo is not None and vmin < lo:
            value, status = vmin, "FAIL"
        else:
            value, status = median, "PASS"
        if skips[key] and status == "PASS":
            status = "CONDITIONAL_PASS"
        detail = (f"n_fields={len(dper)}; median={median:.4f}; min={vmin:.4f}; "
                  f"max={vmax:.4f}; bounds=[{lo}, {hi}]; 口径: {CALIBER[key]}")
        if skips[key]:
            detail += f"; skipped={len(skips[key])} 场"
        return _metric(value, hi, "range", status, detail)

    def mean_result(key: str) -> dict[str, Any]:
        limit = th.get(_TH_KEYS[key])
        dper = per[key]
        if not dper:
            reason = f"{key}: 无可用场；{(skips[key][0] if skips[key] else 'no fields')}"
            return _metric(None, limit, "max", "NOT_RUN", reason)
        value = float(np.mean(list(dper.values())))
        status = "FAIL" if limit is not None and value > limit else "PASS"
        if skips[key] and status == "PASS":
            status = "CONDITIONAL_PASS"
        detail = (f"n_fields={len(dper)}; mean={value:.4f}; "
                  f"最差={max(dper.values()):.4f}; 口径: {CALIBER[key]}")
        if skips[key]:
            detail += f"; skipped={len(skips[key])} 场"
        return _metric(value, limit, "max", status, detail)

    metrics[KEY_VARIO] = range_result(KEY_VARIO)
    metrics[KEY_GRAD] = range_result(KEY_GRAD)
    metrics[KEY_NBR_KS] = mean_result(KEY_NBR_KS)
    metrics[KEY_SLOPE] = mean_result(KEY_SLOPE)
    metrics[KEY_ANISO] = mean_result(KEY_ANISO)
    metrics[KEY_CORR_LEN] = range_result(KEY_CORR_LEN)

    var_counts: dict[str, int] = {}
    for combo, _, _ in fields:
        var_counts[combo[0]] = var_counts.get(combo[0], 0) + 1
    notes.append("抽样场分布: " + ", ".join(f"{k}×{v}" for k, v in sorted(var_counts.items())))
    return {"metrics": metrics,
            "status": aggregate_status(m["status"] for m in metrics.values()),
            "notes": notes}
