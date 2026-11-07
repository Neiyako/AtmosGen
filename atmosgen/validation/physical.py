"""物理合理性验证（实现计划 §5.4, ``validation/physical.py``）。

口径说明
--------
- 硬约束（任一不满足 → 该指标 FAIL，整体由 :func:`report.aggregate_status` 聚合）：
  ``finite``（全部 data_vars 无 NaN/Inf）、``q_nonneg``、``cloud_nonneg``
  （clwc/ciwc/crwc/cswc）、``t_range``（默认 180–340 K）、``wind_range``
  （|u|,|v| ≤ 150 m/s）、``saturation``（Tetens 公式 + 气压层）、
  ``tcwv_range``（0 ≤ tcwv ≤ 120 kg m⁻²）、``c_nonneg``（含 3D 追迹浓度 ``c``
  时；``c`` 的 NaN/Inf 并入 ``finite``，detail 里注明）；
- 软指标（记录数值，不直接判死）：``mass_budget``（周期边界下总质量相对首步
  漂移，期望 ≤5%）、``smoothness``（棋盘格符号翻转率，有 real 时期望
  ≤ real 的 2 倍）、``equation_residual``（平流–扩散残差 RMS；Phase 4 再设阈值，
  本模块记 INFO / 缺量 NOT_RUN）；
- 阈值：``DEFAULT_THRESHOLDS`` 是 §5.4 硬约束默认值，``thresholds`` 覆盖后合并；
  软指标期望可用 thresholds 中的 ``mass_drift_max``（默认 0.05）与
  ``smoothness_factor_max``（默认 2.0）覆盖；detail 里打印实际使用的阈值/因子；
- NaN：各指标按变量各自压掉非有限值（配对比较用同时有限掩码），排除数量写进
  detail；``finite`` 单独对全量 data_vars 判 FAIL，不用 NaN 掩盖硬约束失败；
- 大数组：逐变量、逐时次处理（不在整份数据集上 stack/copy 全部变量）；dask
  后备数组经 ``.values`` 物化，内存受单变量大小限制（8 GB 机器按切片验证）；
- 缺输入不伪造 PASS：核心变量（t/q/u/v）或 check 所需配对输入（level）缺失时对应
  指标 NOT_RUN 并写原因；数据集范围不适用（无云水变量、无 c、非物理模式、无
  periodic 配置）记 INFO 并写原因（INFO 不是 PASS、不参与聚合，但也不会把合法的
  统计模式数据集整体判成 NOT_RUN）；``boundary_conditions`` / ``kappa`` / ``dt_s``
  也可从数据集 ``atmosgen:contract`` 属性回退读取（便于 evaluate 直接传 synthetic）。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from .report import aggregate_status

__all__ = ["DEFAULT_THRESHOLDS", "run_physical_validation"]

#: 实现计划 §5.4 硬约束默认阈值（键名与任务书 physical 模块 API 对齐）。
DEFAULT_THRESHOLDS: dict[str, float] = {
    "t_min": 180.0,
    "t_max": 340.0,
    "wind_max": 150.0,
    "tcwv_max": 120.0,
    "saturation_factor_max": 1.10,
}

#: 软指标期望（不在 DEFAULT_THRESHOLDS 里，避免改变硬约束阈值公开 API；
#: 可由 thresholds 的 ``mass_drift_max`` / ``smoothness_factor_max`` 覆盖）。
MASS_DRIFT_MAX = 0.05
SMOOTHNESS_FACTOR_MAX = 2.0

_EPS = 1e-12
_DEG_M = 111320.0  # 1 度纬度 ≈ 111320 m
_KELVIN_OFFSET = 273.15
_TETENS_A = 6.112  # hPa
_TETENS_B = 17.67
_TETENS_C = 29.65
_CLOUD_VARS = ("clwc", "ciwc", "crwc", "cswc")
_LEVEL_NAMES = ("level", "pressure_level", "plev", "isobaricInhPa", "isobaricinhpa")
#: 可识别的气压层单位 → 换算到 hPa 的因子（大小写/空格无关）。
_LEVEL_HPA_UNITS = ("hpa", "hectopascal", "hectopascals", "mb", "mbar",
                    "millibar", "millibars")
_LEVEL_PA_UNITS = ("pa", "pascal", "pascals")

_DETAIL_CAP = 6


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _metric(value: object, threshold: float | None, comparison: str, status: str,
            detail: str) -> dict[str, Any]:
    nvalue = float(value) if value is not None else float("nan")
    return {"value": nvalue, "threshold": threshold, "comparison": comparison,
            "status": status, "detail": detail}


def _not_run(threshold: float | None, comparison: str, reason: str) -> dict[str, Any]:
    return _metric(None, threshold, comparison, "NOT_RUN", reason)


def _info(threshold: float | None, value: object, reason: str) -> dict[str, Any]:
    return _metric(value, threshold, "max" if threshold is not None else "info",
                   "INFO", reason)


def _short(items: list[str], cap: int = _DETAIL_CAP) -> str:
    if len(items) > cap:
        return "; ".join(items[:cap]) + f"; (+{len(items) - cap} more)"
    return "; ".join(items)


def _merge_thresholds(thresholds: Mapping[str, float] | None) -> dict[str, float]:
    th = dict(DEFAULT_THRESHOLDS)
    for key, raw in (thresholds or {}).items():
        try:
            fv = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            th[str(key)] = fv
    return th


def _numeric(arr: np.ndarray) -> bool:
    return bool(np.issubdtype(arr.dtype, np.number))


def _contract(ds) -> dict[str, Any]:
    """读取 ``atmosgen:contract``（JSON 字符串或 dict）；失败返回空 dict。"""

    raw = ds.attrs.get("atmosgen:contract")
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, (str, bytes)):
        try:
            obj = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return obj if isinstance(obj, dict) else {}
    return {}


def _nonneg_entry(ds, name: str, label: str, extra: str = "",
                  absent_status: str = "NOT_RUN") -> dict[str, Any]:
    """单变量非负硬约束：value = 负值元素数。

    ``absent_status="INFO"`` 用于「变量缺失 = 该约束不适用」的可选变量（c）。
    """

    if name not in ds.data_vars:
        reason = f"数据集无 {name} 变量，无法检查{label}"
        if absent_status == "INFO":
            return _info(None, None, reason + "（该约束不适用，INFO 不是 PASS）")
        return _not_run(0.0, "max", reason)
    arr = np.asarray(ds[name].values)
    if not _numeric(arr):
        return _not_run(0.0, "max",
                        f"{name} 的 dtype={arr.dtype} 非数值，无法检查{label}")
    finite = np.isfinite(arr)
    n_valid = int(np.count_nonzero(finite))
    n_skipped = int(arr.size - n_valid)
    if n_valid == 0:
        return _not_run(0.0, "max",
                        f"{name} 全部为非有限值（n={arr.size}），无法判定{label}")
    n_neg = int(np.count_nonzero(finite & (arr < 0)))
    lo = float(np.nanmin(arr))
    detail = (f"n={n_valid}, n_negative={n_neg}, min={lo:.4g}, "
              f"排除非有限={n_skipped}")
    if extra:
        detail += "; " + extra
    status = "FAIL" if n_neg else "PASS"
    return _metric(n_neg, 0.0, "max", status, detail)


def _range_entry(ds, name: str, lo: float, hi: float, threshold: float,
                 label: str, absent_status: str = "NOT_RUN") -> dict[str, Any]:
    """单变量范围硬约束：value = 越界侧最差值（无越界时取最大值代表测度）。

    ``absent_status="INFO"`` 用于「变量缺失 = 该约束不适用」的可选变量（tcwv）。
    """

    if name not in ds.data_vars:
        reason = f"数据集无 {name} 变量，无法检查{label}"
        if absent_status == "INFO":
            return _info(None, None, reason + "（该约束不适用，INFO 不是 PASS）")
        return _not_run(threshold, "range", reason)
    arr = np.asarray(ds[name].values)
    if not _numeric(arr):
        return _not_run(threshold, "range",
                        f"{name} 的 dtype={arr.dtype} 非数值，无法检查{label}")
    finite = np.isfinite(arr)
    n_valid = int(np.count_nonzero(finite))
    n_skipped = int(arr.size - n_valid)
    if n_valid == 0:
        return _not_run(threshold, "range",
                        f"{name} 全部为非有限值（n={arr.size}），无法检查{label}")
    n_below = int(np.count_nonzero(finite & (arr < lo)))
    n_above = int(np.count_nonzero(finite & (arr > hi)))
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if n_above:
        value = mx
    elif n_below:
        value = mn
    else:
        value = mx
    n_bad = n_below + n_above
    detail = (f"n={n_valid}, min={mn:.4g}, max={mx:.4g}, "
              f"bounds=[{lo:g}, {hi:g}], n_below={n_below}, n_above={n_above}, "
              f"排除非有限={n_skipped}")
    status = "FAIL" if n_bad else "PASS"
    return _metric(value, threshold, "range", status, detail)


# --------------------------------------------------------------------------- #
# 硬约束：finite / q / clouds / t / wind / saturation / tcwv / c
# --------------------------------------------------------------------------- #
def _finite_entry(ds) -> dict[str, Any]:
    names = sorted(str(n) for n in ds.data_vars)
    if not names:
        return _not_run(0.0, "max", "数据集无 data_vars，无法检查有限性")
    bad = 0
    total = 0
    per: list[str] = []
    unverified: list[str] = []
    # Mode 3 screening mask: cells rejected by the hybrid pipeline are stored as
    # NaN (documented in rejected.jsonl). Those are "missing by screening", not
    # corrupt values, so NaN is tolerated while Inf always counts as bad.
    screening = ds.attrs.get("atmosgen_screening") or {}
    masked_allowed = bool(screening.get("mask_applied")) if isinstance(screening, Mapping) else False
    if not masked_allowed:
        # the embedded contract carries the screening declaration (dataset attrs
        # starting with "atmosgen" are stripped by attach_contract)
        raw_contract = ds.attrs.get("atmosgen:contract")
        if isinstance(raw_contract, str):
            try:
                contract_doc = json.loads(raw_contract)
            except Exception:  # pragma: no cover - malformed contract
                contract_doc = {}
            quality = contract_doc.get("quality_metrics") or {}
            for key in ("screening", "screening_before"):
                block = quality.get(key) or {}
                if isinstance(block, Mapping) and block.get("mask_applied"):
                    masked_allowed = True
                    break
    masked = 0
    for name in names:
        arr = np.asarray(ds[name].values)
        total += int(arr.size)
        if not _numeric(arr):
            unverified.append(f"{name}({arr.dtype})")
            continue
        non_finite = ~np.isfinite(arr)
        if masked_allowed:
            n_nan = int(np.count_nonzero(np.isnan(arr)))
            masked += n_nan
            n_bad = int(np.count_nonzero(np.isinf(arr)))
        else:
            n_bad = int(np.count_nonzero(non_finite))
        bad += n_bad
        if n_bad:
            per.append(f"{name}={n_bad}")
    if unverified:
        return _not_run(0.0, "max",
                        f"变量 dtype 非数值，有限性无法验证: {_short(unverified)}")
    detail = f"data_vars={len(names)}, elements={total}, bad={bad}"
    if masked_allowed:
        detail += f"; 筛检遮罩 NaN {masked} 个（已声明 atmosgen_screening.mask_applied）"
    if per:
        detail += "; 逐变量坏值: " + _short(per)
    if "c" in ds.data_vars:
        detail += "; c 的 NaN/Inf 已并入本指标（c_finite）"
    status = "PASS" if bad == 0 else "FAIL"
    return _metric(bad, 0.0, "max", status, detail)


def _cloud_nonneg_entry(ds) -> dict[str, Any]:
    present = [n for n in _CLOUD_VARS if n in ds.data_vars]
    if not present:
        return _info(None, None,
                     "数据集不含云水物质变量 clwc/ciwc/crwc/cswc，"
                     "该约束不适用（INFO 不是 PASS）")
    total_neg = 0
    total_valid = 0
    total_skipped = 0
    parts: list[str] = []
    for name in present:
        arr = np.asarray(ds[name].values)
        if not _numeric(arr):
            parts.append(f"{name}: dtype={arr.dtype} 非数值（未验证）")
            continue
        finite = np.isfinite(arr)
        n_valid = int(np.count_nonzero(finite))
        n_skipped = int(arr.size - n_valid)
        total_valid += n_valid
        total_skipped += n_skipped
        n_neg = int(np.count_nonzero(finite & (arr < 0)))
        total_neg += n_neg
        lo = float(np.nanmin(arr)) if n_valid else float("nan")
        parts.append(f"{name}: neg={n_neg}, n={n_valid}, min={lo:.4g}")
    if total_valid == 0:
        return _not_run(0.0, "max",
                        "所列云水变量全部为非有限值，无法判定: " + _short(parts))
    detail = ("; ".join(parts)
              + f"; 合计负值={total_neg}, 排除非有限={total_skipped}")
    status = "FAIL" if total_neg else "PASS"
    return _metric(total_neg, 0.0, "max", status, detail)


def _wind_entry(ds, th: Mapping[str, float]) -> dict[str, Any]:
    limit = float(th.get("wind_max", DEFAULT_THRESHOLDS["wind_max"]))
    present = [n for n in ("u", "v") if n in ds.data_vars]
    if not present:
        return _not_run(limit, "max",
                        "数据集无 u/v 风场变量，无法检查风速范围（|u|,|v|）")
    worst = 0.0
    total_over = 0
    total_valid = 0
    total_skipped = 0
    unvalidated: list[str] = []
    parts: list[str] = []
    for name in present:
        arr = np.asarray(ds[name].values)
        if not _numeric(arr):
            parts.append(f"{name}: dtype={arr.dtype} 非数值（未验证）")
            continue
        finite = np.isfinite(arr)
        n_valid = int(np.count_nonzero(finite))
        n_skipped = int(arr.size - n_valid)
        total_valid += n_valid
        total_skipped += n_skipped
        if n_valid:
            mn = float(np.nanmin(arr))
            mx = float(np.nanmax(arr))
            absmax = max(abs(mn), abs(mx))
            n_over = int(np.count_nonzero(finite & ((arr > limit) | (arr < -limit))))
            total_over += n_over
            worst = max(worst, absmax)
            parts.append(f"{name}: max|.|={absmax:.4g}, n={n_valid}, n_over={n_over}")
        else:
            unvalidated.append(name)
            parts.append(f"{name}: 全非有限（该分量未验证）")
    if total_valid == 0:
        return _not_run(limit, "max",
                        "u/v 全部为非有限值，无法检查风速范围: " + _short(parts))
    missing = [n for n in ("u", "v") if n not in ds.data_vars]
    if total_over:
        status = "FAIL"
    elif missing or unvalidated:
        status = "CONDITIONAL_PASS"  # 缺分量/某分量全非有限 → 只做了部分验证
    else:
        status = "PASS"
    detail = (f"limit={limit:g} m/s; " + "; ".join(parts)
              + f"; 排除非有限={total_skipped}")
    if missing:
        detail += f"; 缺变量 {missing}（仅对现有分量部分验证）"
    if unvalidated:
        detail += f"; 未验证分量 {unvalidated}"
    return _metric(worst, limit, "max", status, detail)


def _coord_units(ds, coord_name: str) -> tuple[str | None, str]:
    """返回 (level 单位字符串, 来源说明)。coord attrs → dataset attrs → 契约。"""

    coord = ds.coords[coord_name] if coord_name in ds.coords else ds[coord_name]
    units = coord.attrs.get("units") or coord.attrs.get("unit")
    if units:
        return str(units), "level.attrs.units"
    for key in ("level_units", "pressure_level_units"):
        if ds.attrs.get(key):
            return str(ds.attrs[key]), f"dataset.attrs.{key}"
    contract = _contract(ds)
    lv = (contract.get("coordinates") or {}).get("level") if contract else None
    if isinstance(lv, Mapping):
        if str(lv.get("type")) == "model_level":
            return None, "contract.coordinates.level.type=model_level（非气压层）"
        if lv.get("units"):
            return str(lv["units"]), "contract.coordinates.level.units"
    return None, "level 无 units 属性（coord.attrs / dataset.attrs / contract 均无）"


def _level_to_hpa(ds) -> tuple[float | None, np.ndarray | None, str | None, str]:
    """定位气压层坐标并换算到 hPa。

    返回 ``(单位换算因子, level_hpa 数组, 坐标名, 说明)``；不可用时前两项为 None，
    说明为 NOT_RUN 原因。
    """

    coord_name = next((n for n in _LEVEL_NAMES if n in ds.coords or n in ds.variables), None)
    if coord_name is None:
        return None, None, None, "数据集无 level 坐标（已查 level/pressure_level/plev），无法按气压计算 q_sat"
    units, source = _coord_units(ds, coord_name)
    if not units:
        return None, None, coord_name, f"level 单位不可用: {source}"
    norm = units.strip().lower().replace(" ", "")
    if norm in _LEVEL_HPA_UNITS or "hpa" in norm or "hectopascal" in norm:
        factor = 1.0
    elif norm in _LEVEL_PA_UNITS or norm.startswith("pa") or "pascal" in norm:
        factor = 0.01  # Pa → hPa
    else:
        return None, None, coord_name, (
            f"level 单位 {units!r} 无法识别为气压层（需 hPa/mb/millibars/Pa），"
            f"拒绝按气压计算 q_sat（来源 {source}）")
    raw = np.asarray(ds[coord_name].values, dtype=float).ravel()
    if raw.size == 0:
        return None, None, coord_name, "level 坐标为空"
    if not np.all(np.isfinite(raw)):
        return None, None, coord_name, "level 坐标含非有限值"
    level_hpa = raw * factor
    if np.any(level_hpa <= 0):
        return None, None, coord_name, "level 坐标换算后存在 ≤0 的气压值"
    note = f"level_units={units!r}（来源 {source}，换算因子={factor:g}）"
    return factor, level_hpa, coord_name, note


def _saturation_entry(ds, th: Mapping[str, float]) -> dict[str, Any]:
    """饱和约束：逐点 q ≤ factor × q_sat(T, p)（Tetens，p 用 hPa）。

    value = 超限点数；缺 t/q/level 或 level 非气压层 → NOT_RUN（不伪造 PASS）。
    """

    factor = float(th.get("saturation_factor_max",
                          DEFAULT_THRESHOLDS["saturation_factor_max"]))
    if "q" not in ds.data_vars:
        return _not_run(0.0, "max",
                        "数据集无 q（比湿）变量，无法检查饱和约束")
    if "t" not in ds.data_vars:
        return _not_run(0.0, "max",
                        "数据集无 t（温度）变量，无法检查饱和约束")
    _, level_hpa, level_name, level_note = _level_to_hpa(ds)
    if level_hpa is None:
        return _not_run(0.0, "max", f"饱和约束未运行: {level_note}")

    t_da, q_da = ds["t"], ds["q"]
    if set(t_da.dims) != set(q_da.dims):
        return _not_run(0.0, "max", (
            f"饱和约束未运行: t 与 q 维度集合不一致 "
            f"{sorted(t_da.dims)} vs {sorted(q_da.dims)}"))
    if dict(t_da.sizes) != dict(q_da.sizes):
        return _not_run(0.0, "max", (
            f"饱和约束未运行: t 与 q 形状不一致 "
            f"{dict(t_da.sizes)} vs {dict(q_da.sizes)}"))
    if level_name not in t_da.dims:
        return _not_run(0.0, "max", (
            f"饱和约束未运行: t 的维度 {sorted(t_da.dims)} 不含 level 维"
            f"（坐标名 {level_name!r}），无法逐点配对气压"))
    if int(t_da.sizes[level_name]) != int(level_hpa.size):
        return _not_run(0.0, "max", (
            f"饱和约束未运行: level 坐标长度 {level_hpa.size} 与 t 的 "
            f"{level_name} 维 {t_da.sizes[level_name]} 不一致"))

    has_time = "time" in t_da.dims
    n_time = int(t_da.sizes["time"]) if has_time else 1
    if has_time and int(q_da.sizes["time"]) != n_time:
        return _not_run(0.0, "max", "饱和约束未运行: t 与 q 的 time 维长度不一致")

    n_total = 0
    n_nan = 0
    n_bad_pressure = 0
    n_violation = 0
    worst_ratio = float("-inf")
    worst_q = float("nan")
    worst_qsat = float("nan")

    for it in range(n_time):
        if has_time:
            t_slice = np.asarray(t_da.isel(time=it).values, dtype=float)
            q_slice = np.asarray(q_da.isel(time=it).values, dtype=float)
            dims = tuple(d for d in t_da.dims if d != "time")
        else:
            t_slice = np.asarray(t_da.values, dtype=float)
            q_slice = np.asarray(q_da.values, dtype=float)
            dims = tuple(t_da.dims)
        shape = [1] * len(dims)
        shape[dims.index(level_name)] = int(level_hpa.size)
        p_hpa = level_hpa.reshape(shape)  # 只在 level 轴广播，不复制整场

        n_total += int(t_slice.size)
        finite_tq = np.isfinite(t_slice) & np.isfinite(q_slice)
        n_nan += int(t_slice.size - np.count_nonzero(finite_tq))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            es = _TETENS_A * np.exp(
                _TETENS_B * (t_slice - _KELVIN_OFFSET) / (t_slice - _TETENS_C))
            denom = p_hpa - 0.378 * es
            ok_pressure = np.isfinite(es) & np.isfinite(denom) & (denom > 0)
            n_bad_pressure += int(
                np.count_nonzero(finite_tq & ~ok_pressure))
            qsat = np.where(ok_pressure, 0.622 * es / np.where(ok_pressure, denom, 1.0),
                            np.nan)
            valid = finite_tq & ok_pressure & np.isfinite(qsat) & (qsat > 0)
            n_valid = int(np.count_nonzero(valid))
            if n_valid == 0:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(valid, q_slice / qsat, np.nan)
            violation = valid & (q_slice > factor * qsat)
            n_violation += int(np.count_nonzero(violation))
            if np.any(violation):
                masked = np.where(violation, ratio, np.nan)
                local_worst = float(np.nanmax(masked))
                if local_worst > worst_ratio:
                    worst_ratio = local_worst
                    idx = np.unravel_index(
                        int(np.nanargmax(masked)), masked.shape)
                    worst_q = float(q_slice[idx])
                    worst_qsat = float(qsat[idx])

    if n_total == 0:
        return _not_run(0.0, "max", "饱和约束未运行: t/q 无元素")
    if n_violation == 0 and n_bad_pressure + n_nan >= n_total:
        return _not_run(0.0, "max", (
            f"饱和约束未运行: 无可用配对点（非有限 {n_nan}, 非法气压 {n_bad_pressure}, "
            f"总点数 {n_total}）"))

    detail = (f"n_pairs={n_total - n_nan - n_bad_pressure}, n_violation={n_violation}, "
              f"factor={factor:g}, {level_note}, n_time={n_time}, "
              f"排除非有限={n_nan}, 排除非法气压（p≤0.378e_s）={n_bad_pressure}")
    if n_violation:
        detail += (f"; worst q/q_sat={worst_ratio:.4g} "
                   f"(q={worst_q:.6g}, q_sat={worst_qsat:.6g})")
    status = "FAIL" if n_violation else "PASS"
    return _metric(n_violation, 0.0, "max", status, detail)


# --------------------------------------------------------------------------- #
# 软指标：mass_budget / smoothness / equation_residual
# --------------------------------------------------------------------------- #
def _boundary_kind(ds, pp: Mapping[str, Any] | None) -> tuple[str | None, str]:
    """读取边界条件 kind：physical_parameters 优先，contract 回退。"""

    if isinstance(pp, Mapping):
        bc = pp.get("boundary_conditions", pp.get("boundary_condition"))
        if isinstance(bc, Mapping) and bc.get("kind") is not None:
            return str(bc["kind"]), "physical_parameters"
        if isinstance(bc, str):
            return bc, "physical_parameters"
        if pp.get("boundary_condition_kind") is not None:
            return str(pp["boundary_condition_kind"]), "physical_parameters"
    contract = _contract(ds)
    bc = contract.get("boundary_conditions") if contract else None
    if isinstance(bc, Mapping) and bc.get("kind") is not None:
        return str(bc["kind"]), "contract"
    return None, ""


def _mass_target(ds) -> tuple[str | None, object | None, str]:
    """选择总量漂移所用的场：周期物理模式优先 3D 追迹浓度 c，其次水柱 tcwv / q。"""

    for name, note in (
        ("c", "3D 追迹浓度全场求和"),
        ("tcwv", "整层水汽柱全场求和"),
        ("q", "比湿全场求和（未做气压/面积加权，仅相对漂移诊断）"),
    ):
        if name in ds.data_vars:
            da = ds[name]
            if all(d in da.dims for d in ("time", "latitude", "longitude")):
                return name, da, note
    return None, None, "无可用于总量漂移的场（需含 time/latitude/longitude 的 c/tcwv/q）"


def _mass_budget_entry(ds, pp: Mapping[str, Any] | None,
                       th: Mapping[str, float]) -> dict[str, Any]:
    """周期边界下总质量相对首步漂移：|sum_t - sum_0| / max(|sum_0|, eps) ≤ 5%。"""

    drift_max = float(th.get("mass_drift_max", MASS_DRIFT_MAX))
    kind, source = _boundary_kind(ds, pp)
    if kind is None:
        return _info(None, None, (
            "未提供周期性边界条件（physical_parameters.boundary_conditions "
            "与数据集契约均无可识别的 kind），质量守恒不适用（INFO 不是 PASS）"))
    if kind.strip().lower() != "periodic":
        return _info(None, None, (
            f"边界条件 kind={kind!r} 非 periodic，质量守恒/水柱守恒不适用"
            f"（来源 {source}；INFO 不是 PASS）"))

    name, da, note = _mass_target(ds)
    if name is None or da is None:
        return _not_run(drift_max, "max", "质量守恒未运行: " + note)
    arr = np.asarray(da.values)
    if not _numeric(arr):
        return _not_run(drift_max, "max",
                        f"质量守恒未运行: {name} dtype={arr.dtype} 非数值")
    taxis = da.get_axis_num("time")
    moved = np.moveaxis(arr, taxis, 0)
    n_time = int(moved.shape[0])
    if n_time < 2:
        return _not_run(drift_max, "max",
                        f"质量守恒未运行: {name} 只有 {n_time} 个时次，无法计算时序漂移")
    flat = moved.reshape(n_time, -1)
    sums: list[float] = []
    n_skipped = 0
    for it in range(n_time):
        row = flat[it]
        finite = np.isfinite(row)
        n_skipped += int(row.size - np.count_nonzero(finite))
        if not finite.any():
            return _not_run(drift_max, "max",
                            f"质量守恒未运行: {name} 第 {it} 个时次全部为 NaN/Inf")
        sums.append(float(np.sum(row[finite])))
    sums_arr = np.asarray(sums, dtype=float)
    base = float(sums_arr[0])
    denom = max(abs(base), _EPS)
    drift = float(np.max(np.abs(sums_arr - base)) / denom)
    unstable = abs(base) <= _EPS
    status = "CONDITIONAL_PASS" if (drift > drift_max or unstable) else "PASS"
    detail = (f"var={name}（{note}）, n_time={n_time}, mass_0={base:.6g}, "
              f"mass_last={float(sums_arr[-1]):.6g}, "
              f"max|Δ|/max(|m0|,eps)={drift:.4g}（期望 ≤{drift_max:g}）, "
              f"边界={kind}（来源 {source}）, 排除非有限元素={n_skipped}")
    if unstable:
        detail += "; 首步总量≈0（|m0|≤eps），比值按 eps 归一化不稳定"
    return _metric(drift, drift_max, "max", status, detail)


def _select_2d_field(ds) -> tuple[str | None, object | None]:
    """选 2D 场（time×lat×lon 或 lat×lon，无 level 维）：c → tcwv → 其它。"""

    for name in ("c", "tcwv"):
        if name in ds.data_vars:
            da = ds[name]
            extra = set(da.dims) - {"time"}
            if extra == {"latitude", "longitude"}:
                return name, da
    for name in sorted(str(n) for n in ds.data_vars):
        da = ds[name]
        extra = set(da.dims) - {"time"}
        if extra == {"latitude", "longitude"}:
            return name, da
    return None, None


def _checkerboard_stats(da) -> dict[str, Any] | None:
    """2D 场的相邻格点符号翻转率（所有时次均值）。"""

    arr = np.asarray(da.values)
    if not _numeric(arr):
        return None
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(float)
    has_time = "time" in da.dims
    if has_time:
        frames = np.moveaxis(arr, da.get_axis_num("time"), 0)
        frame_list = [frames[i] for i in range(frames.shape[0])]
    else:
        frame_list = [arr]
    rates: list[float] = []
    n_pairs_possible = 0
    n_pairs_valid = 0
    for frame in frame_list:
        if frame.ndim != 2 or frame.shape[0] < 2 or frame.shape[1] < 2:
            continue
        finite = np.isfinite(frame)
        right_valid = finite[:, :-1] & finite[:, 1:]
        down_valid = finite[:-1, :] & finite[1:, :]
        possible = int(frame.shape[0] * (frame.shape[1] - 1)
                       + (frame.shape[0] - 1) * frame.shape[1])
        valid = int(np.count_nonzero(right_valid) + np.count_nonzero(down_valid))
        if valid == 0:
            continue
        with np.errstate(invalid="ignore", over="ignore"):
            right_prod = frame[:, :-1] * frame[:, 1:]
            down_prod = frame[:-1, :] * frame[1:, :]
        flips = int(np.count_nonzero(right_valid & (right_prod < 0)))
        flips += int(np.count_nonzero(down_valid & (down_prod < 0)))
        rates.append(flips / valid)
        n_pairs_possible += possible
        n_pairs_valid += valid
    if not rates:
        return None
    return {
        "rate": float(np.mean(rates)),
        "n_frames": len(rates),
        "n_pairs_valid": n_pairs_valid,
        "n_pairs_skipped": n_pairs_possible - n_pairs_valid,
    }


def _smoothness_entry(ds, real, th: Mapping[str, float]) -> dict[str, Any]:
    """棋盘格指标：无 real → INFO；有 real → gen/real 比值，期望 ≤ 2×（超过 CONDITIONAL_PASS）。"""

    factor = float(th.get("smoothness_factor_max", SMOOTHNESS_FACTOR_MAX))
    name, da = _select_2d_field(ds)
    if name is None or da is None:
        return _not_run(None, "info", (
            "平滑性/棋盘格未运行: 数据集无 2D 场"
            "（time×latitude×longitude 的 c/tcwv 等），指标不适用"))
    gen = _checkerboard_stats(da)
    if gen is None:
        return _not_run(None, "info",
                        f"平滑性/棋盘格未运行: {name} 的 2D 场样本不足或全非有限")
    gen_rate = float(gen["rate"])

    if real is None:
        return _info(None, gen_rate, (
            f"var={name}, gen_flip_rate={gen_rate:.4g}, n_frames={gen['n_frames']}, "
            f"有效相邻对={gen['n_pairs_valid']}, 排除非有限对={gen['n_pairs_skipped']}; "
            f"未提供 real → 仅记录数值（INFO）"))

    if name not in real.data_vars:
        return _info(None, gen_rate, (
            f"var={name}, gen_flip_rate={gen_rate:.4g}; real 无同变量 {name}，"
            f"无法计算 gen/real 比值（INFO）"))
    real_da = real[name]
    if set(real_da.dims) - {"time"} != set(da.dims) - {"time"}:
        return _info(None, gen_rate, (
            f"var={name}, gen_flip_rate={gen_rate:.4g}; real 的维度 "
            f"{sorted(real_da.dims)} 与 synthetic {sorted(da.dims)} 不一致，"
            f"无法比较（INFO）"))
    real_shape = [int(real_da.sizes[d]) for d in ("latitude", "longitude")]
    gen_shape = [int(da.sizes[d]) for d in ("latitude", "longitude")]
    if real_shape != gen_shape:
        return _info(None, gen_rate, (
            f"var={name}, gen_flip_rate={gen_rate:.4g}; real 网格 "
            f"{real_shape[0]}×{real_shape[1]} 与 synthetic "
            f"{gen_shape[0]}×{gen_shape[1]} 不一致，无法比较比值（INFO）"))
    real_stats = _checkerboard_stats(real_da)
    if real_stats is None:
        return _info(None, gen_rate, (
            f"var={name}, gen_flip_rate={gen_rate:.4g}; real 同变量样本不足，"
            f"无法计算比值（INFO）"))
    real_rate = float(real_stats["rate"])

    if real_rate <= _EPS:
        if gen_rate <= _EPS:
            return _metric(1.0, factor, "max", "INFO", (
                f"var={name}, gen_flip_rate=real_flip_rate=0（双方均无棋盘格）；"
                f"退化情形，比值记 1.0（INFO）"))
        return _metric(float("inf"), factor, "max", "CONDITIONAL_PASS", (
            f"var={name}, gen_flip_rate={gen_rate:.4g}, real_flip_rate=0 → "
            f"比值无穷（real 无相邻符号翻转），超过期望 ≤{factor:g}×real；"
            f"n_frames gen/real={gen['n_frames']}/{real_stats['n_frames']}"))
    ratio = gen_rate / real_rate
    status = "CONDITIONAL_PASS" if ratio > factor else "PASS"
    detail = (f"var={name}, gen_flip_rate={gen_rate:.4g}, "
              f"real_flip_rate={real_rate:.4g}, ratio={ratio:.4g}"
              f"（期望 ≤{factor:g}×real）, n_frames gen/real="
              f"{gen['n_frames']}/{real_stats['n_frames']}, "
              f"排除非有限对 gen={gen['n_pairs_skipped']}/real={real_stats['n_pairs_skipped']}")
    if status == "CONDITIONAL_PASS":
        detail += "；生成场棋盘格显著强于真实场（软指标告警，不直接 FAIL）"
    return _metric(ratio, factor, "max", status, detail)


def _physics_context(ds, pp: Mapping[str, Any] | None) -> bool:
    """判断数据集是否处于物理模式语境：有 c / 提供 physical_parameters / 契约声明。

    物理模式语境下缺量是「未完成的验证」（NOT_RUN）；非物理模式（纯统计数据集）
    缺 c 是「不适用」（INFO），避免把合法统计数据集整体误判成 NOT_RUN。
    """

    if "c" in ds.data_vars:
        return True
    if isinstance(pp, Mapping) and bool(pp):
        return True
    contract = _contract(ds)
    if contract and (contract.get("physical_parameters")
                     or contract.get("boundary_conditions")):
        return True
    return False


def _squeeze_single_level(da):
    """若存在长度为 1 的垂直层维（physics 模式 level=1），压掉该维。"""

    for lname in _LEVEL_NAMES:
        if lname in da.dims and int(da.sizes[lname]) == 1:
            return da.isel({lname: 0}, drop=True)
    return da


def _kappa(pp: Mapping[str, Any] | None, ds) -> tuple[float | None, str]:
    """读取 kappa（扩散系数）：physical_parameters → contract 回退。"""

    candidates: list[tuple[object, str]] = [(pp, "physical_parameters")]
    contract = _contract(ds)
    if contract:
        candidates.append((contract.get("physical_parameters"),
                           "contract.physical_parameters"))
    for mapping, source in candidates:
        if not isinstance(mapping, Mapping):
            continue
        raw = mapping.get("kappa")
        if raw is None and isinstance(mapping.get("parameters"), Mapping):
            raw = mapping["parameters"].get("kappa")
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        return value, source
    return None, ""


def _dt_s(pp: Mapping[str, Any] | None, ds) -> tuple[float | None, str]:
    """读取 dt_s（仅当 time 坐标不是 datetime 时使用）：physical_parameters → contract。"""

    candidates: list[tuple[object, str]] = [(pp, "physical_parameters")]
    contract = _contract(ds)
    if contract:
        candidates.append((contract.get("physical_parameters"),
                           "contract.physical_parameters"))
    for mapping, source in candidates:
        if not isinstance(mapping, Mapping):
            continue
        raw = mapping.get("dt_s")
        if raw is None:
            continue
        try:
            return float(raw), source
        except (TypeError, ValueError):
            continue
    return None, ""


def _uniform_dt_seconds(ds, pp) -> tuple[float | None, str]:
    """从 time 坐标（datetime64）或 physical_parameters.dt_s 得到均匀步长（秒）。"""

    if "time" in ds.coords:
        values = np.asarray(ds["time"].values)
        if np.issubdtype(values.dtype, np.datetime64):
            ns = values.astype("datetime64[ns]").astype(np.int64)
            if np.any(ns == np.iinfo(np.int64).min):
                return None, "time 坐标含 NaT"
            if ns.size < 2:
                return None, "time 长度 < 2"
            diffs = np.diff(ns).astype(float) / 1e9
            if not np.all(np.isfinite(diffs)) or np.any(diffs <= 0):
                return None, "time 非严格单调递增"
            med = float(np.median(diffs))
            if med <= 0:
                return None, "time 步长 ≤ 0"
            if float(diffs.max() - diffs.min()) > 1e-6 * max(med, 1.0):
                return None, (
                    f"time 步长不均匀（{float(diffs.min()):g}.."
                    f"{float(diffs.max()):g} s），残差需均匀 dt")
            return med, "time 坐标差分"
    dt_s, source = _dt_s(pp, ds)
    if dt_s is None:
        return None, ("time 非 datetime 且 physical_parameters.dt_s 缺失，"
                      "无法确定均匀时间步长")
    if not np.isfinite(dt_s) or dt_s <= 0:
        return None, f"physical_parameters.dt_s={dt_s!r} 非法（需 >0）"
    return dt_s, source


def _equation_residual_entry(ds, pp: Mapping[str, Any] | None) -> dict[str, Any]:
    """平流–扩散方程残差 RMS：∂c/∂t + u∂c/∂x + v∂c/∂y − κ∇²c。

    仅在 u、v、c 与 ``physical_parameters.kappa`` 齐备时计算；缺量 → NOT_RUN。
    Phase 4 才设阈值，故本项固定 INFO（数值 + 单位 + 各项 RMS 写进 detail）。
    """

    physics_mode = _physics_context(ds, pp)
    pp_block = pp if isinstance(pp, Mapping) else {}
    constant_u = pp_block.get("u0")
    constant_v = pp_block.get("v0")
    missing = [n for n in ("c",) if n not in ds.data_vars]
    if missing:
        reason = (f"方程残差未运行: 缺变量 {missing}"
                  f"（需 c 才能代入 ∂c/∂t+u∂c/∂x+v∂c/∂y−κ∇²c）")
        if not physics_mode:
            return _info(None, None, reason + "；数据集无 c 且未提供物理参数，"
                        "非物理模式 → 不适用（INFO 不是 PASS）")
        return _not_run(None, "info", reason)
    # Plan §5.4 allows u/v either as dataset fields OR as solver parameters
    # (u0/v0 from contract.physical_parameters): a constant-wind assumption
    # matches exactly what the Mode 2 solver ran with, so the residual against
    # the ACTUAL parameters is the meaningful check. Missing BOTH → NOT_RUN.
    if "u" not in ds.data_vars and constant_u is None:
        return _not_run(None, "info",
                        "方程残差未运行: 数据集无 u 场且契约 physical_parameters.u0 缺失"
                        "（平流速度不可得）")
    if "v" not in ds.data_vars and constant_v is None:
        return _not_run(None, "info",
                        "方程残差未运行: 数据集无 v 场且契约 physical_parameters.v0 缺失"
                        "（平流速度不可得）")
    kappa, kappa_source = _kappa(pp, ds)
    if kappa is None:
        reason = ("方程残差未运行: physical_parameters.kappa 缺失"
                  "（无法计算扩散项；契约 contract.physical_parameters.kappa 也未提供）")
        if not physics_mode:
            return _info(None, None, reason + "；非物理模式 → 不适用（INFO 不是 PASS）")
        return _not_run(None, "info", reason)
    if not np.isfinite(kappa) or kappa < 0:
        return _not_run(None, "info",
                        f"方程残差未运行: kappa={kappa!r} 非法（需 ≥0 的有限值）")
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        return _not_run(None, "info",
                        "方程残差未运行: 缺 latitude/longitude 坐标，无法换算米制网格")

    c_da = _squeeze_single_level(ds["c"])
    if "u" in ds.data_vars:
        u_da = _squeeze_single_level(ds["u"])
    else:
        u_da = None  # constant u0 from physical_parameters
    if "v" in ds.data_vars:
        v_da = _squeeze_single_level(ds["v"])
    else:
        v_da = None  # constant v0 from physical_parameters
    need = {"time", "latitude", "longitude"}
    if set(c_da.dims) != need:
        return _not_run(None, "info", (
            f"方程残差未运行: c 维度 {sorted(c_da.dims)} 不是 "
            f"(time, latitude, longitude)（level>1 时本实现不选层）"))
    for nm, da in (("u", u_da), ("v", v_da)):
        if da is None:
            continue  # constant wind from physical_parameters (u0/v0)
        if not set(da.dims) <= need or not {"latitude", "longitude"} <= set(da.dims):
            return _not_run(None, "info", (
                f"方程残差未运行: {nm} 维度 {sorted(da.dims)} 不兼容单层 2D 风场"))
        if "time" in da.dims and int(da.sizes["time"]) != int(c_da.sizes["time"]):
            return _not_run(None, "info",
                            f"方程残差未运行: {nm} 的 time 长度与 c 不一致")

    n_time = int(c_da.sizes["time"])
    if n_time < 2:
        return _not_run(None, "info", (
            f"方程残差未运行: c 只有 {n_time} 个时次，无法计算 ∂c/∂t"))
    dt_s, dt_source = _uniform_dt_seconds(ds, pp)
    if dt_s is None:
        return _not_run(None, "info", "方程残差未运行: " + dt_source)

    lat = np.asarray(ds["latitude"].values, dtype=float).ravel()
    lon = np.asarray(ds["longitude"].values, dtype=float).ravel()
    nlat = int(c_da.sizes["latitude"])
    nlon = int(c_da.sizes["longitude"])
    if lat.size != nlat or lon.size != nlon:
        return _not_run(None, "info", (
            f"方程残差未运行: 坐标长度 lat={lat.size}/lon={lon.size} 与 c 网格 "
            f"{nlat}×{nlon} 不一致"))
    if nlat < 3 or nlon < 3:
        return _not_run(None, "info", (
            f"方程残差未运行: 网格 {nlat}×{nlon} 小于 3×3，无法用中心差分算扩散项"))
    if not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))):
        return _not_run(None, "info", "方程残差未运行: 经纬度坐标含非有限值")
    dlat = float(np.mean(np.abs(np.diff(lat))))
    dlon = float(np.mean(np.abs(np.diff(lon))))
    if not (dlat > 0 and dlon > 0):
        return _not_run(None, "info",
                        "方程残差未运行: 经纬度网格间距为 0（无法换算米制）")
    dy_m = _DEG_M * dlat
    dx_row = np.maximum(
        _DEG_M * np.cos(np.deg2rad(lat[1:-1])) * dlon, _EPS)  # 随纬度变化

    c_da = c_da.transpose("time", "latitude", "longitude")
    u_static = v_static = None
    if u_da is None:
        u_static = np.full((c_da.sizes["latitude"], c_da.sizes["longitude"]),
                           float(constant_u), dtype=float)
    elif "time" in u_da.dims:
        u_da = u_da.transpose("time", "latitude", "longitude")
    else:
        u_static = np.asarray(u_da.transpose("latitude", "longitude").values,
                              dtype=float)
    if v_da is None:
        v_static = np.full((c_da.sizes["latitude"], c_da.sizes["longitude"]),
                           float(constant_v), dtype=float)
    elif "time" in v_da.dims:
        v_da = v_da.transpose("time", "latitude", "longitude")
    else:
        v_static = np.asarray(v_da.transpose("latitude", "longitude").values,
                              dtype=float)

    ss = ss_adv = ss_diff = 0.0
    n_valid = 0
    n_skipped = 0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for it in range(n_time - 1):
            c0 = np.asarray(c_da.isel(time=it).values, dtype=float)
            c1 = np.asarray(c_da.isel(time=it + 1).values, dtype=float)
            u_t = (np.asarray(u_da.isel(time=it).values, dtype=float)
                   if (u_da is not None and "time" in u_da.dims) else u_static)
            v_t = (np.asarray(v_da.isel(time=it).values, dtype=float)
                   if (v_da is not None and "time" in v_da.dims) else v_static)
            dcdt = (c1[1:-1, 1:-1] - c0[1:-1, 1:-1]) / dt_s
            dcdx = ((c0[1:-1, 2:] - c0[1:-1, :-2])
                    / (2.0 * dx_row[:, None]))
            dcdy = (c0[2:, 1:-1] - c0[:-2, 1:-1]) / (2.0 * dy_m)
            lapx = ((c0[1:-1, 2:] - 2.0 * c0[1:-1, 1:-1] + c0[1:-1, :-2])
                    / (dx_row[:, None] ** 2))
            lapy = ((c0[2:, 1:-1] - 2.0 * c0[1:-1, 1:-1] + c0[:-2, 1:-1])
                    / (dy_m ** 2))
            adv = u_t[1:-1, 1:-1] * dcdx + v_t[1:-1, 1:-1] * dcdy
            diff = kappa * (lapx + lapy)
            residual = dcdt + adv - diff
            valid = np.isfinite(residual)
            n_step_valid = int(np.count_nonzero(valid))
            n_skipped += int(residual.size - n_step_valid)
            if n_step_valid == 0:
                continue
            r = residual[valid]
            a = adv[valid]
            d = diff[valid]
            ss += float(np.sum(r * r))
            ss_adv += float(np.sum(a * a))
            ss_diff += float(np.sum(d * d))
            n_valid += n_step_valid

    if n_valid == 0:
        return _not_run(None, "info", (
            f"方程残差未运行: 无有效点（{n_time - 1} 个时间步全部因 NaN/Inf "
            f"被排除，排除 {n_skipped} 点）"))
    rms = float(np.sqrt(ss / n_valid))
    adv_rms = float(np.sqrt(ss_adv / n_valid))
    diff_rms = float(np.sqrt(ss_diff / n_valid))
    detail = (f"n={n_valid}, rms={rms:.6g}, adv_rms={adv_rms:.6g}, "
              f"diffusion_rms={diff_rms:.6g}; kappa={kappa:g}"
              f"（来源 {kappa_source}）, dt={dt_s:g} s（{dt_source}）, "
              f"dy≈{dy_m:.6g} m, dx≈{float(np.mean(dx_row)):.6g} m（随 cos(lat)）, "
              f"排除非有限/邻居缺失点={n_skipped}, 时间步={n_time - 1}; "
              f"单位=c·s⁻¹；Phase 4 再设阈值（本项 INFO，不作为通过依据）")
    return _metric(rms, None, "info", "INFO", detail)


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def run_physical_validation(
    synthetic,
    *,
    thresholds: Mapping[str, float],
    physical_parameters: Mapping[str, Any] | None = None,
    real=None,
) -> dict[str, Any]:
    """对 ``synthetic`` 做物理合理性验证，返回统一结构。

    参数
    ----
    synthetic:
        ``xarray.Dataset``（规范数据集：t/q/u/v/clwc.../tcwv/c 视数据源而定）。
    thresholds:
        覆盖阈值（与 :data:`DEFAULT_THRESHOLDS` 合并；未知键保留给软指标用）。
    physical_parameters:
        物理参数（含 ``boundary_conditions.kind`` / ``kappa`` / ``dt_s``）；缺失时
        相应指标 NOT_RUN/INFO 并写原因；也可从数据集契约回退读取。
    real:
        可选真实参考数据集，仅用于 ``smoothness`` 的 gen/real 比值。
    """

    th = _merge_thresholds(thresholds)
    notes: list[str] = []
    metrics: dict[str, dict[str, Any]] = {}

    # ---- 硬约束 -----------------------------------------------------------
    metrics["finite"] = _finite_entry(synthetic)
    metrics["q_nonneg"] = _nonneg_entry(
        synthetic, "q", "比湿非负性（q ≥ 0）")
    metrics["cloud_nonneg"] = _cloud_nonneg_entry(synthetic)
    metrics["t_range"] = _range_entry(
        synthetic, "t",
        th["t_min"], th["t_max"], th["t_max"],
        "温度范围（t_min ≤ t ≤ t_max）")
    metrics["wind_range"] = _wind_entry(synthetic, th)
    metrics["saturation"] = _saturation_entry(synthetic, th)
    metrics["tcwv_range"] = _range_entry(
        synthetic, "tcwv",
        0.0, th["tcwv_max"], th["tcwv_max"],
        "整层水汽范围（0 ≤ tcwv ≤ tcwv_max）", absent_status="INFO")
    metrics["c_nonneg"] = _nonneg_entry(
        synthetic, "c", "追迹浓度非负性（c ≥ 0）",
        extra="c 的 NaN/Inf 已计入 finite 指标（c_finite），本项只判负值",
        absent_status="INFO")

    # ---- 软指标 -----------------------------------------------------------
    metrics["mass_budget"] = _mass_budget_entry(synthetic, physical_parameters, th)
    metrics["smoothness"] = _smoothness_entry(synthetic, real, th)
    metrics["equation_residual"] = _equation_residual_entry(
        synthetic, physical_parameters)

    # "数据集无 X / 不适用" 的 NOT_RUN 表示该约束对本数据集不成立（例如统计
    # 数据集没有 q/云水），它不是"跑过但失败"，也不应把整体拖成 NOT_RUN；
    # 真正因输入残缺而无法运行的 NOT_RUN 仍参与聚合（见 notes 分组）。
    _ABSENT_MARKERS = ("数据集无", "数据集不含", "不适用")
    applicable = {
        key: m["status"] not in ("NOT_RUN", "INFO")
        or not any(marker in str(m["detail"]) for marker in _ABSENT_MARKERS)
        for key, m in metrics.items()
    }
    status = aggregate_status(
        m["status"] for key, m in metrics.items() if applicable[key]
    )
    # Degenerate datasets that contain NONE of the canonical state variables
    # (t/q/u/v/c, e.g. a raw tcwv-only file) cannot be judged physically at
    # all: the module must say NOT_RUN rather than PASS on the leftovers.
    core_missing = not any(
        name in synthetic.data_vars for name in ("t", "q", "u", "v", "c")
    )
    if core_missing:
        status = "NOT_RUN"
        notes.insert(
            0,
            "核心状态变量（t/q/u/v/c）均缺失，无法判定物理合理性 → 整体 NOT_RUN"
            "（tcwv/tcwv-only 或外来数据集须先过 ingest/contract）",
        )

    # ---- notes ------------------------------------------------------------
    notes.append(
        "使用阈值: " + ", ".join(f"{k}={v:g}" for k, v in sorted(th.items()))
        + "（DEFAULT_THRESHOLDS + thresholds 覆盖）")
    soft_th = {
        "mass_drift_max": float(th.get("mass_drift_max", MASS_DRIFT_MAX)),
        "smoothness_factor_max": float(
            th.get("smoothness_factor_max", SMOOTHNESS_FACTOR_MAX)),
    }
    notes.append("软指标期望: " + ", ".join(
        f"{k}={v:g}" for k, v in sorted(soft_th.items())))
    notes.append(f"检查变量: {sorted(str(n) for n in synthetic.data_vars)}")
    if real is None:
        notes.append(
            "未提供 real：smoothness 只记录生成场翻转率（INFO），"
            "其余指标不依赖真实参考数据")
    not_run_skipped = [f"{k}: {m['detail']}" for k, m in metrics.items()
                       if m["status"] == "NOT_RUN" and not applicable[k]]
    if not_run_skipped:
        notes.append("NOT_RUN 指标（对本数据集不适用/变量缺失，不参与聚合）: "
                     + _short(not_run_skipped))
    not_run_real = [f"{k}: {m['detail']}" for k, m in metrics.items()
                    if m["status"] == "NOT_RUN" and applicable[k]]
    if not_run_real:
        notes.append("NOT_RUN 指标（尝试运行但输入残缺，需人工确认）: "
                     + _short(not_run_real))
    conditional = [k for k, m in metrics.items() if m["status"] == "CONDITIONAL_PASS"]
    if conditional:
        notes.append("CONDITIONAL_PASS 指标（软指标告警）: " + ", ".join(conditional))
    info_metrics = [k for k, m in metrics.items() if m["status"] == "INFO"]
    if info_metrics:
        notes.append("INFO 指标（仅记录数值，不参与通过判定）: "
                     + ", ".join(info_metrics))

    return {"metrics": metrics, "status": status, "notes": notes}
