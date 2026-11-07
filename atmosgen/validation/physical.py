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
  也可从数据集 ``atmosgen:contract`` 属性回退读取（便于 evaluate 直接传 synthetic）；
- schema 1.2 变量系统扩展（登记册见 ``atmosgen/variables.py``）新增的检查：
  ``sp_range``（地面气压 Pa）、``w_range``（垂直速度 Pa/s，正=上升）、``z_range``
  （位势 m² s⁻²）、``tp_nonneg``（累积降水 kg m⁻² ≡ mm，非率）、``rh_range``
  （相对湿度 %，云内过饱和只报告不裁剪）、``column_consistency``（列变量定义
  tclw≡cwp / tciw≡iwp / twp=cwp+iwp）、``tcwv_integral_consistency``（tcwv 与
  (1/g)∫q dp 的比较，软指标：统计/混合模式下 tcwv 是联合生成量，偏差只报告）、
  ``flux_consistency``（qflux_u=q·u、qflux_v=q·v）、``mfc_consistency``
  （mfc=−div(qflux_u_col,qflux_v_col)、mfc_sl=−div(qflux_u,qflux_v) 最低层）；
  这几项缺「可选变量」时记 INFO（该约束不适用），缺「必需输入」时记 NOT_RUN。
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
    # --- schema 1.2 变量系统扩展（与 atmosgen/config.py 的 THRESHOLD_LIMITS 对齐，
    #     用户可用 validation.thresholds_overrides 覆盖）------------------------ #
    "sp_min": 50000.0,           # Pa
    "sp_max": 110000.0,          # Pa
    "w_max": 10.0,               # Pa/s
    "z_min": -5000.0,            # m2/s2
    "z_max": 60000.0,            # m2/s2
    "rh_min": 0.0,               # %
    "rh_max": 200.0,             # %
    "rh_supersaturation_max": 120.0,  # % 超过此值记为过饱和（只报告，不裁剪）
    "tp_max": 1000.0,            # kg/m2
    "column_max": 200.0,         # kg/m2（列水汽路径通用上限）
    "consistency_rtol": 1.0e-6,  # 定义一致性的相对容差
    "consistency_atol": 1.0e-9,  # 定义一致性的绝对容差（kg/m2 等）
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
    masked_declared_rate: float | None = None
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
                    try:
                        masked_declared_rate = float(block.get("rejection_rate", 0.0))
                    except (TypeError, ValueError):
                        masked_declared_rate = 0.0
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
        detail += (
            f"; 筛检遮罩 NaN {masked} 个（契约声明 mask_applied，"
            f"声明拒绝率 {(masked_declared_rate or 0.0):.4f}）"
        )
        # A dataset must not be able to launder arbitrary missing values: the
        # masked fraction is bounded by the declared screening rejection rate
        # (with a small tolerance for multi-variable masking), capped at 50%.
        observed_rate = (masked / total) if total else 0.0
        declared = masked_declared_rate if masked_declared_rate is not None else 0.0
        allowed_rate = min(0.5, declared * 1.5 + 0.01)
        if observed_rate > allowed_rate:
            return _metric(
                bad,
                0.0,
                "max",
                "FAIL",
                detail
                + f"; 遮罩比例 {observed_rate:.2%} 超过声明拒绝率允许上限 {allowed_rate:.2%}"
                "（不允许用遮罩声明掩盖任意 NaN）",
            )
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
# schema 1.2 扩展：地面/垂直坐标/湿度/降水 硬约束 + 定义一致性检查
# --------------------------------------------------------------------------- #
#: 列变量（单位 kg m⁻²）。``tcwv`` 的专属范围由 ``tcwv_range`` 把关，这里只做
#: 「有限 + 非负 + ≤ column_max」的通用检查（负的列水汽路径没有物理含义 → 硬 FAIL）。
_COLUMN_VARS: tuple[str, ...] = ("tcwv", "tclw", "tciw", "cwp", "iwp", "twp")
#: 参与「定义一致性」两两比较的列变量。``tcwv`` 不在其中：它与 q 的垂直积分关系
#: 单列为 ``tcwv_integral_consistency``（统计模式下 tcwv 是联合生成量，语义不同）。
_COLUMN_PAIR_VARS: tuple[str, ...] = ("tclw", "tciw", "cwp", "iwp", "twp")
#: 登记册里水汽通量变量的期望单位（仅用于「报告」单位不一致，不据此 FAIL）。
_FLUX_UNITS_EXPECTED = "kg kg**-1 m s**-1"
#: 上述单位的宽松等价写法（归一化后比较）。
_FLUX_UNITS_SIGNATURES = frozenset({"kgkg1ms1", "kg/kg/m/s", "kgkg1m1s1"})
#: 标准重力加速度（m s⁻²），用于 (1/g)∫q dp → kg m⁻²。
_GRAVITY = 9.80665
#: 变量维度的规范顺序（比较前统一排列，避免形状相同但轴序不同造成的误判）。
_CANONICAL_DIMS: tuple[str, ...] = ("time", "level", "latitude", "longitude")
#: schema 1.2 新增的指标名（notes 里单独列一行，便于读者看清哪些跑了）。
_SCHEMA12_METRICS: tuple[str, ...] = (
    "sp_range", "w_range", "z_range", "tp_nonneg", "rh_range",
    "column_consistency", "tcwv_integral_consistency",
    "flux_consistency", "mfc_consistency",
)


def _consistency_tol(th: Mapping[str, float]) -> tuple[float, float]:
    """定义一致性检查的相对/绝对容差（相对容差随量级缩放，绝对容差兜底）。"""

    rtol = float(th.get("consistency_rtol", DEFAULT_THRESHOLDS["consistency_rtol"]))
    atol = float(th.get("consistency_atol", DEFAULT_THRESHOLDS["consistency_atol"]))
    return rtol, atol


def _units_signature(text: object) -> str:
    """把单位字符串归一化（去空格、去 ``**``、``-1`` → ``1``），用于宽松比较。"""

    s = str(text).strip().lower()
    s = s.replace("**", "").replace("^", "")
    s = s.replace("−", "-").replace("–", "-")
    s = s.replace(" ", "").replace("\u00b7", "")
    s = s.replace("-1", "1")
    return s


def _da_to_array(da) -> np.ndarray | None:
    """把 DataArray 转成 float numpy 数组（按规范维度顺序排列）；非数值返回 None。"""

    dims = [d for d in _CANONICAL_DIMS if d in da.dims]
    dims += [d for d in da.dims if d not in dims]
    arr = np.asarray(da.transpose(*dims).values)
    if not _numeric(arr):
        return None
    return arr.astype(float, copy=False)


def _ordered_array(ds, name: str) -> np.ndarray | None:
    """取 ``ds[name]`` 的 float numpy 数组（规范维度顺序）；非数值返回 None。"""

    if name not in ds.data_vars:
        return None
    return _da_to_array(ds[name])


def _compare_pair(a: np.ndarray, b: np.ndarray, rtol: float,
                  atol: float) -> dict[str, Any]:
    """逐元素比较两个同形状数组。

    判据：``|a-b| ≤ atol + rtol·max(|a|,|b|)``（相对容差随量级缩放，绝对容差兜底）。
    只在双方同时有限的元素上比较（非有限元素由 ``finite`` 指标负责）。
    """

    if a is None or b is None:
        return {"not_comparable": True}
    if a.shape != b.shape:
        return {"shape_mismatch": True, "a_shape": a.shape, "b_shape": b.shape}
    finite = np.isfinite(a) & np.isfinite(b)
    n_valid = int(np.count_nonzero(finite))
    if n_valid == 0:
        return {"n_valid": 0, "n_violation": 0, "max_abs": float("nan"),
                "max_rel": float("nan"), "max_tol": atol}
    av = a[finite]
    bv = b[finite]
    diff = np.abs(av - bv)
    scale = np.maximum(np.abs(av), np.abs(bv))
    tol = atol + rtol * scale
    n_violation = int(np.count_nonzero(diff > tol))
    max_abs = float(np.max(diff))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(scale > 0.0,
                       diff / np.where(scale > 0.0, scale, 1.0),
                       np.where(diff > 0.0, np.inf, 0.0))
    max_rel = float(np.max(rel))
    return {"n_valid": n_valid, "n_violation": n_violation, "max_abs": max_abs,
            "max_rel": max_rel, "max_tol": float(np.max(tol))}


def _trapz_integral(q_sorted: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    """沿第 0 轴做梯形积分（``q_sorted`` 已按气压升序排列，``p_pa`` 单位 Pa）。"""

    n = int(p_pa.size)
    if n < 2:
        return np.zeros(q_sorted.shape[1:], dtype=float)
    dp = np.diff(p_pa)
    weights = np.empty(n, dtype=float)
    weights[0] = dp[0] / 2.0
    weights[-1] = dp[-1] / 2.0
    weights[1:-1] = (dp[:-1] + dp[1:]) / 2.0
    return np.tensordot(weights, q_sorted, axes=(0, 0))


def _lowest_level_index(ds) -> tuple[int, str]:
    """返回（最低层索引, 说明）。气压坐标取气压最大处，否则按降序层序取最后一层。"""

    for lname in _LEVEL_NAMES:
        if lname in ds.coords or lname in ds.variables:
            raw = np.asarray(ds[lname].values, dtype=float).ravel()
            if raw.size == 0:
                continue
            _, level_hpa, _, note = _level_to_hpa(ds)
            if level_hpa is not None and level_hpa.size == raw.size:
                idx = int(np.argmax(level_hpa))
                return idx, (f"{lname} 的最低层（气压最大处 {level_hpa[idx]:g} hPa，"
                             f"{note}）")
            return int(raw.size - 1), f"{lname} 的最后一层（非气压坐标，按降序层序约定）"
    return 0, "无 level 维（单层场直接使用）"


def _z_entry(ds, th: Mapping[str, float]) -> dict[str, Any]:
    """位势 z 的合理范围（逐层水静力参考带，m² s⁻²）。

    位势量产纲 m² s⁻² 随气压层强变化（100 hPa 处约 1.7e5，1000 hPa 处约 0），
    单一全局区间会把合法的高层判 FAIL；因此本检查在存在气压层坐标时改为
    逐层参考带：z_ref(p) = R_d·T·ln(p_s/p)，T ∈ [200, 320] K，再放宽
    ±2000 m² s⁻² 以兼容基准面（MSL/地面）取法差异。无气压层坐标时退化为
    全局区间〔z_min, z_max〕。缺 z 变量 → INFO（该约束不适用）。
    """

    if "z" not in ds.data_vars:
        return _info(None, None, "数据集无 z 变量，无法检查位势范围"
                    "（该约束不适用，INFO 不是 PASS）")
    _, level_hpa, _, note = _level_to_hpa(ds)
    arr = np.asarray(ds["z"].values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return _not_run(th.get("z_max", 60000.0), "range", "z 全部为非有限值，无法检查范围")

    if level_hpa is not None and "level" in ds["z"].dims:
        # z 维顺序与 level 一致或已按时间在前，按 level 轴展开参考带
        z_values = np.moveaxis(arr, ds["z"].get_axis_num("level"), -1)  # [..., nlev]
        z_finite = np.isfinite(z_values)
        ln = np.log(np.asarray(level_hpa, dtype=float) ** 0)  # placeholder, replaced below
        # 参考带：z_ref(p) = R_d * T * ln(p_surface/p)，T ∈ [200, 320] K
        p_hpa = np.asarray(level_hpa, dtype=float)
        if (p_hpa > 0).all():
            p_s = float(np.max(p_hpa))
            ratio = np.log(p_s / p_hpa)  # >= 0，地面上最大
            z_ref_lo = 287.058 * 200.0 * ratio - 2000.0
            z_ref_hi = 287.058 * 320.0 * ratio + 2000.0
            n_below = int(np.count_nonzero(z_finite & (z_values < z_ref_lo)))
            n_above = int(np.count_nonzero(z_finite & (z_values > z_ref_hi)))
            mn = float(np.nanmin(z_values))
            mx = float(np.nanmax(z_values))
            detail = (
                f"逐层水静力参考带（R_d·T·ln(p_s/p)，T∈[200,320] K，±2000 m² s⁻² 放宽）, "
                f"{note}, n_z={int(z_finite.sum())}, z∈[{mn:.4g}, {mx:.4g}], "
                f"n_below={n_below}, n_above={n_above}"
            )
            status = "FAIL" if (n_below or n_above) else "PASS"
            value = mx if n_above else (mn if n_below else mx)
            return _metric(value, th.get("z_max", 60000.0), "range", status, detail)
    lo = float(th.get("z_min", -5000.0))
    hi = float(th.get("z_max", 60000.0))
    n_bad = int(np.count_nonzero(finite & ((arr < lo) | (arr > hi))))
    detail = (f"全局区间 [{lo:g}, {hi:g}] m² s⁻²（无可用气压层坐标，退化检查）, "
              f"n_bad={n_bad}, z∈[{float(np.nanmin(arr)):.4g}, {float(np.nanmax(arr)):.4g}]")
    return _metric(float(np.nanmax(arr)), hi, "range", "FAIL" if n_bad else "PASS", detail)

def _w_entry(ds, th: Mapping[str, float]) -> dict[str, Any]:
    """垂直速度 w 的范围硬约束：|w| ≤ w_max（Pa/s，正 = 上升运动）。"""

    limit = float(th.get("w_max", DEFAULT_THRESHOLDS["w_max"]))
    if "w" not in ds.data_vars:
        return _info(None, None,
                     "数据集无 w（垂直速度）变量，无法检查垂直速度范围"
                     "（该约束不适用，INFO 不是 PASS）")
    arr = np.asarray(ds["w"].values)
    if not _numeric(arr):
        return _not_run(limit, "max",
                        f"w 的 dtype={arr.dtype} 非数值，无法检查垂直速度范围")
    finite = np.isfinite(arr)
    n_valid = int(np.count_nonzero(finite))
    n_skipped = int(arr.size - n_valid)
    if n_valid == 0:
        return _not_run(limit, "max",
                        f"w 全部为非有限值（n={arr.size}），无法检查垂直速度范围")
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    absmax = max(abs(mn), abs(mx))
    n_over = int(np.count_nonzero(finite & ((arr > limit) | (arr < -limit))))
    detail = (f"limit=±{limit:g} Pa/s; n={n_valid}, min={mn:.4g} Pa/s, "
              f"max={mx:.4g} Pa/s, max|w|={absmax:.4g} Pa/s, n_over={n_over}, "
              f"排除非有限={n_skipped}; 单位 Pa/s（ERA5 约定：正值 = 上升运动，"
              "即气压随高度减小），不是 m/s")
    status = "FAIL" if n_over else "PASS"
    return _metric(absmax, limit, "max", status, detail)


def _tp_entry(ds, th: Mapping[str, float]) -> dict[str, Any]:
    """累积降水 tp：≥ 0 且 ≤ tp_max（kg m⁻²）。"""

    tp_max = float(th.get("tp_max", DEFAULT_THRESHOLDS["tp_max"]))
    if "tp" not in ds.data_vars:
        return _info(None, None,
                     "数据集无 tp（累积降水）变量，无法检查降水量范围"
                     "（该约束不适用，INFO 不是 PASS）")
    arr = np.asarray(ds["tp"].values)
    if not _numeric(arr):
        return _not_run(tp_max, "range",
                        f"tp 的 dtype={arr.dtype} 非数值，无法检查降水量范围")
    finite = np.isfinite(arr)
    n_valid = int(np.count_nonzero(finite))
    n_skipped = int(arr.size - n_valid)
    if n_valid == 0:
        return _not_run(tp_max, "range",
                        f"tp 全部为非有限值（n={arr.size}），无法检查降水量范围")
    n_below = int(np.count_nonzero(finite & (arr < 0.0)))
    n_above = int(np.count_nonzero(finite & (arr > tp_max)))
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    value = mx if n_above else (mn if n_below else mx)
    detail = (f"n={n_valid}, min={mn:.4g} kg/m2, max={mx:.4g} kg/m2, "
              f"bounds=[0, {tp_max:g}] kg/m2, n_below(0)={n_below}, "
              f"n_above({tp_max:g})={n_above}, 排除非有限={n_skipped}; "
              "tp 是「以该时戳为终点的时间区间」的累积量（kg/m2 ≡ mm 液态当量），"
              "不是瞬时率（瞬时率 = tp / dt）")
    status = "FAIL" if (n_below or n_above) else "PASS"
    return _metric(value, tp_max, "range", status, detail)


def _rh_entry(ds, th: Mapping[str, float]) -> dict[str, Any]:
    """相对湿度 rh/r 的范围检查（%）。

    - 越出 ``[rh_min, rh_max]`` → 物理上不可能 → FAIL；
    - 落在 ``(rh_supersaturation_max, rh_max]`` → 云内过饱和（物理上可能）→
      CONDITIONAL_PASS，按项目口径只报告、不裁剪、不改写；
    - ``rh`` 与 ``r`` 是同一量的两个别名，逐个检查后取最差状态。
    """

    lo = float(th.get("rh_min", DEFAULT_THRESHOLDS["rh_min"]))
    hi = float(th.get("rh_max", DEFAULT_THRESHOLDS["rh_max"]))
    ss_max = float(th.get("rh_supersaturation_max",
                           DEFAULT_THRESHOLDS["rh_supersaturation_max"]))
    present = [n for n in ("rh", "r") if n in ds.data_vars]
    if not present:
        return _info(None, None,
                     "数据集无 rh / r（相对湿度）变量，无法检查相对湿度范围"
                     "（该约束不适用，INFO 不是 PASS）")
    parts: list[str] = []
    n_below = 0
    n_above = 0
    n_super = 0
    n_valid = 0
    n_skipped = 0
    worst = float("nan")
    for name in present:
        arr = np.asarray(ds[name].values)
        if not _numeric(arr):
            parts.append(f"{name}: dtype={arr.dtype} 非数值（未验证）")
            continue
        finite = np.isfinite(arr)
        nv = int(np.count_nonzero(finite))
        n_valid += nv
        n_skipped += int(arr.size - nv)
        if nv == 0:
            parts.append(f"{name}: 全非有限（该别名未验证）")
            continue
        below = int(np.count_nonzero(finite & (arr < lo)))
        above = int(np.count_nonzero(finite & (arr > hi)))
        super_sat = int(np.count_nonzero(finite & (arr > ss_max) & (arr <= hi)))
        n_below += below
        n_above += above
        n_super += super_sat
        mx = float(np.nanmax(arr))
        worst = mx if np.isnan(worst) else max(worst, mx)
        parts.append(f"{name}: n={nv}, min={float(np.nanmin(arr)):.4g} %, "
                     f"max={mx:.4g} %, n_below={below}, n_above={above}, "
                     f"n_supersat={super_sat}")
    if n_valid == 0:
        return _not_run(hi, "range",
                        "rh / r 全部为非有限值，无法检查相对湿度范围: " + _short(parts))
    detail = (f"bounds=[{lo:g}, {hi:g}] %, supersat_limit={ss_max:g} %, "
              + "; ".join(parts)
              + f"; 合计 n_below={n_below}, n_above={n_above}, n_supersat={n_super}, "
              f"排除非有限={n_skipped}; rh 与 r 是同一量的别名（同公式、同数值），"
              f"在场别名 {present}")
    if n_below or n_above:
        status = "FAIL"
        detail += ("; 相对湿度越出容许区间在物理上不可能 —— 这是硬约束违例，"
                   "不是数值误差，也不是「真实极端」")
    elif n_super:
        status = "CONDITIONAL_PASS"
        detail += (f"; 有 {n_super} 个点落在 ({ss_max:g}, {hi:g}] % 的过饱和区"
                   "（云内过饱和物理上可能），按项目口径只报告、不裁剪、不改写")
    else:
        status = "PASS"
    return _metric(worst, hi, "range", status, detail)


def _column_consistency_entry(ds, th: Mapping[str, float]) -> dict[str, Any]:
    """列变量（kg m⁻²）的定义一致性与通用范围检查。

    定义关系（硬定义，逐点必须相等）：``tclw ≡ cwp``、``tciw ≡ iwp``、
    ``twp = cwp + iwp``。通用范围：在场的列变量必须有限、≥ 0、≤ ``column_max``。
    ``tcwv`` 与 q 的垂直积分关系单列为 ``tcwv_integral_consistency``。
    """

    rtol, atol = _consistency_tol(th)
    column_max = float(th.get("column_max", DEFAULT_THRESHOLDS["column_max"]))
    present = [n for n in _COLUMN_VARS if n in ds.data_vars]
    if not present:
        return _info(None, None,
                     "数据集不含 tcwv/tclw/tciw/cwp/iwp/twp 列变量，"
                     "列变量一致性约束不适用（INFO 不是 PASS）")

    # --- 通用范围（有限 / 非负 / 上限） ---
    gen_parts: list[str] = []
    n_neg = 0
    n_above = 0
    for name in present:
        arr = np.asarray(ds[name].values)
        if not _numeric(arr):
            gen_parts.append(f"{name}: dtype={arr.dtype} 非数值（未验证）")
            continue
        finite = np.isfinite(arr)
        nv = int(np.count_nonzero(finite))
        if nv == 0:
            gen_parts.append(f"{name}: 全非有限（未验证）")
            continue
        neg = int(np.count_nonzero(finite & (arr < 0.0)))
        above = int(np.count_nonzero(finite & (arr > column_max)))
        n_neg += neg
        n_above += above
        gen_parts.append(f"{name}: n={nv}, min={float(np.nanmin(arr)):.4g}, "
                         f"max={float(np.nanmax(arr)):.4g}, n_negative={neg}, "
                         f"n_above({column_max:g})={above}")
    gen_fail = bool(n_neg or n_above)

    # --- 定义一致性（两两比较） ---
    pairs: list[tuple[str, np.ndarray | None, np.ndarray | None, str]] = []
    for left, right, note in (("tclw", "cwp", "tclw ≡ cwp（同一量，两个名字）"),
                               ("tciw", "iwp", "tciw ≡ iwp（同一量，两个名字）")):
        if left in ds.data_vars and right in ds.data_vars:
            pairs.append((f"{left} vs {right}",
                          _ordered_array(ds, left), _ordered_array(ds, right), note))
    if "twp" in ds.data_vars and "cwp" in ds.data_vars and "iwp" in ds.data_vars:
        cwp_a = _ordered_array(ds, "cwp")
        iwp_a = _ordered_array(ds, "iwp")
        sum_arr = None
        if cwp_a is not None and iwp_a is not None and cwp_a.shape == iwp_a.shape:
            sum_arr = cwp_a + iwp_a
        pairs.append(("twp vs cwp+iwp", _ordered_array(ds, "twp"), sum_arr,
                      "twp = cwp + iwp（按定义）"))

    pair_parts: list[str] = []
    n_violation = 0
    compared = 0
    worst_pair = ""
    worst_abs = 0.0
    worst_rel = 0.0
    for label, a, b, note in pairs:
        if a is None or b is None:
            pair_parts.append(f"{label}: 非数值，未比较")
            continue
        res = _compare_pair(a, b, rtol, atol)
        if res.get("shape_mismatch"):
            pair_parts.append(f"{label}: 形状不一致 {res['a_shape']} vs "
                              f"{res['b_shape']}，未比较")
            continue
        compared += 1
        n_violation += int(res["n_violation"])
        pair_parts.append(f"{label}（{note}）: n={res['n_valid']}, "
                          f"max|Δ|={res['max_abs']:.4g} kg/m2, "
                          f"max相对偏差={res['max_rel']:.4g}, "
                          f"越限点={res['n_violation']}")
        if float(res["max_abs"]) > worst_abs:
            worst_abs = float(res["max_abs"])
            worst_rel = float(res["max_rel"])
            worst_pair = label

    detail = ("通用范围: " + _short(gen_parts)
              + f"; 容差 tol = atol + rtol·max(|a|,|b|) (atol={atol:g}, rtol={rtol:g}), "
              f"column_max={column_max:g} kg/m2")
    if pair_parts:
        detail += "; 定义一致性: " + _short(pair_parts)
    if gen_fail:
        status = "FAIL"
        detail += (f"; 列变量出现负值({n_neg})或超过 column_max({n_above}) —— "
                   "负的列水汽路径没有物理含义，这是硬约束违例，不是数值误差")
    elif n_violation:
        status = "FAIL"
        detail += (f"; 越限的一致性对: {worst_pair}（按定义为同一量，必须逐点相等），"
                   f"最差绝对偏差 {worst_abs:.4g} kg/m2（相对 {worst_rel:.4g}）")
    elif compared:
        status = "PASS"
    else:
        status = "INFO"
        detail += ("; 数据集含列变量但缺少可比较的一致性对"
                   "（tclw≡cwp / tciw≡iwp / twp=cwp+iwp 均缺一端），"
                   "定义一致性检查不适用——本项只完成通用范围检查（INFO 不是 PASS）")
    return _metric(n_violation, 0.0, "max", status, detail)


def _tcwv_integral_entry(ds, th: Mapping[str, float]) -> dict[str, Any]:
    """tcwv 与 ``(1/g)∫q dp`` 的一致性（软指标）。

    - 偏差在容差内 → PASS；
    - 偏差超出容差 → **CONDITIONAL_PASS**（不是 FAIL）：统计 / 混合生成模式下 tcwv
      是与 q 联合生成的列变量，并不是所生成 q 的垂直积分，两者出现偏差是预期内的；
      偏差只报告，不把统计上的相关关系说成物理因果关系。
    """

    rtol, atol = _consistency_tol(th)
    if "tcwv" not in ds.data_vars:
        return _info(None, None,
                     "数据集不含 tcwv（整层水汽）变量，tcwv 与 q 垂直积分的"
                     "一致性检查不适用（INFO 不是 PASS）")
    if "q" not in ds.data_vars:
        return _not_run(None, "max",
                        "tcwv 与垂直积分一致性未运行: 数据集含 tcwv 但缺少 q"
                        "（比湿，整层积分的输入），无法计算 (1/g)∫q dp")
    _, level_hpa, level_name, level_note = _level_to_hpa(ds)
    if level_hpa is None:
        # NOTE: the shared level note may contain the "absent" markers used by
        # the aggregator ("数据集无" / "不适用"). Here the level coordinate is a
        # genuinely missing INPUT, so the reason must not carry those markers:
        # the metric stays a real NOT_RUN and still participates in the verdict.
        safe_note = (level_note.replace("数据集无", "数据集缺少")
                             .replace("数据集不含", "数据集缺少")
                             .replace("不适用", "不成立"))
        return _not_run(None, "max",
                        f"tcwv 与垂直积分一致性未运行: level 坐标不可用（{safe_note}）")
    if level_hpa.size < 2:
        return _not_run(None, "max", (
            f"tcwv 与垂直积分一致性未运行: level 坐标只有 {level_hpa.size} 层，"
            "梯形积分退化（Δp=0，整层积分不可估），该检查对单层数据集不适用"
            "（physics 模式的单层数据集没有垂直范围）"))
    q_da = ds["q"]
    if level_name not in q_da.dims:
        return _not_run(None, "max",
                        f"tcwv 与垂直积分一致性未运行: q 的维度 "
                        f"{sorted(q_da.dims)} 不含 level 维（坐标名 {level_name!r}）")

    order = np.argsort(level_hpa)
    p_pa = level_hpa[order] * 100.0
    q_sorted = q_da.isel({level_name: order})
    tcwv_da = ds["tcwv"]
    has_time_q = "time" in q_sorted.dims
    has_time_t = "time" in tcwv_da.dims
    n_q_time = int(q_sorted.sizes["time"]) if has_time_q else 1
    n_t_time = int(tcwv_da.sizes["time"]) if has_time_t else 1

    integrals: list[np.ndarray] = []
    for it in range(n_q_time):
        frame = q_sorted.isel(time=it) if has_time_q else q_sorted
        if not {"latitude", "longitude"} <= set(frame.dims):
            return _not_run(None, "max",
                            f"tcwv 与垂直积分一致性未运行: q 缺少 "
                            f"latitude/longitude 维（{sorted(frame.dims)}）")
        arr = np.asarray(frame.transpose(level_name, "latitude",
                                          "longitude").values, dtype=float)
        integrals.append(_trapz_integral(arr, p_pa) / _GRAVITY)

    max_abs = 0.0
    max_rel = 0.0
    n_violation = 0
    n_valid = 0
    for it in range(n_t_time):
        frame = tcwv_da.isel(time=it) if has_time_t else tcwv_da
        if not {"latitude", "longitude"} <= set(frame.dims):
            return _not_run(None, "max",
                            f"tcwv 与垂直积分一致性未运行: tcwv 缺少 "
                            f"latitude/longitude 维（{sorted(frame.dims)}）")
        tv = np.asarray(frame.transpose("latitude", "longitude").values, dtype=float)
        expected = integrals[it if it < len(integrals) else 0]
        if tv.shape != expected.shape:
            return _not_run(None, "max",
                            f"tcwv 与垂直积分一致性未运行: tcwv 形状 {tv.shape} 与 "
                            f"积分结果 {expected.shape} 不一致，无法逐点比较")
        res = _compare_pair(tv, expected, rtol, atol)
        if res.get("shape_mismatch"):
            return _not_run(None, "max",
                            f"tcwv 与垂直积分一致性未运行: 形状不一致 "
                            f"{res['a_shape']} vs {res['b_shape']}")
        n_valid += int(res["n_valid"])
        n_violation += int(res["n_violation"])
        max_abs = max(max_abs, float(res["max_abs"]))
        max_rel = max(max_rel, float(res["max_rel"]))
    if n_valid == 0:
        return _not_run(None, "max",
                        "tcwv 与垂直积分一致性未运行: tcwv 与积分结果没有可比较的"
                        "有限元素")

    status = "PASS" if n_violation == 0 else "CONDITIONAL_PASS"
    detail = (f"tcwv vs (1/g)∫q dp: n={n_valid}, max|Δ|={max_abs:.6g} kg/m2 "
              f"（相对 {max_rel:.4g} = {max_rel * 100:.4g} %）, "
              f"越限点={n_violation}, 容差 tol = atol + rtol·max(|a|,|b|) "
              f"(atol={atol:g}, rtol={rtol:g}), 层数={level_hpa.size}, {level_note}")
    if status == "CONDITIONAL_PASS":
        detail += ("; 在统计 / 混合（statistical / hybrid）生成模式中 tcwv 是与 q "
                   "联合生成的列变量，并不是所生成 q 的垂直积分，因此两者出现偏差是"
                   "预期内的——不得把统计上的相关关系当作物理因果关系；"
                   "偏差只报告，不修正、不裁剪、不改写数据")
    return _metric(max_rel, rtol, "max", status, detail)


def _flux_consistency_entry(ds, th: Mapping[str, float]) -> dict[str, Any]:
    """水汽通量的定义一致性：``qflux_u = q·u``、``qflux_v = q·v``（硬定义）。"""

    rtol, atol = _consistency_tol(th)
    available: list[tuple[str, str]] = []
    missing: list[str] = []
    for flux, wind in (("qflux_u", "u"), ("qflux_v", "v")):
        lack = [n for n in (flux, "q", wind) if n not in ds.data_vars]
        if lack:
            missing.append(f"{flux} 缺 {'/'.join(lack)}")
        else:
            available.append((flux, wind))
    if not available:
        return _info(None, None,
                     "数据集不含 qflux_u/qflux_v 或缺少配对输入 q/u、q/v，"
                     "水汽通量定义一致性约束不适用（INFO 不是 PASS）")

    parts: list[str] = []
    n_violation = 0
    compared = 0
    worst_abs = 0.0
    worst_rel = 0.0
    for flux, wind in available:
        f_arr = _ordered_array(ds, flux)
        q_arr = _ordered_array(ds, "q")
        w_arr = _ordered_array(ds, wind)
        if f_arr is None or q_arr is None or w_arr is None:
            parts.append(f"{flux}: 非数值，未比较")
            continue
        if f_arr.shape != q_arr.shape or f_arr.shape != w_arr.shape:
            parts.append(f"{flux}: 形状不一致 {f_arr.shape} vs q{q_arr.shape}/"
                          f"{wind}{w_arr.shape}，未比较")
            continue
        res = _compare_pair(f_arr, q_arr * w_arr, rtol, atol)
        compared += 1
        n_violation += int(res["n_violation"])
        parts.append(f"{flux} vs q·{wind}: n={res['n_valid']}, "
                     f"max|Δ|={res['max_abs']:.4g}, max相对偏差={res['max_rel']:.4g}, "
                     f"越限点={res['n_violation']}")
        if float(res["max_abs"]) > worst_abs:
            worst_abs = float(res["max_abs"])
            worst_rel = float(res["max_rel"])

    unit_notes: list[str] = []
    for flux in ("qflux_u", "qflux_v"):
        if flux not in ds.data_vars:
            continue
        raw_units = ds[flux].attrs.get("units")
        if raw_units is None:
            unit_notes.append(f"{flux}: 无 units 属性（仅记录，不判 FAIL）")
        elif _units_signature(raw_units) in _FLUX_UNITS_SIGNATURES:
            unit_notes.append(f"{flux}: units={raw_units!r} 与登记册一致")
        else:
            unit_notes.append(f"{flux}: units={raw_units!r} 与登记册期望 "
                              f"{_FLUX_UNITS_EXPECTED!r} 不一致（仅记录，不判 FAIL）")

    detail = ("; ".join(parts)
              + f"; 容差 tol = atol + rtol·max(|a|,|b|) (atol={atol:g}, rtol={rtol:g})")
    if unit_notes:
        detail += "; 单位核对: " + _short(unit_notes)
    if missing:
        detail += f"; 未检查的通量分量: {_short(missing)}"
    if n_violation:
        status = "FAIL"
        detail += ("; 通量与 q·风 的定义关系被违反（按定义为恒等式）——"
                   f"最差绝对偏差 {worst_abs:.4g}（相对 {worst_rel:.4g}）")
    elif compared:
        status = "PASS"
    else:
        status = "NOT_RUN"
        detail += "; 没有任何一对可逐元素比较（形状/类型不兼容）"
    return _metric(n_violation, 0.0, "max", status, detail)


def _mfc_consistency_entry(ds, th: Mapping[str, float]) -> dict[str, Any]:
    """水汽通量辐散的定义一致性（硬定义）。

    - ``mfc = −div(qflux_u_col, qflux_v_col)``（整层辐散，kg m⁻² s⁻¹）；
    - ``mfc_sl = −div(qflux_u, qflux_v)``（最低层的单层诊断，kg kg⁻¹ s⁻¹）。

    ``horizontal_divergence`` 来自 ``atmosgen.derived``（惰性导入）；导入失败时
    本项记 NOT_RUN 并写明原因，不伪造 PASS。
    """

    rtol, atol = _consistency_tol(th)
    has_mfc = "mfc" in ds.data_vars
    has_sl = "mfc_sl" in ds.data_vars
    if not has_mfc and not has_sl:
        return _info(None, None,
                     "数据集不含 mfc / mfc_sl（水汽通量辐散）变量，"
                     "辐散定义一致性约束不适用（INFO 不是 PASS）")
    try:
        from ..derived import horizontal_divergence
    except Exception as exc:  # noqa: BLE001 - 依赖缺失时降级为 NOT_RUN
        return _not_run(None, "max",
                        f"水汽通量辐散一致性未运行: 依赖 atmosgen.derived."
                        f"horizontal_divergence 不可用"
                        f"（{type(exc).__name__}: {exc}）")

    parts: list[str] = []
    states: list[str] = []
    n_violation = 0
    compared = 0
    worst_abs = 0.0
    worst_rel = 0.0

    if has_mfc:
        lack = [n for n in ("qflux_u_col", "qflux_v_col") if n not in ds.data_vars]
        if lack:
            parts.append("mfc: 数据集存在 mfc 变量，但缺少 " + "/".join(lack)
                         + "（mfc 的定义输入），未比较")
            states.append("NOT_RUN")
        else:
            try:
                expected = -horizontal_divergence(ds["qflux_u_col"],
                                                  ds["qflux_v_col"])
            except Exception as exc:  # noqa: BLE001 - 依赖侧失败降级为 NOT_RUN
                parts.append(f"mfc: 调用 atmosgen.derived.horizontal_divergence "
                             f"失败（{type(exc).__name__}: {exc}），未比较")
                states.append("NOT_RUN")
                expected = None
            if expected is not None:
                exp_arr = _da_to_array(expected)
                if exp_arr is None:
                    parts.append("mfc: 散度结果非数值，未比较")
                    states.append("NOT_RUN")
                    expected = None
            if expected is not None:
                res = _compare_pair(_ordered_array(ds, "mfc"),
                                    _da_to_array(expected), rtol, atol)
                if res.get("not_comparable"):
                    parts.append("mfc: 非数值，未比较")
                    states.append("NOT_RUN")
                elif res.get("shape_mismatch"):
                    parts.append(f"mfc: 形状不一致 {res['a_shape']} vs "
                                 f"{res['b_shape']}，未比较")
                    states.append("NOT_RUN")
                else:
                    compared += 1
                    n_violation += int(res["n_violation"])
                    worst_abs = max(worst_abs, float(res["max_abs"]))
                    worst_rel = max(worst_rel, float(res["max_rel"]))
                    parts.append(f"mfc vs −div(qflux_u_col, qflux_v_col): "
                                 f"n={res['n_valid']}, max|Δ|={res['max_abs']:.4g}, "
                                 f"max相对偏差={res['max_rel']:.4g}, "
                                 f"越限点={res['n_violation']}")
                    states.append("FAIL" if res["n_violation"] else "PASS")

    if has_sl:
        lack = [n for n in ("qflux_u", "qflux_v") if n not in ds.data_vars]
        if lack:
            parts.append("mfc_sl: 数据集存在 mfc_sl 变量，但缺少 " + "/".join(lack)
                         + "（单层辐散的定义输入），未比较")
            states.append("NOT_RUN")
        else:
            idx, level_note = _lowest_level_index(ds)
            qu = ds["qflux_u"]
            qv = ds["qflux_v"]
            for lname in _LEVEL_NAMES:
                if lname in qu.dims:
                    qu = qu.isel({lname: idx})
                    qv = qv.isel({lname: idx})
                    break
            try:
                expected = -horizontal_divergence(qu, qv)
            except Exception as exc:  # noqa: BLE001 - 依赖侧失败降级为 NOT_RUN
                parts.append(f"mfc_sl: 调用 atmosgen.derived.horizontal_divergence "
                             f"失败（{type(exc).__name__}: {exc}），未比较")
                states.append("NOT_RUN")
                expected = None
            if expected is not None:
                exp_arr = _da_to_array(expected)
                if exp_arr is None:
                    parts.append("mfc_sl: 散度结果非数值，未比较")
                    states.append("NOT_RUN")
                    expected = None
            if expected is not None:
                res = _compare_pair(_ordered_array(ds, "mfc_sl"),
                                    _da_to_array(expected), rtol, atol)
                if res.get("not_comparable"):
                    parts.append("mfc_sl: 非数值，未比较")
                    states.append("NOT_RUN")
                elif res.get("shape_mismatch"):
                    parts.append(f"mfc_sl: 形状不一致 {res['a_shape']} vs "
                                 f"{res['b_shape']}，未比较")
                    states.append("NOT_RUN")
                else:
                    compared += 1
                    n_violation += int(res["n_violation"])
                    worst_abs = max(worst_abs, float(res["max_abs"]))
                    worst_rel = max(worst_rel, float(res["max_rel"]))
                    parts.append(f"mfc_sl vs −div(qflux_u, qflux_v)（{level_note}）: "
                                 f"n={res['n_valid']}, max|Δ|={res['max_abs']:.4g}, "
                                 f"max相对偏差={res['max_rel']:.4g}, "
                                 f"越限点={res['n_violation']}")
                    states.append("FAIL" if res["n_violation"] else "PASS")

    detail = ("; ".join(parts)
              + f"; 容差 tol = atol + rtol·max(|a|,|b|) (atol={atol:g}, rtol={rtol:g})")
    if "FAIL" in states:
        status = "FAIL"
        detail += ("; 辐散与通量的散度定义被违反（按定义为恒等式）——"
                   f"最差绝对偏差 {worst_abs:.4g}（相对 {worst_rel:.4g}）")
    elif "NOT_RUN" in states:
        status = "NOT_RUN"
        detail += "; 有分量因缺输入未比较（真实 NOT_RUN，不是 PASS）"
    else:
        status = "PASS"
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
    # NOTE: the residual is implemented for a single-layer 2-D wind field and a
    # single advected scalar `c` (Mode 2). Multi-level datasets (hybrid/statistical
    # with 4-D winds) are reported NOT_RUN rather than approximated silently —
    # the per-level residual form is a documented Phase 4+ limitation.
    advected_name = "c"
    missing = [n for n in ("c",) if n not in ds.data_vars]
    # multifield 耦合场生成器不产出独立示踪变量 c（所有场由平流-扩散模式场
    # 驱动构造），扩散/平流结构已由该驱动承担，方程残差对该数据集不适用，
    # 记 INFO（不参与聚合，不把 NOT_RUN 伪装成 PASS）。
    if missing and str(_contract(ds).get("model_name") or "").startswith(
            "physics.multifield"):
        return _info(None, None, (
            "方程残差不适用: multifield 耦合场数据集无独立示踪变量 c"
            "（平流-扩散结构由模式场驱动承担，其 CFL/扩散数/质量漂移诊断已"
            "记录在契约 diagnostics 中）；无独立示踪场，本指标不适用。"
            "（INFO 不是 PASS）"))
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

    # ---- schema 1.2 变量系统扩展：地面/垂直坐标/湿度/降水 + 定义一致性 --------
    metrics["sp_range"] = _range_entry(
        synthetic, "sp",
        th["sp_min"], th["sp_max"], th["sp_max"],
        "地面气压范围（sp_min ≤ sp ≤ sp_max，Pa）", absent_status="INFO")
    metrics["w_range"] = _w_entry(synthetic, th)
    metrics["z_range"] = _z_entry(synthetic, th)
    metrics["tp_nonneg"] = _tp_entry(synthetic, th)
    metrics["rh_range"] = _rh_entry(synthetic, th)
    metrics["column_consistency"] = _column_consistency_entry(synthetic, th)
    metrics["tcwv_integral_consistency"] = _tcwv_integral_entry(synthetic, th)
    metrics["flux_consistency"] = _flux_consistency_entry(synthetic, th)
    metrics["mfc_consistency"] = _mfc_consistency_entry(synthetic, th)

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
    notes.append(
        "schema 1.2 扩展指标: "
        + ", ".join(f"{key}={metrics[key]['status']}" for key in _SCHEMA12_METRICS))
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
