"""atmosgen.validation.physical 的单元测试（物理硬约束 + 软指标）。

自建合成 ``xarray.Dataset``（不依赖 conftest 的私有 fixture）：

- ``_legal_dataset()``：Mode 2 物理模式风格的合法数据集——t 255–265 K、
  q = 5e-4、u/v < 50 m/s、tcwv = 20、c ≥ 0、level = 850 hPa（hPa）、
  ``physical_parameters`` 带 ``boundary_conditions.kind="periodic"`` 与 ``kappa``
  → 全部硬约束 PASS、mass_budget PASS、equation_residual INFO，整体 PASS；
- 其余用例在合法数据集上做单点/单变量破坏，验证对应指标的 FAIL / NOT_RUN /
  CONDITIONAL_PASS 与 detail 可读性。
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import gaussian_filter

import atmosgen.validation.physical as physical_module
from atmosgen.validation.physical import (
    DEFAULT_THRESHOLDS,
    run_physical_validation,
)

NT = 2
NLAT = 4
NLON = 5
TIMES = np.array(["2020-01-01T00", "2020-01-01T06"], dtype="datetime64[ns]")
LEVEL = np.array([850.0])
LAT = np.linspace(20.0, 29.0, NLAT)
LON = np.linspace(100.0, 104.0, NLON)

DIMS4 = ("time", "level", "latitude", "longitude")
DIMS3 = ("time", "latitude", "longitude")

#: Hard constraints that existed before the schema-1.2 variable extension.
#: They are kept separate from the new metrics so that the legacy regression
#: test below keeps proving that the original behaviour did not change.
HARD_KEYS = (
    "finite", "q_nonneg", "cloud_nonneg", "t_range", "wind_range",
    "saturation", "tcwv_range", "c_nonneg",
)
#: Hard constraints added by the schema-1.2 variable-system extension.
NEW_HARD_KEYS = (
    "sp_range", "w_range", "z_range", "tp_nonneg", "rh_range",
    "column_consistency", "tcwv_integral_consistency",
    "flux_consistency", "mfc_consistency",
)
SOFT_KEYS = ("mass_budget", "smoothness", "equation_residual")
ALL_KEYS = HARD_KEYS + NEW_HARD_KEYS + SOFT_KEYS
STATUSES = {"PASS", "FAIL", "CONDITIONAL_PASS", "NOT_RUN", "INFO"}


# --------------------------------------------------------------------------- #
# 数据集构造
# --------------------------------------------------------------------------- #
def _spatial_pattern() -> np.ndarray:
    lat2, lon2 = np.meshgrid(LAT, LON, indexing="ij")
    return np.exp(-(((lat2 - 24.5) ** 2 + (lon2 - 102.0) ** 2) / (2.0 * 1.2 ** 2)))


def _legal_dataset(*, with_c: bool = True, with_cloud: bool = False) -> xr.Dataset:
    """Mode 2 物理模式风格的合法数据集（所有硬约束应 PASS）。"""

    pat = _spatial_pattern()
    t = np.broadcast_to(255.0 + 10.0 * pat, (NT, 1, NLAT, NLON)).copy()
    q = np.full((NT, 1, NLAT, NLON), 5e-4)
    u = np.broadcast_to(8.0 + 1.0 * pat, (NT, 1, NLAT, NLON)).copy()
    v = np.broadcast_to(2.0 - 0.5 * pat, (NT, 1, NLAT, NLON)).copy()
    tcwv = np.broadcast_to(20.0 + 2.0 * pat, (NT, NLAT, NLON)).copy()
    c = np.broadcast_to(1e-6 * (1.0 + pat), (NT, NLAT, NLON)).copy()  # 时间上恒定

    data_vars: dict[str, tuple] = {
        "t": (DIMS4, t),
        "q": (DIMS4, q),
        "u": (DIMS4, u),
        "v": (DIMS4, v),
        "tcwv": (DIMS3, tcwv),
    }
    if with_c:
        data_vars["c"] = (DIMS3, c)
    if with_cloud:
        data_vars["clwc"] = (DIMS4, np.full((NT, 1, NLAT, NLON), 1e-6))
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": TIMES, "level": LEVEL,
                "latitude": LAT, "longitude": LON},
    )
    ds["level"].attrs["units"] = "hPa"
    ds["latitude"].attrs["units"] = "degrees_north"
    ds["longitude"].attrs["units"] = "degrees_east"
    return ds


def _physical_parameters(*, periodic: bool = True,
                         kappa: float | None = 1.0e5) -> dict:
    pp: dict = {
        "boundary_conditions": {"kind": "periodic" if periodic else "open"},
    }
    if kappa is not None:
        pp["kappa"] = kappa
    return pp


def _run(ds, *, thresholds=None, pp=None, real=None) -> dict:
    return run_physical_validation(
        ds, thresholds=thresholds or {}, physical_parameters=pp, real=real)


def _saturation_dataset(q_value: float, *, level_value: float = 850.0,
                        level_units: str = "hPa") -> xr.Dataset:
    t = np.full((NT, 1, NLAT, NLON), 290.0)  # 290 K @ 850 hPa → q_sat ≈ 0.0142
    q = np.full((NT, 1, NLAT, NLON), q_value)
    ds = xr.Dataset(
        data_vars={"t": (DIMS4, t), "q": (DIMS4, q)},
        coords={"time": TIMES, "level": np.array([level_value]),
                "latitude": LAT, "longitude": LON},
    )
    ds["level"].attrs["units"] = level_units
    return ds


# --------------------------------------------------------------------------- #
# 1) 合法数据集：全部硬约束 PASS、整体 PASS
# --------------------------------------------------------------------------- #
def test_legal_dataset_all_hard_pass_and_overall_pass():
    out = _run(_legal_dataset(with_cloud=True), pp=_physical_parameters())
    assert set(out) == {"metrics", "status", "notes"}
    assert out["status"] == "PASS", {k: m["status"] for k, m in out["metrics"].items()}
    for key in HARD_KEYS:
        assert out["metrics"][key]["status"] == "PASS", (key, out["metrics"][key])
    # 软指标：周期边界质量守恒 PASS；无 real 的平滑性/残差为 INFO
    assert out["metrics"]["mass_budget"]["status"] == "PASS"
    assert out["metrics"]["mass_budget"]["value"] == pytest.approx(0.0)
    assert out["metrics"]["smoothness"]["status"] == "INFO"
    assert out["metrics"]["equation_residual"]["status"] == "INFO"
    # 饱和约束通过时 value 是 0 个超限点
    assert out["metrics"]["saturation"]["value"] == 0.0
    assert out["metrics"]["t_range"]["threshold"] == DEFAULT_THRESHOLDS["t_max"]


def test_legal_dataset_without_physics_vars_is_overall_pass():
    """t/q/u/v/tcwv 齐备、无 c/云水/物理参数的合法数据集不得被可选约束误判成 NOT_RUN。"""

    out = _run(_legal_dataset(with_c=False))
    for key in ("finite", "q_nonneg", "t_range", "wind_range", "saturation",
                "tcwv_range"):
        assert out["metrics"][key]["status"] == "PASS", (key, out["metrics"][key])
    for key in ("cloud_nonneg", "c_nonneg", "mass_budget", "smoothness",
                "equation_residual"):
        assert out["metrics"][key]["status"] == "INFO", (key, out["metrics"][key])
    assert out["status"] == "PASS"


# --------------------------------------------------------------------------- #
# 2) 负 q → FAIL
# --------------------------------------------------------------------------- #
def test_negative_q_fails():
    ds = _legal_dataset()
    ds["q"].values[0, 0, 1, 1] = -1e-4
    out = _run(ds, pp=_physical_parameters())
    m = out["metrics"]["q_nonneg"]
    assert m["status"] == "FAIL"
    assert m["value"] == 1.0
    assert m["comparison"] == "max"
    assert "n_negative=1" in m["detail"]
    assert out["status"] == "FAIL"


# --------------------------------------------------------------------------- #
# 3) 注入 NaN → finite FAIL
# --------------------------------------------------------------------------- #
def test_nan_injection_fails_finite():
    ds = _legal_dataset()
    ds["t"].values[0, 0, 1, 1] = np.nan
    out = _run(ds, pp=_physical_parameters())
    m = out["metrics"]["finite"]
    assert m["status"] == "FAIL"
    assert m["value"] >= 1.0
    assert "t=" in m["detail"]
    # 其它指标按变量压掉 NaN 后仍可判定（不因 NaN 额外 FAIL）
    assert out["metrics"]["t_range"]["status"] == "PASS"
    assert "排除非有限=1" in out["metrics"]["t_range"]["detail"]
    assert out["status"] == "FAIL"


# --------------------------------------------------------------------------- #
# 4) t=400 K / u=200 m/s → 温度范围、风速范围 FAIL
# --------------------------------------------------------------------------- #
def test_temperature_out_of_range_fails():
    ds = _legal_dataset()
    ds["t"].values[...] = 400.0
    out = _run(ds, pp=_physical_parameters())
    m = out["metrics"]["t_range"]
    assert m["status"] == "FAIL"
    assert m["value"] == 400.0
    assert m["threshold"] == 340.0
    assert "n_above=" in m["detail"]
    assert out["status"] == "FAIL"


def test_wind_out_of_range_fails():
    ds = _legal_dataset()
    ds["u"].values[...] = 200.0
    out = _run(ds, pp=_physical_parameters())
    m = out["metrics"]["wind_range"]
    assert m["status"] == "FAIL"
    assert m["value"] == pytest.approx(200.0)
    assert m["threshold"] == 150.0
    assert "limit=150" in m["detail"]
    assert out["status"] == "FAIL"


# --------------------------------------------------------------------------- #
# 5) 饱和违背 / 合法比湿 / Pa 单位换算
# --------------------------------------------------------------------------- #
def test_saturation_violation_fails_and_legal_q_passes():
    # t=290 K, p=850 hPa → q_sat≈0.0142；q=0.2 远超 1.10×q_sat
    bad = _saturation_dataset(0.2)
    out = _run(bad)
    m = out["metrics"]["saturation"]
    assert m["status"] == "FAIL"
    assert m["value"] == float(NT * NLAT * NLON)  # 全部格点超限
    assert "factor=1.1" in m["detail"]
    assert "worst q/q_sat" in m["detail"]
    assert m["threshold"] == 0.0

    good = _saturation_dataset(0.005)  # 0.005 < 1.10×0.0142
    out2 = _run(good)
    m2 = out2["metrics"]["saturation"]
    assert m2["status"] == "PASS"
    assert m2["value"] == 0.0


def test_saturation_pa_level_units_converted():
    # level 单位为 Pa 时必须换算到 hPa（85000 Pa → 850 hPa），否则 q_sat 会算错
    bad = _saturation_dataset(0.2, level_value=85000.0, level_units="Pa")
    out = _run(bad)
    m = out["metrics"]["saturation"]
    assert m["status"] == "FAIL"
    assert "Pa" in m["detail"]
    assert "换算因子=0.01" in m["detail"]


# --------------------------------------------------------------------------- #
# 6) 缺 level / tcwv-only：NOT_RUN 且有原因，整体不误判 PASS
# --------------------------------------------------------------------------- #
def test_saturation_not_run_without_level():
    t = np.full((NT, NLAT, NLON), 290.0)
    q = np.full((NT, NLAT, NLON), 0.005)
    ds = xr.Dataset(
        data_vars={"t": (DIMS3, t), "q": (DIMS3, q)},
        coords={"time": TIMES, "latitude": LAT, "longitude": LON},
    )
    out = _run(ds)
    m = out["metrics"]["saturation"]
    assert m["status"] == "NOT_RUN"
    assert "level" in m["detail"]
    assert m["detail"]  # 原因非空


def test_saturation_not_run_unknown_level_units():
    ds = _saturation_dataset(0.005)
    del ds["level"].attrs["units"]
    out = _run(ds)
    m = out["metrics"]["saturation"]
    assert m["status"] == "NOT_RUN"
    assert "单位" in m["detail"]


def test_tcwv_only_dataset_not_false_pass():
    tcwv = np.full((NT, NLAT, NLON), 20.0)
    ds = xr.Dataset(
        data_vars={"tcwv": (DIMS3, tcwv)},
        coords={"time": TIMES, "latitude": LAT, "longitude": LON},
    )
    out = _run(ds)  # 未提供 physical_parameters
    for key in ("q_nonneg", "t_range", "wind_range", "saturation"):
        m = out["metrics"][key]
        assert m["status"] == "NOT_RUN", (key, m)
        assert m["detail"], key
    # 约束本身不适用（无云水变量 / 无 c / 非物理模式）→ INFO，不是 NOT_RUN
    for key in ("cloud_nonneg", "c_nonneg", "equation_residual"):
        assert out["metrics"][key]["status"] == "INFO", (key, out["metrics"][key])
    assert out["status"] == "NOT_RUN"
    assert out["status"] != "PASS"
    assert any("NOT_RUN 指标" in n for n in out["notes"])
    # tcwv 本身有限且在范围内 → 该项硬约束照常 PASS
    assert out["metrics"]["finite"]["status"] == "PASS"
    assert out["metrics"]["tcwv_range"]["status"] == "PASS"


# --------------------------------------------------------------------------- #
# 7) c 负值 → c_nonneg FAIL；c 的 Inf 由 finite 抓
# --------------------------------------------------------------------------- #
def test_c_negative_and_nonfinite():
    ds = _legal_dataset()
    ds["c"].values[1, 0, 0] = -1e-6
    out = _run(ds, pp=_physical_parameters())
    m = out["metrics"]["c_nonneg"]
    assert m["status"] == "FAIL"
    assert m["value"] == 1.0
    assert "c_finite" in m["detail"]
    assert out["status"] == "FAIL"

    ds2 = _legal_dataset()
    ds2["c"].values[0, 0, 0] = np.inf
    out2 = _run(ds2, pp=_physical_parameters())
    assert out2["metrics"]["finite"]["status"] == "FAIL"
    assert out2["metrics"]["c_nonneg"]["status"] == "PASS"  # Inf 由 finite 指标抓
    assert out2["status"] == "FAIL"


# --------------------------------------------------------------------------- #
# 8) mass_budget：periodic 守恒 PASS / 漂移 CONDITIONAL_PASS / 非周期 INFO
# --------------------------------------------------------------------------- #
def test_mass_budget_periodic_conserving_passes():
    out = _run(_legal_dataset(), pp=_physical_parameters(periodic=True))
    m = out["metrics"]["mass_budget"]
    assert m["status"] == "PASS"
    assert m["value"] == pytest.approx(0.0)
    assert m["threshold"] == 0.05
    assert "var=c" in m["detail"]
    assert "边界=periodic" in m["detail"]


def test_mass_budget_drift_is_conditional_pass():
    ds = _legal_dataset()
    ds["c"].values[1] *= 1.2  # 总量漂移 20% > 5% 期望
    out = _run(ds, pp=_physical_parameters(periodic=True))
    m = out["metrics"]["mass_budget"]
    assert m["status"] == "CONDITIONAL_PASS"
    assert m["value"] > 0.15
    assert out["status"] == "CONDITIONAL_PASS"


def test_mass_budget_nonperiodic_or_missing_params_is_info():
    out = _run(_legal_dataset(), pp=_physical_parameters(periodic=False))
    m = out["metrics"]["mass_budget"]
    assert m["status"] == "INFO"
    assert "非 periodic" in m["detail"]

    out2 = _run(_legal_dataset())  # 无 physical_parameters
    m2 = out2["metrics"]["mass_budget"]
    assert m2["status"] == "INFO"
    assert "未提供" in m2["detail"]


def test_contract_fallback_for_boundary_and_kappa():
    ds = _legal_dataset(with_c=False)
    ds.attrs["atmosgen:contract"] = json.dumps({
        "boundary_conditions": {"kind": "periodic"},
        "physical_parameters": {"kappa": 1.0e5},
        "coordinates": {"level": {"type": "pressure_level", "units": "hPa"}},
    })
    out = _run(ds)  # physical_parameters=None，回退读契约
    m = out["metrics"]["mass_budget"]
    assert m["status"] == "PASS"
    assert "来源 contract" in m["detail"]
    assert "var=tcwv" in m["detail"]  # 无 c → 用整层水汽柱
    # 契约声明了 physical_parameters → 物理模式语境：缺 c 属未完成验证 → NOT_RUN
    assert out["metrics"]["equation_residual"]["status"] == "NOT_RUN"
    assert "c" in out["metrics"]["equation_residual"]["detail"]
    assert out["metrics"]["c_nonneg"]["status"] == "INFO"


# --------------------------------------------------------------------------- #
# 9) smoothness：白噪声 vs 平滑 real → CONDITIONAL_PASS；无 real → INFO
# --------------------------------------------------------------------------- #
def test_smoothness_white_noise_vs_smooth_real():
    rng = np.random.default_rng(0)
    gen = _legal_dataset()
    gen["c"].values = rng.standard_normal((NT, NLAT, NLON))  # 白噪声：翻转率≈0.5
    real = _legal_dataset()
    smooth = np.stack([gaussian_filter(rng.standard_normal((NLAT, NLON)), sigma=1.2)
                       for _ in range(NT)])
    real["c"].values = smooth  # 平滑场：翻转率≈0

    out = _run(gen, pp=_physical_parameters(), real=real)
    m = out["metrics"]["smoothness"]
    assert m["status"] == "CONDITIONAL_PASS"
    assert m["value"] > 2.0
    assert "gen_flip_rate" in m["detail"]
    assert "real_flip_rate" in m["detail"]
    assert "ratio=" in m["detail"]


def test_smoothness_without_real_is_info():
    out = _run(_legal_dataset(), pp=_physical_parameters())
    m = out["metrics"]["smoothness"]
    assert m["status"] == "INFO"
    assert "gen_flip_rate" in m["detail"]
    assert "未提供 real" in m["detail"]


# --------------------------------------------------------------------------- #
# 10) equation_residual：齐备 → INFO；缺 kappa / 缺 c → NOT_RUN + 原因
# --------------------------------------------------------------------------- #
def test_equation_residual_info_with_parameters():
    out = _run(_legal_dataset(), pp=_physical_parameters())
    m = out["metrics"]["equation_residual"]
    assert m["status"] == "INFO"
    assert np.isfinite(m["value"])
    assert m["value"] >= 0.0
    for token in ("rms=", "adv_rms=", "diffusion_rms=", "kappa=", "单位=c"):
        assert token in m["detail"], token


def test_equation_residual_not_run_reasons():
    out = _run(_legal_dataset(), pp={"boundary_conditions": {"kind": "periodic"}})
    m = out["metrics"]["equation_residual"]
    assert m["status"] == "NOT_RUN"
    assert "kappa" in m["detail"]

    out2 = _run(_legal_dataset(with_c=False), pp=_physical_parameters())
    m2 = out2["metrics"]["equation_residual"]
    assert m2["status"] == "NOT_RUN"
    assert "c" in m2["detail"]


# --------------------------------------------------------------------------- #
# 11) 阈值覆盖
# --------------------------------------------------------------------------- #
def test_threshold_overrides_are_applied():
    ds = _legal_dataset()
    ds["u"].values[...] = 120.0
    assert _run(ds)["metrics"]["wind_range"]["status"] == "PASS"  # 默认 150
    out = _run(ds, thresholds={"wind_max": 100.0})
    assert out["metrics"]["wind_range"]["status"] == "FAIL"
    assert out["metrics"]["wind_range"]["threshold"] == 100.0

    tds = _legal_dataset()
    tds["t"].values[...] = 250.0  # 默认 [180,340] → PASS
    assert _run(tds)["metrics"]["t_range"]["status"] == "PASS"
    out2 = _run(tds, thresholds={"t_min": 260.0})
    assert out2["metrics"]["t_range"]["status"] == "FAIL"


# --------------------------------------------------------------------------- #
# 12) cloud / tcwv 硬约束
# --------------------------------------------------------------------------- #
def test_cloud_and_tcwv_hard_constraints():
    out = _run(_legal_dataset(with_cloud=True), pp=_physical_parameters())
    assert out["metrics"]["cloud_nonneg"]["status"] == "PASS"
    # 不含云水变量 → 约束不适用（INFO），不是 NOT_RUN
    absent = _run(_legal_dataset(), pp=_physical_parameters())
    assert absent["metrics"]["cloud_nonneg"]["status"] == "INFO"

    ds = _legal_dataset()
    clwc = np.full((NT, 1, NLAT, NLON), 1e-6)
    clwc[0, 0, 0, 0] = -1e-6
    ds["clwc"] = (DIMS4, clwc)
    out2 = _run(ds, pp=_physical_parameters())
    m = out2["metrics"]["cloud_nonneg"]
    assert m["status"] == "FAIL"
    assert m["value"] == 1.0
    assert "clwc" in m["detail"]

    ds2 = _legal_dataset()
    ds2["tcwv"].values[0, 0, 0] = -1.0
    assert _run(ds2)["metrics"]["tcwv_range"]["status"] == "FAIL"

    ds3 = _legal_dataset()
    ds3["tcwv"].values[0, 0, 0] = 500.0
    m3 = _run(ds3)["metrics"]["tcwv_range"]
    assert m3["status"] == "FAIL"
    assert m3["threshold"] == 120.0


# --------------------------------------------------------------------------- #
# 13) 结构 / 空数据集
# --------------------------------------------------------------------------- #
def test_output_structure_and_metric_fields():
    out = _run(_legal_dataset(), pp=_physical_parameters())
    assert set(out) == {"metrics", "status", "notes"}
    assert out["status"] in STATUSES
    assert isinstance(out["notes"], list) and out["notes"]
    assert set(out["metrics"]) == set(ALL_KEYS)
    for key, m in out["metrics"].items():
        assert set(m) == {"value", "threshold", "comparison", "status", "detail"}
        assert isinstance(m["value"], float), key
        assert m["threshold"] is None or isinstance(m["threshold"], float), key
        assert m["comparison"] in {"max", "min", "range", "info"}, key
        assert m["status"] in STATUSES, key
        assert isinstance(m["detail"], str) and m["detail"], key


def test_empty_dataset_is_not_run_not_pass():
    out = _run(xr.Dataset())
    assert out["metrics"]["finite"]["status"] == "NOT_RUN"
    assert out["status"] == "NOT_RUN"
    assert all(m["status"] in {"NOT_RUN", "INFO"} for m in out["metrics"].values())


def test_masked_nan_fraction_is_bounded_by_declared_rejection_rate():
    """F2: a dataset cannot launder arbitrary NaN by claiming a screening mask."""

    import json as _json

    def _ds(nan_count: int) -> xr.Dataset:
        n = 20
        values = np.full((2, 2, 2, n), 280.0)
        values.reshape(-1)[:nan_count] = np.nan
        ds = xr.Dataset(
            {
                "t": (("time", "level", "latitude", "longitude"), values, {"units": "K"}),
                "q": (
                    ("time", "level", "latitude", "longitude"),
                    np.full((2, 2, 2, n), 1e-3),
                    {"units": "kg kg**-1"},
                ),
            },
            coords={
                "time": np.array(["2020-01-01T00", "2020-01-01T06"], dtype="datetime64[ns]"),
                "level": (("level",), [500.0, 700.0], {"units": "hPa"}),
                "latitude": (("latitude",), [20.0, 30.0], {"units": "degrees_north"}),
                "longitude": (("longitude",), np.linspace(100.0, 120.0, n), {"units": "degrees_east"}),
            },
        )
        contract = {
            "schema_version": "1.1",
            "atmosgen_version": "0.1.0",
            "variables": ["t", "q"],
            "coordinates": {
                "time": {"name": "time", "units": "h", "calendar": "proleptic_gregorian",
                         "range": ["2020-01-01T00", "2020-01-01T06"], "step_hours": 6.0},
                "level": {"name": "level", "type": "pressure_level", "units": "hPa", "values": [500, 700]},
                "latitude": {"name": "latitude", "units": "degrees_north", "range": [20.0, 30.0],
                             "step": 10.0, "ascending": True},
                "longitude": {"name": "longitude", "units": "degrees_east", "range": [100.0, 120.0],
                              "step": 1.0, "range_style": "0-360", "ascending": True},
            },
            "units": {"t": "K", "q": "kg kg**-1"},
            "source": {"kind": "synthetic", "dataset": "test"},
            "generation_method": "hybrid",
            "model_name": "hybrid.test",
            "model_version": "0.1.0",
            "random_seed": 1,
            "initial_conditions": None,
            "boundary_conditions": None,
            "physical_parameters": None,
            "generation_config": None,
            "quality_metrics": {"screening": {
                "mask_applied": True, "rejection_rate": 0.05, "rejected_samples": 4,
                "total_samples": 80, "accepted_samples": 76,
            }},
            "metadata": {"created_at": "2026-11-07T00:00:00Z"},
        }
        ds.attrs["atmosgen:contract"] = _json.dumps(contract)
        return ds

    # 4 masked elements out of 160 = 2.5% -> within declared 5% (allowed 8.5%)
    legal = physical_module.run_physical_validation(_ds(4), thresholds={})
    assert legal["metrics"]["finite"]["status"] == "PASS", legal["metrics"]["finite"]
    # 60 masked elements = 37.5% >> declared 5% -> must FAIL even though declared
    sneaky = physical_module.run_physical_validation(_ds(60), thresholds={})
    assert sneaky["metrics"]["finite"]["status"] == "FAIL", sneaky["metrics"]["finite"]
