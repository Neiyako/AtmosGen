"""atmosgen.validation 的单元测试（统计/时间/空间/报告）。

fixture 说明：
- ``real``：AR(1)（φ=0.7, 时间相关）+ 空间高斯平滑场（σ=2 格），4D t/q/u/v
  + 3D tcwv，16×16 网格，latitude 降序；
- ``synthetic_good``：real + 0.1% σ 的独立噪声 → 所有指标应 PASS；
- ``synthetic_bad``：时间打乱 + 每时次/层换成均值常数 + 1σ 白噪声（空间白噪声化）
  → temporal/spatial 应 FAIL；
- NOT_RUN 路径：整体 60% NaN、缺 level/time/lat 维、monkeypatch sdmetrics 崩溃。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import gaussian_filter

from atmosgen.validation import (
    aggregate_status,
    render_markdown,
    run_spatial_validation,
    run_statistical_validation,
    run_temporal_validation,
    timeseries_efficacy,
)
from atmosgen.validation import report as _report
from atmosgen.validation import statistical as _statistical

N_TIME = 16
N_LEVEL = 3
N_GRID = 16

_STATUSES = {"PASS", "FAIL", "CONDITIONAL_PASS", "NOT_RUN", "INFO"}


def _ar1_smooth(rng: np.random.Generator, *, phi: float = 0.7) -> np.ndarray:
    eps = rng.standard_normal((N_TIME, N_GRID, N_GRID))
    x = np.empty_like(eps)
    x[0] = eps[0]
    for t in range(1, N_TIME):
        x[t] = phi * x[t - 1] + math.sqrt(1 - phi ** 2) * eps[t]
    for t in range(N_TIME):
        x[t] = gaussian_filter(x[t], sigma=2.0)
    return x


def _build(seed: int) -> xr.Dataset:
    times = (np.datetime64("2020-01-01T00:00:00")
             + np.arange(N_TIME, dtype=np.int64) * np.timedelta64(6, "h"))
    coords = {
        "time": times,
        "level": np.array([850.0, 500.0, 300.0]),
        "latitude": np.linspace(40.0, -40.0, N_GRID),  # 降序
        "longitude": np.linspace(100.0, 140.0, N_GRID),
    }
    rng = np.random.default_rng(seed)
    dims4 = ("time", "level", "latitude", "longitude")
    level_offsets = np.array([290.0, 275.0, 255.0])  # (level,)

    def levelled(base: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        return base[:, None, :, :] + offsets[None, :, None, None]

    t = levelled(8.0 * _ar1_smooth(np.random.default_rng(1)), level_offsets)
    q = levelled(3.0e-4 * np.abs(_ar1_smooth(np.random.default_rng(2))),
                 np.array([1.1e-3, 3.0e-4, 8.0e-5]))
    u = levelled(8.0 * _ar1_smooth(np.random.default_rng(3)),
                 np.array([14.0, 10.0, 12.0]))
    v = levelled(5.0 * _ar1_smooth(np.random.default_rng(4)),
                 np.array([4.0, 3.0, 2.0]))
    tcwv = 20.0 + 8.0 * _ar1_smooth(np.random.default_rng(5))  # 3D
    ds = xr.Dataset(
        coords=coords,
        data_vars={
            "t": (dims4, t), "q": (dims4, q),
            "u": (dims4, u), "v": (dims4, v),
            "tcwv": (("time", "latitude", "longitude"), tcwv),
        },
    )
    return ds


def _perturb(real_ds: xr.Dataset, rel: float, seed: int) -> xr.Dataset:
    """per-variable σ×rel 的独立扰动。"""
    rng = np.random.default_rng(seed)
    good = real_ds.copy(deep=True)
    for name, da in real_ds.data_vars.items():
        sigma = float(np.std(np.asarray(da.values, dtype=float)))
        good[name].values = np.asarray(da.values) + sigma * rel * rng.standard_normal(
            np.asarray(da.values).shape)
    return good


def _whiten_time_shuffle(real_ds: xr.Dataset, seed: int) -> xr.Dataset:
    """时间打乱 + 每时次/层场白噪声化（均值保留、去掉空间结构）。"""
    rng = np.random.default_rng(seed)
    bad = real_ds.copy(deep=True)
    for name, da in real_ds.data_vars.items():
        arr = np.asarray(da.values, dtype=float)
        t_axis = da.get_axis_num("time")
        arr = np.moveaxis(np.take(arr, rng.permutation(arr.shape[t_axis]),
                                  axis=t_axis), t_axis, 0)
        has_level = "level" in da.dims
        for ti in range(arr.shape[0]):
            for li in range(arr.shape[1] if has_level else 1):
                field = arr[ti, li] if has_level else arr[ti]
                mu, sd = float(np.mean(field)), float(np.std(field))
                white = mu + sd * rng.standard_normal(field.shape)
                if has_level:
                    arr[ti, li] = white
                else:
                    arr[ti] = white
        bad[name].values = arr
    return bad


@pytest.fixture(scope="module")
def real() -> xr.Dataset:
    return _build(seed=0)


@pytest.fixture(scope="module")
def synthetic_good(real: xr.Dataset) -> xr.Dataset:
    return _perturb(real, rel=0.001, seed=101)


@pytest.fixture(scope="module")
def synthetic_bad(real: xr.Dataset) -> xr.Dataset:
    return _whiten_time_shuffle(real, seed=202)


# --------------------------------------------------------------------------
# 结构 / statistical
# --------------------------------------------------------------------------

def test_struct_keys_all_modules(real, synthetic_good):
    for fn in (run_statistical_validation, run_temporal_validation, run_spatial_validation):
        out = fn(real, synthetic_good, thresholds={})
        assert set(out) == {"metrics", "status", "notes"}
        assert out["status"] in _STATUSES
        for key, m in out["metrics"].items():
            assert set(m) == {"value", "threshold", "comparison", "status", "detail"}, (fn, key)
            assert m["status"] in _STATUSES
            assert m["comparison"] in {"max", "min", "range", "info"}
            assert isinstance(m["detail"], str)


def test_statistical_good_passes(real, synthetic_good):
    out = run_statistical_validation(real, synthetic_good, thresholds={})
    assert out["status"] == "PASS"
    assert 0.0 <= out["metrics"]["ks"]["value"] <= 1.0
    assert out["metrics"]["ks"]["threshold"] == 0.15
    for key in ("mean_ratio", "std_ratio", "extreme_ratio"):
        assert out["metrics"][key]["status"] == "PASS", out["metrics"][key]
    assert out["metrics"]["column_shapes"]["status"] == "PASS"
    assert out["metrics"]["column_pair_trends"]["status"] == "PASS"
    assert out["metrics"]["quantile_rmse"]["status"] == "PASS"
    assert out["metrics"]["corr_frobenius"]["status"] == "PASS"


def test_statistical_threshold_sensitivity(real, synthetic_good):
    out = run_statistical_validation(
        real, synthetic_good, thresholds={"ks_max": 1e-6})
    assert out["metrics"]["ks"]["status"] == "FAIL"
    # 其余阈值仍用默认 → 不受影响
    assert out["metrics"]["column_shapes"]["status"] != "FAIL"


def test_statistical_nan_heavy_not_run(real):
    heavy = real.copy(deep=True)
    for name, da in heavy.data_vars.items():
        arr = np.asarray(da.values, dtype=float)
        mask = np.random.default_rng(7).random(arr.shape) < 0.6
        arr[mask] = np.nan
        heavy[name].values = arr
    out = run_statistical_validation(heavy, heavy.copy(deep=True), thresholds={})
    for key in ("ks", "quantile_rmse", "corr_frobenius", "mean_ratio"):
        assert out["metrics"][key]["status"] == "NOT_RUN", (key, out["metrics"][key])
    assert any("缺测" in n for n in out["notes"])


def test_statistical_partial_nan_conditional(real):
    partial = real.copy(deep=True)
    arr = np.asarray(partial["tcwv"].values, dtype=float)
    mask = np.random.default_rng(8).random(arr.shape) < 0.6
    arr[mask] = np.nan
    partial["tcwv"].values = arr
    out = run_statistical_validation(partial, partial.copy(deep=True), thresholds={})
    # tcwv 独列被剔除 → 简单统计只剩部分列 → CONDITIONAL_PASS
    assert out["metrics"]["mean_ratio"]["status"] == "CONDITIONAL_PASS"
    assert any("tcwv" in n for n in out["notes"])


def test_statistical_sdmetrics_failure_degrades(real, synthetic_good, monkeypatch):
    def boom():
        raise RuntimeError("simulated sdmetrics crash")

    monkeypatch.setattr(_statistical, "_quality_report_cls", boom)
    out = run_statistical_validation(real, synthetic_good, thresholds={})
    for key in ("column_shapes", "column_pair_trends"):
        m = out["metrics"][key]
        assert m["status"] == "NOT_RUN", m
        assert "simulated sdmetrics crash" in m["detail"]
    # 其余指标不受影响
    assert out["metrics"]["ks"]["status"] == "PASS"


# --------------------------------------------------------------------------
# temporal
# --------------------------------------------------------------------------

def test_temporal_good_passes(real, synthetic_good):
    out = run_temporal_validation(real, synthetic_good, thresholds={})
    assert out["status"] == "PASS"
    assert out["metrics"]["acf_diff"]["value"] >= 0.0
    assert out["metrics"]["acf_diff"]["threshold"] == 0.20
    assert out["metrics"]["timeseries_efficacy"]["status"] == "NOT_RUN"
    assert any("Phase 2 默认关闭" in n for n in
               [out["metrics"]["timeseries_efficacy"]["detail"]])


def test_temporal_bad_fails(real, synthetic_bad):
    out = run_temporal_validation(real, synthetic_bad, thresholds={})
    assert out["status"] == "FAIL"
    failing = [k for k, m in out["metrics"].items() if m["status"] == "FAIL"]
    assert "acf_diff" in failing  # 打乱时间 → ACF 差远超 0.20


def test_temporal_nan_heavy_not_run(real):
    # 全体缺测 >50% 的 synthetic：real 保持原样 → 全部序列被剔 → NOT_RUN
    heavy = real.copy(deep=True)
    for name, da in heavy.data_vars.items():
        arr = np.asarray(da.values, dtype=float)
        arr[...] = np.where(np.random.default_rng(9).random(arr.shape) < 0.9,
                            np.nan, arr)
        heavy[name].values = arr
    out = run_temporal_validation(real, heavy, thresholds={})
    assert out["status"] == "NOT_RUN"
    assert all(m["status"] == "NOT_RUN" for m in out["metrics"].values())


def test_temporal_not_run_missing_time(real):
    snapshot = real.isel(time=0, drop=True)  # 去掉 time 维
    out = run_temporal_validation(real, snapshot, thresholds={})
    assert out["status"] == "NOT_RUN"
    assert "time" in out["notes"][0]


def test_timeseries_efficacy_function(real, synthetic_good):
    out = timeseries_efficacy(real, synthetic_good, max_sequences=5)
    assert set(out) == {"metrics", "status", "notes"}
    m = out["metrics"]["timeseries_efficacy"]
    assert set(m) == {"value", "threshold", "comparison", "status", "detail"}
    # sdmetrics TimeSeries 依赖（sklearn/tf）可能不在 → NOT_RUN 也接受
    assert m["status"] in _STATUSES


# --------------------------------------------------------------------------
# spatial
# --------------------------------------------------------------------------

def test_spatial_good_passes(real, synthetic_good):
    out = run_spatial_validation(real, synthetic_good, thresholds={})
    assert out["status"] == "PASS"
    for key, m in out["metrics"].items():
        assert m["status"] == "PASS", (key, m)
    assert any("场抽样" in n or "抽样场分布" in n for n in out["notes"])


def test_spatial_bad_fails(real, synthetic_bad):
    out = run_spatial_validation(real, synthetic_bad, thresholds={})
    assert out["status"] == "FAIL"
    failing = [k for k, m in out["metrics"].items() if m["status"] == "FAIL"]
    assert {"gradient_rms_ratio", "neighbor_ks"} & set(failing)


def test_spatial_not_run_missing_dims(real):
    no_lat = real.mean("latitude", keep_attrs=False)  # 去掉 latitude 维
    out = run_spatial_validation(real, no_lat, thresholds={})
    assert out["status"] == "NOT_RUN"
    assert all(m["status"] == "NOT_RUN" for m in out["metrics"].values())


def test_spatial_nan_heavy_not_run(real):
    heavy = real.copy(deep=True)
    for name, da in heavy.data_vars.items():
        arr = np.asarray(da.values, dtype=float)
        arr[...] = np.where(np.random.default_rng(10).random(arr.shape) < 0.9,
                            np.nan, arr)
        heavy[name].values = arr
    out = run_spatial_validation(real, heavy, thresholds={})
    assert out["status"] == "NOT_RUN"


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def test_aggregate_status_rules():
    assert aggregate_status([]) == "NOT_RUN"
    assert aggregate_status(["PASS", "PASS"]) == "PASS"
    assert aggregate_status(["PASS", "CONDITIONAL_PASS"]) == "CONDITIONAL_PASS"
    assert aggregate_status(["PASS", "NOT_RUN"]) == "NOT_RUN"
    assert aggregate_status(["PASS", "NOT_RUN", "FAIL"]) == "FAIL"
    assert aggregate_status(["NOT_RUN", "NOT_RUN"]) == "NOT_RUN"
    assert aggregate_status(["FAIL"]) == "FAIL"
    assert aggregate_status(["INFO", "PASS"]) == "PASS"  # INFO 只记录数值
    assert aggregate_status(["UNKNOWNTOKEN"]) == "FAIL"  # 未知状态保守按 FAIL
    assert aggregate_status([None]) == "FAIL"


def test_render_markdown_contains_tables_and_status(real, synthetic_good):
    report = {
        "statistical": run_statistical_validation(real, synthetic_good, thresholds={}),
        "temporal": run_temporal_validation(real, synthetic_good, thresholds={}),
        "spatial": run_spatial_validation(real, synthetic_good, thresholds={}),
        "meta": {"title": "test", "model_name": "unit", "random_seed": 42},
    }
    md = render_markdown(report)
    assert md.startswith("# AtmosGen 质量验证报告")
    assert "**Overall status:" in md
    assert "**Section status:" in md
    assert md.count("| metric | value | threshold | comparison | status | detail |") == 3
    assert "| acf_diff |" in md or "| acf_diff " in md
    assert "| ks |" in md
    assert "| gradient_rms_ratio |" in md
    assert "note:" in md  # notes 被渲染


def test_report_is_pure_python():
    import inspect
    src = inspect.getsource(_report)
    assert "import numpy" not in src and "import pandas" not in src
