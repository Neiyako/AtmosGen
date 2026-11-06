# 统一数据标准（DATA SCHEMA）

> 状态：Phase 0 设计定稿（2026-11-06），Phase 1 按此实现。
> 原则：复用 `xarray`/NumPy/NetCDF/Zarr，不发明新文件格式；AtmosGen 只定义**变量、维度、坐标、单位、元数据契约**。

## 1. 设计目标

1. 同一种结构承载：真实 ERA5 切片、统计生成结果、物理仿真结果、混合生成结果；
2. 变量语义、单位、缺测值处理**依据数据源元数据**，不靠变量名猜测；
3. 原始数据只读；生成数据独立存储；可回溯（哪个模型、哪个种子、哪份配置生成了它）；
4. 8 GB 内存下可分块读写（不在内存中展开完整 ERA5）。

## 2. 规范数据集结构

逻辑结构（任务书 §4 指定）：

```
X(time, level, latitude, longitude, variable)
```

物理存储（NetCDF4 / Zarr）采用 xarray 的**变量-维度**模型：

- 三维变量（`t, q, u, v, clwc, ciwc, crwc, cswc`）：`(time, level, latitude, longitude)`
- 二维变量（`tcwv`）：`(time, latitude, longitude)`（整层积分量，物理上没有 level 维，**不得**复制到每一层冒充三维）

> 说明：任务书的 `variable` 维是**逻辑**维度——xarray 中用"每个变量一个 data variable"表达，二者等价且更符合 CF 约定；若下游需要 `(time, level, lat, lon, variable)` 的稠密张量，由 `data/tabular.py` 的 `to_dense_array()` 在内存允许时现做。这一点必须在 `docs/ARCHITECTURE.md` 里写明，避免误解。

### 2.1 变量命名（AtmosGen 规范名）

| 规范名 | 物理含义 | 单位 | 典型维度 | ERA5 源变量名（CDS / pressure-level） | 备注 |
|---|---|---|---|---|---|
| `t` | 温度 | K | 4D | `temperature` | 本机归档文件内变量名为 `t`（`long_name=Temperature`） |
| `q` | 比湿 | kg kg⁻¹ | 4D | `specific_humidity` | 本机归档文件内变量名为 `q` |
| `u` | 东西向风速 | m s⁻¹ | 4D | `u_component_of_wind` | 本机归档文件内变量名为 `u` |
| `v` | 南北向风速 | m s⁻¹ | 4D | `v_component_of_wind` | 本机归档文件内变量名为 `v` |
| `clwc` | 云液态水含量（比湿形式） | kg kg⁻¹ | 4D | `specific_cloud_liquid_water_content` | 已有 CDS 真实小样本（2024-01-01，青藏高原） |
| `ciwc` | 云冰水含量 | kg kg⁻¹ | 4D | `specific_cloud_ice_water_content` | 同上 |
| `crwc` | 雨水含量 | kg kg⁻¹ | 4D | `specific_rain_water_content` | 需 CDS 下载；Phase 1 时间盒补测，失败标 NOT_RUN |
| `cswc` | 雪水含量 | kg kg⁻¹ | 4D | `specific_snow_water_content` | 同上 |
| `tcwv` | 整层可降水量（水柱当量） | kg m⁻²（= mm） | 3D | `total_column_water_vapour` | 本机归档无；Phase 1 由 q 垂直积分派生（deviation 标注）或 CDS 获取 |

**单位不硬编码**：读取时从源数据 `units` 属性解析；规范化时写入契约 `units` 字段；若源单位与上表不一致（如 tcwv 为 `m`），转换并记录 `unit_conversion_applied`。绝不"按变量名猜单位"。

**本机主数据源（`/Volumes/Untitled/{2010..2013}`）的映射**：文件内变量短名已经是 `t/q/u/v`（`temperature`/`specific_humidity`/`u_component_of_wind`/`v_component_of_wind` 的 `long_name` 与 `units` 已实测）；`tcwv` 不在该归档中，Phase 1 采用 **q 垂直积分派生**：`tcwv = (1/g)∫q dp`（37 层近似），契约 `source.kind="derived"`、`derivation="tcwv = (1/g) * integral(q dp) over 1000..1 hPa"`，**不得**被报告当作 ERA5 官方 tcwv。

### 2.2 坐标

| 坐标 | 类型 | 要求 |
|---|---|---|
| `time` | `datetime64[ns]`（UTC，无时区） | 唯一、单调递增；保留原始 `calendar`/`units` 于 attrs |
| `level` | float/int | **必须记录垂直坐标类型**：`pressure_level`（hPa）或 `model_level`（无量纲层号）；二者不得混用 |
| `latitude` | float | 保留原始方向（ERA5 自北向南）；契约记录 `ascending: false`；生成时不得静默翻转 |
| `longitude` | float | 保留原始 0–360 或 -180–180；契约记录 `range: [0,360]`；跨 0 度时由预处理负责，不静默改写 |

### 2.3 缺测值

- 源数据 `_FillValue`/NaN 原样保留，**不做静默插值**；
- 契约记录：`missing_policy: "preserve"`、`missing_count`、`missing_fraction`（按变量）；
- 统计生成训练前：缺失行剔除并记录剔除比例；生成结果若出现物理域外值，按物理硬约束处理（不通过即拒绝/标记）；
- 任何填充（无论何种方法）必须写入 `initial_conditions`/`generation_config` 并在报告中出现。

## 3. 数据契约（Dataset attrs）

统一写在一个 JSON 字符串属性 `atmosgen:contract` 中（NetCDF global attr / Zarr root attr），避免属性爆炸；同时把常用字段平铺为普通 attr 便于工具读取。

```json
{
  "schema_version": "1.0",
  "atmosgen_version": "0.1.0",
  "variables": ["t", "q", "u", "v", "clwc", "ciwc", "tcwv"],
  "coordinates": {
    "time":   {"name": "time",   "units": "hours since 1900-01-01", "calendar": "proleptic_gregorian", "range": ["2020-01-01T00", "2020-01-02T18"], "step_hours": 6},
    "level":  {"name": "level",  "type": "pressure_level", "units": "hPa", "values": [500, 700, 850]},
    "latitude":  {"name": "latitude",  "units": "degrees_north", "range": [20.0, 45.0], "step": 1.0, "ascending": false},
    "longitude": {"name": "longitude", "units": "degrees_east", "range": [100.0, 125.0], "step": 1.0, "range_style": "0-360"}
  },
  "units": {"t": "K", "q": "kg kg**-1", "u": "m s**-1", "v": "m s**-1", "clwc": "kg kg**-1", "ciwc": "kg kg**-1", "tcwv": "kg m**-2"},
  "source": {
    "kind": "real",
    "dataset": "local_era5_archive",
    "uri": "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    "variable_map": {"t": "temperature", "q": "specific_humidity", "...": "..."},
    "access_date": "2026-11-07",
    "subset": {"time": "2020-01-01/2020-01-02T18", "level": [500, 700, 850], "lat": [20, 45], "lon": [100, 125]},
    "sha256": "…",                     // 规范化后文件校验和
    "readonly": true,                  // 原始数据未被修改的声明
    "license": "Copernicus Climate Change Service (C3S) licence"
  },
  "generation_method": "statistical | physics | hybrid | ingest",
  "model_name": "sdv.GaussianCopulaSynthesizer | physics.advection_diffusion | hybrid.v1 | null",
  "model_version": "2.1.0 | 0.1.0",
  "random_seed": 42,
  "initial_conditions": {"kind": "era5_slice | gaussian_plume | real_q_field", "details": {}},
  "boundary_conditions": {"kind": "periodic | open | fixed_from_real", "details": {}},
  "physical_parameters": {"u0": 8.0, "v0": 2.0, "kappa": 1.0e5, "grid": {"nx": 64, "ny": 64, "dx_m": 42250, "dy_m": 42250}, "dt_s": 600.0},
  "generation_config": {"config_path": "configs/…yaml", "config_sha256": "…", "samples": 20000, "training": {"epochs": 100, "batch_size": 500}},
  "quality_metrics": {"summary": {"statistical": "PASS", "temporal": "CONDITIONAL_PASS", "spatial": "FAIL", "physical": "PASS"}, "report_path": "output/…/quality_report.json"},
  "metadata": {
    "created_at": "2026-11-07T00:30:00Z",
    "host": "macOS arm64 8GB",
    "python": "3.11.15",
    "env_lock": "research/env/logs/atmosgen_pip_freeze.txt",
    "elapsed_seconds": 123.4,
    "peak_rss_mb": 512.0,
    "notes": ""
  }
}
```

字段覆盖任务书 §4 要求的全部 14 项（`variables/coordinates/units/source/generation_method/model_name/model_version/random_seed/initial_conditions/boundary_conditions/physical_parameters/generation_config/quality_metrics/metadata`）。

## 4. 文件布局与存储约定

```
data/
  raw/            # 只写一次：ingest 脚本产出的小切片；生成过程禁止写入
  processed/      # 规范化后的真实数据（canonical NetCDF）
output/
  <run_id>/       # 一次生成一个目录：数据 + 配置副本 + 契约 + 质量报告 + rejected.jsonl
    data.nc
    config.yaml            # 原始配置快照
    contract.json          # 契约（也内嵌在 data.nc）
    quality_report.json
    quality_report.md
    rejected.jsonl         # 被硬约束拒绝的样本（不静默丢弃）
    figures/*.png
```

- 格式：默认 **NetCDF4**（`engine="netcdf4"`，`zlib=True, complevel=4`，chunk 由 `data:` 配置控制）；`zarr` 通过同一接口可选；
- 文件名/run_id：`<mode>_<model>_<YYYYmmdd-HHMMSS>_<seed>`；
- 原始数据只读的实现层约束：`data/io.py` 的 `open_source()` 只返回只读句柄；`save_canonical()` 拒绝写入 `data/raw/` 前缀（路径断言 + 单测）。

## 5. 生成结果的可追溯性

每个生成文件必须能回答：**谁生成的、用什么模型版本、什么种子、什么配置、原始数据来自哪里、质量如何**。因此：

1. `contract.json` 与数据文件同目录保存（冗余但可独立读取）；
2. `config.yaml` 原样快照（含 `sha256`）；
3. `quality_report.json` 由 `atmosgen evaluate` 产出，**允许后续补充**（文件存在不代表质量合格；状态字段才是结论）；
4. 重新生成同一文件名时必须拒绝（`--overwrite` 显式覆盖）；
5. `output/` 与 `data/` 分离（任务书 §9）。

## 6. 与生成模式的对应

| 模式 | 产出的维度 | 关键契约字段 |
|---|---|---|
| ingest（Phase 1） | 真实切片规范化 | `source.kind="real"`, `generation_method="ingest"` |
| Mode 1 统计 | `(time, level, lat, lon)` 与真实同网格（或配置指定） | `model_name="sdv.*"`, `generation_config.samples` |
| Mode 2 物理 | `(time, level=1 或自定义, lat, lon)`；可有多变量 | `physical_parameters`, `initial_conditions`, `boundary_conditions` 必填 |
| Mode 3 混合 | 物理基底 + 统计条件生成 | 同时含 `physics` 与 `statistical` 子记录 + `screening` 统计 |

## 7. 兼容性

- `schema_version` 变更规则：新增可选字段 → minor；删除/改名/单位变化 → major，且读取端必须显式拒绝不认识的 major；
- 读取 `atmosgen:contract` 失败 → 明确报错（不"尽力而为"）；
- 第三方可直接用 xarray 打开，不依赖 AtmosGen 即可读变量（契约只是属性，不影响数据可读性）。
