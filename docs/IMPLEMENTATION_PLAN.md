# AtmosGen 实施计划（Phase 0 结论 → Phase 1–5 执行方案）

- 版本：v0.1（计划稿），2026-11-06 深夜定稿
- 输入：[`docs/THIRD_PARTY_AUDIT.md`](THIRD_PARTY_AUDIT.md)、[`docs/REUSE_MATRIX.md`](REUSE_MATRIX.md)、[`docs/DEPENDENCY_COMPATIBILITY.md`](DEPENDENCY_COMPATIBILITY.md)、[`docs/DATA_SCHEMA.md`](DATA_SCHEMA.md)
- 状态：**Phase 0 完成（含 1 个数据源更正 + 3 个兼容性结论）**；Phase 1–5 待实施，按 §7 逐阶段验收，禁止虚报。

---

## 0. 一句话方案

用 **SDV（BSD-Copula/CTGAN/TVAE）+ sdmetrics（MIT）** 做统计生成与统计评价，用 **Synthcity 隔离环境**做深度时序补充，用 **最小自研 2D advection–diffusion** 做物理仿真（PDEBench 只作方程/数据格式参考），用**本机 ERA5 归档**（`/Volumes/Untitled`，只读）做真实数据，全部产出走 **同一 NetCDF 契约**，由 AtmosGen 自己实现数据契约、适配层、任务执行、统一保存与四类质量验证。**不复刻任何成熟算法。**

---

## 1. 已确认的技术决策（Decision Log）

| ID | 决策 | 依据 | 影响 |
|---|---|---|---|
| D1 | 真实数据主源 = **本机 ERA5 归档**（`/Volumes/Untitled/{2010..2013}`，0.25°、37 层、逐时、t/q/u/v，每年 5–10 月 184 天，实测文件 `time=24,level=37,lat=321,lon=441`） | 匿名 GCS 实测仅 **83 KB/s**、ARCO 单 chunk 85 MB，整片传输不可行；本机归档无网络依赖、结构已核实 | Test A / Mode1 训练数据（小切片进 `data/raw/`） |
| D2 | 云水：`clwc/ciwc` 用**已有 CDS 真实样本**；`crwc/cswc` 与更大云水范围在 Phase 1 用 **CDS 时间盒补测**，失败即 NOT_RUN；`tcwv` 由 q 垂直积分**派生**并标注（或后续 CDS 获取真实值） | 本机归档无云水；DBAS 已有 CDS 配置与成本分析（单请求限额 60000，1 天全层全时次≈10656）；`~/.cdsapirc` 存在 | 契约支持 9 变量；云水单独小数据集验证 |
| D3 | 统计生成 **L1 直接用 SDV**（GaussianCopula + CTGAN + TVAE），条件生成用 `sample_from_conditions` / `sample_remaining_columns` | 源码核对（`sdv/single_table/base.py`）；主环境实测 sdv 2.1.0 装好 | Mode 1 主力；无需自研算法 |
| D4 | 统计质量指标 **L1 用 sdmetrics**（非 deprecated 的 `sdmetrics.reports.QualityReport/DiagnosticReport`；时序判别用 `sdmetrics.timeseries`） | sdmetrics 0.33.0 实测；单表旧 QualityReport 已弃用（源码 `_warn_deprecated_report`） | 统计/时序评价层 |
| D5 | **不安装 SDGym**，只借其 benchmark 协议 | BSL + `google-cloud-compute` 等重依赖；指标实现同源在 MIT 的 sdmetrics | Benchmark 用自研编排脚本 |
| D6 | Synthcity 走 **L2 隔离环境 + subprocess 适配**（py3.10 + torch 2.2.x + numpy<2） | PyPI `requires_dist` + `prereq.txt` 均 pin `torch<2.3`；与主环境 torch 2.15 冲突 | `generators/synthcity_adapter.py` + `scripts/synthcity_worker.py`；失败则 NOT_RUN |
| D7 | 物理仿真 **L4 最小自研 2D advection–diffusion**（显式迎风 + 中心扩散），不接管 PDEBench 栈 | PDEBench python<3.11、torch~=1.13、无该方程、浅水求解器为外部二进制 | `generators/physics/advection_diffusion.py`，对解析解实测 |
| D8 | RealPDEBench **只用方法**（RMSE/MAE/R²/RelL2/fRMSE/KE 定义），不复制代码、不接数据集 | CC BY-NC 4.0；数据集为流体/燃烧；其 metrics 本身 "Adapted from PDEBench"（MIT） | Sim-to-Real 评价的指标设计 |
| D9 | 输出默认 **NetCDF4**（zlib=4），Zarr 可选；契约内嵌 `atmosgen:contract` JSON | 任务书 §4；xarray/netCDF4/zarr 均为 L1 通用库 | `data/io.py` |
| D10 | 分块读取 + 小样本实验；默认 CPU、不假设 GPU；单实验 ≤30 min、峰值内存目标 <4 GB | 8 GB 内存 / 8 核 / 无 GPU 实测 | 实验规模见 §9 |
| D11 | 原始数据只读：`data/raw/` 只由下载脚本写一次，代码层路径断言拒绝写入 | 任务书 §4；单测 `test_no_raw_mutation.py` | 防误改 |
| D12 | 模式 1 的"样本"明确区分 **表格行样本** 与 **时空场样本**：表格模型生成的样本数 ≠ 完整时空场数 | 任务书 §7 注；配置校验器强制检查 | 防"10000 样本=10000 张场"的语义错误 |

---

## 2. 系统架构

### 2.1 分层（尽可能薄）

```
┌─────────────────────────────────────────────────────────────┐
│ CLI (typer): validate-config / ingest / generate / evaluate / report │
├─────────────────────────────────────────────────────────────┤
│ config.py (pydantic v2)        │ 执行编排 generation/runner.py      │
│  - YAML → 模型；跨字段校验        │  - 建 run 目录、记录耗时/内存      │
├────────────────────────────────┼───────────────────────────────┤
│ 数据层 data/                                                     │
│  contract.py（数据契约）  io.py（NetCDF/Zarr/只读/chunk）          │
│  preprocess.py（源→规范） tabular.py（场↔表/序列 转换）             │
├─────────────────────────────────────────────────────────────────┤
│ 生成层 generators/                                               │
│  registry.py  base.py（fit/sample/save 接口）                      │
│  sdv_adapter.py（L1）  synthcity_adapter.py（L2 subprocess）       │
│  physics/advection_diffusion.py（L4）+ physics/io.py              │
├─────────────────────────────────────────────────────────────────┤
│ 验证层 validation/                                               │
│  statistical.py（sdmetrics + 自定义）  temporal.py  spatial.py     │
│  physical.py（硬约束/软指标） report.py（JSON+Markdown+图）         │
├─────────────────────────────────────────────────────────────────┤
│ hybrid.py：physics → conditional generation → screening          │
└─────────────────────────────────────────────────────────────────┘
```

外部依赖边界：
- **主环境（`atmosgen`, py3.11）**：sdv 2.1.0、sdmetrics 0.33.0、xarray/netCDF4/zarr/dask/gcsfs、pydantic/typer。
- **隔离环境（`atmosgen-synthcity`, py3.10）**：仅由 `scripts/synthcity_worker.py` 以子进程方式调用，主进程不 import。
- **不引入**：SDGym 本体、PDEBench、RealPDEBench 代码。

### 2.2 目录结构（计划）

```
Atmospheric Data Generation/
├── pyproject.toml
├── README.md
├── atmosgen/                     # 包（约 2500–3500 行，不含测试）
│   ├── __init__.py               # __version__
│   ├── cli.py
│   ├── config.py
│   ├── contract.py
│   ├── runner.py                 # run 目录 / 计时 / 内存 / 种子
│   ├── data/{__init__,io,preprocess,tabular}.py
│   ├── generators/{__init__,base,registry,sdv_adapter,synthcity_adapter}.py
│   ├── generators/physics/{__init__,advection_diffusion,conditions}.py
│   ├── validation/{__init__,metrics,statistical,temporal,spatial,physical,report}.py
│   └── hybrid.py
├── configs/                      # 示例配置（demo / statistical / physics / hybrid）
├── scripts/
│   ├── ingest_local_era5.py      # 本机 ERA5 切片 → 规范数据（只写 data/raw）
│   ├── synthcity_worker.py       # 隔离环境工作进程（L2 Adapter 落地）
│   ├── validate_solver_analytic.py
│   └── run_experiments.py        # Phase 5 benchmark
├── tests/                        # pytest
├── data/{raw,processed}/         # 原始只读 / 规范化真实数据
├── output/                       # 每次生成一个 run 目录
├── docs/                         # 本计划 + 审计 + 契约 + 指南 + 报告
└── research/                     # Phase 0 证据（env / external）+ third_party/_ref
```

### 2.3 三条数据流

```
Mode 1 统计：
  data/processed/*.nc ──tabular.py（场→表格：cells×time 行）──> SDV.fit ──sample──>
  表格样本 ──reconstruct──> 生成的场 Dataset ──contract──> output/<run>/data.nc

Mode 2 物理：
  config.physics（IC/BC/参数/网格）──advection_diffusion.solve──> 场序列 Dataset
  ──物理验证（解析解/守恒/CFL）──> output/<run>/data.nc

Mode 3 混合：
  物理场（tcwv 等）──conditions──> SDV.sample_remaining_columns（条件生成 t/q/clwc/...）
  ──物理硬约束筛检（拒绝/标记）──> output/<run>/data.nc + rejected.jsonl
```

---

## 3. 统一数据契约

完整规范见 [`docs/DATA_SCHEMA.md`](DATA_SCHEMA.md)。要点：

- 维度：4D 变量 `(time, level, latitude, longitude)`；`tcwv` 为 3D `(time, latitude, longitude)`；
- 变量规范名：`t, q, u, v, clwc, ciwc, crwc, cswc, tcwv`；
- 单位/垂直坐标类型/经纬度风格/缺测策略全部显式写入契约；
- 14 个必需契约字段全部落盘（`atmosgen:contract` JSON + 独立 `contract.json`）；
- 原始数据只读，输出目录独立。

---

## 4. 三种生成模式的设计

### 4.1 Mode 1 统计生成（Phase 2）

**样本单元**（必须显式配置，防止语义混淆）：

| `samples.unit` | 含义 | 行数 | 适用 |
|---|---|---|---|
| `cell_time`（默认） | 每个 (time, level, lat, lon) 格点一条记录 | 时间×层×格点 | SDV 表格模型 |
| `field` | 每个时次一张完整场 | 时间步数 | 仅作对比实验（维度过高，CPU/8GB 不现实，默认拒绝） |

**表格构造**（`data/tabular.py`）：
- 特征列：`t,q,u,v,clwc,ciwc,tcwv…`（目标变量）
- 协变量列（可选、可配置）：`hour_sin, hour_cos, doy_sin, doy_cos, level, lat_norm, lon_norm`
- 行数控制：`data.subsample: {lat_step: 1.0, lon_step: 1.0}` 控制空间抽样（如 101×101 → 26×26）；
- 训练前剔除 NaN 行并记录比例。

**模型与参数**（`generation.model`）：

| 名称 | 实现 | 原理 | 默认超参（CPU 可承受） |
|---|---|---|---|
| `gaussian_copula` | `sdv.single_table.GaussianCopulaSynthesizer` | 统计 Copula | 默认；`enforce_min_max_values=True` |
| `ctgan` | `sdv.single_table.CTGANSynthesizer` | 深度 GAN | `epochs=100, batch_size=500, discriminator_steps=1` |
| `tvae` | `sdv.single_table.TVAESynthesizer` | 深度 VAE | `epochs=100, batch_size=500` |

**"两种不同原理"的满足**：`gaussian_copula`（Copula）与 `ctgan`/`tvae`（神经生成）满足任务书 Phase 2 的最低要求；Synthcity 为加分项（D6）。

**生成后重建场**：
- 按配置的目标网格/时间（默认沿用真实数据网格）逐格点采样；
- 写回 Dataset + 契约；`generation_config.samples` 记录实际行样本数与对应场数；
- 时空连续性**不承诺自动成立**（Gaussian Copula 不建模空间协方差），由 §5 评价如实报告；这是已知且被接受的限制（见 D3/审计 R7）。

**明确不做的**：不为了提升时空指标而把真实数据混入生成数据（会污染评价）；不做未经验证的空间平滑后处理。

### 4.2 Mode 2 物理生成（Phase 3）

**方程**（2D 平流–扩散，标量输运类，对应水汽/示踪物输送的最小模型）：

$$\frac{\partial c}{\partial t} + u\frac{\partial c}{\partial x} + v\frac{\partial c}{\partial y} = \kappa \nabla^2 c$$

- 离散：显式 Euler（时间）+ 一阶迎风（平流）+ 5 点中心差分（扩散）；
- CFL 自动检查：$dt \le \min(\Delta x/|u|, \Delta y/|v|)$ 与扩散稳定性 $\kappa dt \le \Delta x^2/2$，不满足直接报错（不静默降步长，除非配置允许）；
- 初始条件：`gaussian_plume`（解析高斯羽流）、`from_real`（取真实 q/tcwv 场）；
- 边界条件：`periodic`、`open`（零梯度）、`fixed_from_real`；
- 物理参数：`u0,v0,kappa` 可设常量或空间场；不同参数组合构成实验组；
- 输出：`(time, latitude, longitude)` 的浓度场 + 完整参数记录。

**验证（自研代码必须被独立验证）**：
1. **解析解测试**：纯扩散（u=v=0）对高斯解析解，L2 相对误差随网格加密收敛（记录 3 个分辨率的误差与收敛阶）；
2. **平流测试**：恒定风场下高斯羽流中心位移 = u·t（误差 < 2%）；
3. **质量守恒**：周期边界下总质量漂移 < 1e-6/步；
4. **稳定性**：最大 CFL 下无数值爆炸（有限值、无振荡异常）；
5. Test C：改变 IC 与参数 → 输出可测量地变化（记录各实验组的统计差异数值）。

### 4.3 Mode 3 混合生成（Phase 4）

链路（不实现端到端 Physics-Informed Diffusion）：

```
① 用 Mode 2 生成物理场：tcwv（或 q）作为"物理基底"
② 在真实数据上训练 SDV 条件模型：列 = t,q,u,v,clwc,ciwc,tcwv + 协变量
③ 条件生成：把物理场的 tcwv（及小时/层/经纬协变量）作为 known_columns，
   调 SDV.sample_remaining_columns 生成其余变量 → 得到物理+统计融合场
④ 物理硬约束筛检：非负、有限、范围、q ≤ q_sat(T,p) 等
⑤ 不合格样本：写 rejected.jsonl（原因+指标），不静默丢弃；统计拒绝率
⑥ 质量对比：真实 vs 统计 vs 物理 vs 混合，四组同表对比（§5 指标）
```

**独立性要求**：统计后处理/修正不得被当作"物理一致性已保证"；任何修正的样本必须单独通过物理验证，并在报告中区分"修正前/修正后"指标。

---

## 5. 质量验证规格（核心）

判定状态：`PASS` / `CONDITIONAL_PASS`（软指标部分不达标但记录明确）/ `FAIL` / `NOT_RUN`。**硬约束任一不满足 → 样本拒绝（物理/混合模式）或整体 FAIL（统计模式输出）**；软指标只记录数值与阈值对比，不允许"全 PASS 默认通过"。

### 5.1 统计质量（`validation/statistical.py`）

| 指标 | 方法/工具 | 默认阈值 | 来源 |
|---|---|---|---|
| 列形状（边缘分布） | sdmetrics `QualityReport`（Column Shapes：KSComplement / TVComplement） | score ≥ 0.80 | sdmetrics（MIT） |
| 列对趋势（相关） | sdmetrics `QualityReport`（Column Pair Trends：CorrelationSimilarity） | score ≥ 0.80 | sdmetrics |
| 均值比 | `mean_gen/mean_real`（按变量） | 0.85–1.15 | 自研（简单统计） |
| 标准差比 | `std_gen/std_real` | 0.75–1.25 | 自研 |
| 分位数 RMSE | q05/q25/q50/q75/q95 归一化 RMSE | ≤ 0.10 | 自研 |
| KS 统计量 | `scipy.stats.ks_2samp` 每变量 | ≤ 0.15 | scipy |
| 相关矩阵 Frobenius 距离 | 变量间相关矩阵（按 level 聚合） | ≤ 0.25 | 自研 |
| 极端值分布 | 上/下 1% 分位数的比值 | 0.5–2.0 | 自研 |

### 5.2 时间连续性（`validation/temporal.py`）

| 指标 | 方法 | 默认阈值 |
|---|---|---|
| 自相关函数差 | lag 1–6 mean |ΔACF| | ≤ 0.20 |
| 一阶差分方差比 | `var(Δx_gen)/var(Δx_real)` | 0.5–2.0 |
| 功率谱斜率差 | 沿时间 rfft → log-log 线性拟合斜率 | ≤ 0.50 |
| 突变率 | 相对真实 5σ 阈值的超阈比例 | 额外 ≤ 1% |
| 时序判别 | sdmetrics `TimeSeriesEfficacyMetric` / `TimeSeriesDetectionMetric`（LSTM，构造真实/合成序列） | 判别准确率 ≤ 0.65（≈不可区分） |
| 时间窗口统计 | 分块均值/标准差比的稳定性 | 窗口间波动 ≤ 真实 2× |

### 5.3 空间连续性（`validation/spatial.py`）

| 指标 | 方法 | 默认阈值 |
|---|---|---|
| 空间变异函数比 | 经验半变异函数 γ(h)，lag 1–5 格 | 比值 0.5–2.0 |
| 空间梯度 RMS 比 | `‖∇x_gen‖/‖∇x_real‖` | 0.5–2.0 |
| 邻域差分布 | `x - mean(4 邻域)` 的 KS | ≤ 0.20 |
| 二维功率谱斜率 | 径向平均谱 log-log 斜率差 | ≤ 0.30 |
| 各向异性比 | x/y 方向梯度能量比 | 与真实差 ≤ 30% |
| 空间相关长度 | 相关函数降到 1/e 的距离 | 与真实比 0.5–2.0 |

### 5.4 物理合理性（`validation/physical.py`）

**硬约束（任一不满足即拒绝/标记为 `HARD_FAIL`）**：

| 约束 | 判据 | 适用 |
|---|---|---|
| 有限性 | 无 NaN/Inf | 全部 |
| 比湿非负 | `q ≥ 0` | 全部 |
| 水物质非负 | `clwc, ciwc, crwc, cswc ≥ 0` | 含这些变量的数据 |
| 温度范围 | `180 K ≤ t ≤ 340 K`（默认，可配置） | 全部 |
| 风速范围 | `|u|, |v| ≤ 150 m/s`（默认，可配置） | 全部 |
| 饱和约束 | `q ≤ 1.10 × q_sat(t, p)`，`q_sat` 用 Tetens 公式（对 pressure_level 数据） | 有 t 且有压力坐标时 |
| tcwv 范围 | `0 ≤ tcwv ≤ 120 kg m⁻²`（默认） | 含 tcwv |

> 阈值不是全局常量：写在配置 `validation.physical.thresholds`，默认值按变量单位/数据源/区域给出，**报告中必须打印所用阈值**。

**软指标（记录数值，不直接判死）**：

| 指标 | 方法 | 默认期望 |
|---|---|---|
| 水汽柱守恒误差 | 周期边界下总水柱漂移 | ≤ 5% |
| 方程残差 | 生成/混合场代入 advection–diffusion 的残差 RMS（需 u,v,κ） | ≤ 相对量级阈值（实验确定后写死） |
| 数值稳定性 | 最大梯度/CFL 超限计数 | 0 |
| 平滑性 | 棋盘格指标（相邻格点符号翻转率） | ≤ 真实 2× |

### 5.5 判定与报告

- `quality_report.json`：逐指标数值、阈值、状态、样本统计量（均值/方差/KS/谱等原值）、运行资源；
- `quality_report.md`：人类可读表 + 图（分布对比、ACF、谱、变异函数、空间场快照）；
- 总体状态 = 四类中最低状态；**任何 NOT_RUN 必须在报告显著位置列出**；
- 拒绝样本：`rejected.jsonl`（逐样本原因），拒绝率写入契约 `quality_metrics`。

---

## 6. 配置与 CLI 接口

### 6.1 YAML（完整 schema，示例见 `configs/`）

```yaml
project:
  name: atmosgen_demo
  seed: 42
  output_dir: ./output
data:
  input: ./data/processed/era5_local.nc          # 由 scripts/ingest_local_era5.py 生成
  format: auto                                   # auto|netcdf|zarr
  variables: [t, q, u, v, clwc, ciwc, tcwv]
  time_range: ["2020-01-01T00", "2020-01-02T18"]
  levels: [500, 700, 850]
  spatial_extent: {lat: [20, 45], lon: [100, 125]}
  chunks: {time: 6, latitude: 50, longitude: 50}
  source_variable_map: {}      # 可选：显式覆盖源变量名
  subsample: {lat_step: 1.0, lon_step: 1.0}
generation:
  mode: statistical            # statistical | physics | hybrid
  model: gaussian_copula       # gaussian_copula|ctgan|tvae|advection_diffusion|hybrid
  samples:
    unit: cell_time            # cell_time|field（field 默认拒绝）
    rows: 20000                # 统计模式行数
    fields: 24                 # 场数（physics 模式 = 时间步数；hybrid = 场数）
  training: {epochs: 100, batch_size: 500}
physics:                       # mode=physics/hybrid 时必填
  equation: advection_diffusion
  grid: {nx: 64, ny: 64, dx_m: 42000, dy_m: 42000}
  time: {dt_s: 600, steps: 96}
  parameters: {u0: 8.0, v0: 2.0, kappa: 1.0e5}
  initial_condition: {kind: gaussian_plume, center: [32.5, 112.5], sigma_deg: 1.5, amplitude: 1.0}
  boundary_condition: {kind: periodic}
hybrid:
  physics_component: {enabled: true, output_variable: tcwv}
  statistical_component: {model: gaussian_copula, trained_on: [real, physics]}
  screening: {hard_constraints: true, reject_ratio_limit: 0.2, soft_metrics: [mass_budget]}
validation:
  statistical: true
  temporal: true
  spatial: true
  physical: true
  thresholds_overrides: {}
output:
  format: netcdf               # netcdf|zarr
  directory: ./output
  compression: {zlib: true, complevel: 4}
```

**配置校验（`atmosgen validate-config`）必须拒绝**：
- `samples.unit: field` + 表格模型（SDV）的组合；
- `mode: physics` 缺 `physics` 段 / `mode: statistical` 缺 SDV 依赖；
- `variables` 与输入数据实际变量不匹配；
- 时间范围/层/空间范围超出输入数据；
- 阈值越界（如 KS 阈值 >1）；
- 未知 model 名（列出现有 registry 项）。

### 6.2 CLI（任务书 §7 要求的 4 条全部覆盖）

```bash
atmosgen validate-config configs/demo.yaml
atmosgen ingest --config configs/era5_ingest.yaml        # Phase 1：真实数据 → data/processed
atmosgen generate --config configs/stat_gaussian_copula.yaml [--overwrite]
atmosgen evaluate --input output/<run>/data.nc [--real data/processed/era5_local.nc] [--out quality_report.json]
atmosgen report --input output/<run>/data.nc [--format md|json] [--out report.md]
atmosgen list-models
```

所有命令：退出码 0/非 0 语义明确；错误信息包含修复建议；`--help` 完整。

---

## 7. 阶段计划（工作块制，每晚一个工作块 3–4h）

### Phase 1：最小运行骨架（第 1 个工作日）

| 任务 | 产物 | 验收命令/标准 |
|---|---|---|
| 项目结构 + pyproject + 包安装 | `pyproject.toml`、`atmosgen/__init__.py`；`pip install -e .` 成功 | `atmosgen --version` |
| 配置模型 + 校验 | `config.py`；`validate-config` | 合法/非法配置各 3 例，退出码正确 |
| 数据契约 + IO（NetCDF/Zarr/chunk/只读断言） | `contract.py`、`data/io.py` | 往返读写一致；写 `data/raw` 被拒绝 |
| 真实数据 ingest | `scripts/ingest_local_era5.py`、`preprocess.py` | **Test A**：读本机 ERA5 归档小切片 → `data/processed/era5_local.nc`；变量/单位/坐标/时间/缺测/范围全部打印；原始卷只读断言 |
| registry + 生成器基类骨架 | `generators/base.py`、`registry.py` | 单测 |
| 基础测试 | `tests/` 8–10 个 | `pytest -q` 全绿 |

### Phase 2：统计生成 + 统计/时序/空间初版评价（第 2 个工作日）

| 任务 | 产物 | 验收 |
|---|---|---|
| 场↔表转换 | `data/tabular.py` | 往返可逆（允许数值精度误差） |
| SDV Adapter（2+ 模型；条件生成） | `generators/sdv_adapter.py` | **Test B**：gaussian_copula + ctgan 跑通，生成→保存→重读一致 |
| 统计质量（sdmetrics + 自定义） | `validation/statistical.py` | 数值报告，非 PASS/FAIL 空话 |
| 时序/空间质量 v1 | `temporal.py`、`spatial.py` | 指标计算在真实合成对上给出合理数值 |
| generate / evaluate CLI | `cli.py` 完整 4 命令 | 端到端可跑 |
| 实验 E1 | `output/…` + `docs/EXPERIMENT_REPORT.md` 初稿 | 模型对比表（质量+时间+内存） |
| Synthcity 隔离环境探测（时间盒） | 结论写入 `DEPENDENCY_COMPATIBILITY.md` | PASS 或 NOT_RUN（有日志证据） |

### Phase 3：物理仿真（第 3 个工作日）

| 任务 | 产物 | 验收 |
|---|---|---|
| advection–diffusion 求解器 | `generators/physics/advection_diffusion.py` | 解析解测试：3 分辨率收敛；平流位移 <2%；质量守恒 <1e-6/步 |
| IC/BC/参数配置与校验 | `conditions.py`、`config.py` 扩展 | 非法参数（CFL 超限）被拒绝并报错 |
| 物理验证 | `validation/physical.py` | 硬/软约束指标数值化 |
| **Test C**：改 IC/参数 | `scripts/validate_solver_analytic.py` + 实验记录 | 输出差异可量化（表格数值） |
| Mode 2 端到端 | `atmosgen generate --config configs/physics_advection.yaml` | 产出 NetCDF + 契约 + 报告 |

### Phase 4：混合生成（第 4 个工作日）

| 任务 | 产物 | 验收 |
|---|---|---|
| 物理 → 条件生成链路 | `hybrid.py` | 物理场作为条件列，SDV 条件生成其余变量 |
| 物理筛检 + 拒绝记录 | `rejected.jsonl` | 拒绝率与原因统计 |
| 四组对比 | 真实/统计/物理/混合 | 同表指标对比（§5） |
| 独立验证 | 报告 | 修正前/后指标分列，不把修正当物理保证 |

### Phase 5：Benchmark、报告与交付（第 5 个工作日）

| 任务 | 产物 | 验收 |
|---|---|---|
| 模型对比 benchmark | `scripts/run_experiments.py` + JSON | 统一条件、质量、速度、内存 |
| 四类质量报告汇总 | `docs/EXPERIMENT_REPORT.md` | **Test D**：具体数值 + 图 |
| 可复现性 | 固定种子二次运行 | **Test E**：数据/指标一致性（附环境快照，注明非确定性例外） |
| 完整链路 | 一条命令脚本 | **Test F**：真实读取→预处理→训练/拟合→生成→验证→NetCDF→报告 |
| 文档 8 件套 + README + 依赖/许可证清单 | `docs/*` | 与任务书 §11 对齐 |
| 自动化测试全量 | `pytest` | 全绿 + 覆盖率报告（不设无意义覆盖率 KPI） |

---

## 8. 测试与验收计划（对应任务书 Test A–F）

| 测试 | 内容 | 判定（必须有数值证据） | 证据落点 |
|---|---|---|---|
| **A** 真实数据读取 | ARCO 切片 → 规范数据集 | 变量/单位/坐标/时间/缺测/范围清单 + 与源元数据逐项一致 | `research/env/` + `output/ingest_*/*.json` |
| **B** 统计数据生成 | 至少 2 种不同原理 | 生成→保存→重读一致；指标报告非空 | `output/stat_*` + pytest |
| **C** 物理仿真 | 改 IC/参数 | 输出统计差异可量化；解析解误差达标 | `docs/EXPERIMENT_REPORT.md` E2 |
| **D** 质量评价 | 统计/时间/空间/物理 | 每类 ≥3 个数值指标 + 阈值 + 状态 | `quality_report.json` |
| **E** 可复现性 | 同配置同种子 | 两次运行指标差 ≤1%（或 bitwise 相同）；环境/版本记录完整 | `docs/EXPERIMENT_REPORT.md` |
| **F** 完整链路 | 真实→预处理→训练→生成→验证→保存→报告 | 单脚本真实跑通（非 mock），退出码 0 | `scripts/run_e2e.sh` + 日志 |

测试不通过的项：**如实标记 FAIL/NOT_RUN**，写入报告的"未完成项"章节。

---

## 9. 实验设计（Phase 5 汇总）

真实数据（`data/processed/era5_local.nc`）：本机归档 `/Volumes/Untitled/2010/` 的 2010-05-01 起若干天，0.25° 原始网格中取 level = [500, 700, 850] hPa、lat 20–45°N、lon 100–125°E、变量 t/q/u/v（tcwv 为 q 垂直积分派生）；训练表格在 1° 网格上抽样（26×26=676 格 × 3 层 × 24 时（1 天）≈ 4.9 万行；4 天 ≈ 19.5 万行，可配置）。若 CDS 云水补测成功，另加 `clwc/ciwc` 小数据集参与物理约束与混合模式。

| 实验 | 配置 | 对比指标 | 预期/关注点 |
|---|---|---|---|
| E1 统计模型对比 | gaussian_copula / ctgan / tvae | sdmetrics 质量分、KS、相关距离、训练时长、峰值内存 | Copula 快、CTGAN 慢但可能更好；记录真实差异，不预设结论 |
| E2 物理参数敏感性 | 3 组 (u0,v0,κ, IC) | 场统计、质心位移、质量守恒、谱 | 参数变化 → 输出可测变化（Test C） |
| E3 混合 vs 统计 vs 物理 | 4 组 | §5 全部指标 + 物理拒绝率 | 检验"物理基底 + 条件生成"是否真的改善时空/物理指标 |
| E4 Sim-to-Real（若数据与下游任务允许） | 真实 vs 生成 | RMSE/MAE/R²/RelL2/fRMSE/KE（RealPDEBench 指标定义） | 若下游任务不具备 → 标 NOT_RUN，不硬凑 |

资源预算：E1 单模型 ≤20 min；E2 ≤5 min；E3 ≤20 min；总实验 ≤1.5 h；每实验记录 `elapsed_seconds`/`peak_rss_mb`。

---

## 10. 资源、性能与安全

1. **内存**：所有读取用 xarray+dask 分块；禁止 `compute()` 全量；单实验峰值内存记录（`psutil`/`resource`），目标 <4 GB；
2. **算力**：默认 CPU；CTGAN/TVAE 小 epoch；Synthcity 时间盒；PDE 求解器 numpy 向量化；
3. **存储**：原始数据 <100 MB；每 run 输出 <1 GB；`output/` 可清理；
4. **数据安全**：`data/raw` 只读断言；生成不覆盖输入；输出目录含配置快照；
5. **许可证**：README + `docs` 记录 BSL/CC BY-NC 限制（见审计 §8）；不 vendor 第三方代码；
6. **成本**：无云资源默认；不假设 GPU；不启动长时间训练（>30 min 需显式配置与环境说明）。

---

## 11. 风险登记册（在 Phase 0 风险基础上补充执行期风险）

| # | 风险 | 触发信号 | 对策 |
|---|---|---|---|
| X1 | CTGAN 在 CPU 上过慢/质量差 | 单次训练 >20 min 或质量分低于 Copula 很多 | 降 epoch/行数；报告中如实对比；不硬吹 |
| X2 | 统计生成的时空指标差 | 时间/空间指标大面积 FAIL | 如实报告（设计已预期）；在混合模式中弥补；不伪造 |
| X3 | Synthcity 隔离环境不可用 | pip 解析失败/pydantic 冲突 | NOT_RUN + 证据；不影响验收底线 |
| X4 | 远程数据带宽极低（GCS 实测 38–83 KB/s，单 chunk 85 MB） | 切片 >3 min | **已改本机 ERA5 归档**；远程源仅作备选，不在主链路 |
| X5 | 物理求解器"能跑但不准" | 解析解误差大 | 必须调通再进入 Phase 4；不做"看起来像"的数据 |
| X6 | 混合筛选后拒绝率过高 | >50% | 记录原因分布，调整条件/参数；不降低阈值凑数 |
| X7 | 8 GB 内存不足 | OOM | 降分辨率/变量数；分块；记录实际资源 |
| X8 | 时间不足（晚间工作制） | 阶段延期 | 按 Phase 顺序交付可验证子集；未完成项写明 |

---

## 12. 明天（第 1 个工作日）开工清单

> 已具备：`atmosgen` 环境（sdv 2.1.0/sdmetrics 0.33.0）、**本机 ERA5 归档**（`/Volumes/Untitled`，0.25°/37 层/逐时/t,q,u,v）、云水 CDS 小样本、`third_party/_ref` 五个仓库只读参考、Phase 0 文档。
> 尚未完成：真实切片文件（engineer 今晚在拉取，若未完成第一个动作就是它）。

| 顺序 | 动作 | 预计 | 完成标志 |
|---|---|---|---|
| 1 | `git init` + 写 `pyproject.toml` + `atmosgen/__init__.py` + `pip install -e .` | 30 min | `atmosgen --version` 输出 |
| 2 | `config.py`（pydantic 模型 + 跨字段校验）+ `configs/demo.yaml` | 45 min | `atmosgen validate-config` 正反例通过 |
| 3 | `contract.py` + `data/io.py`（NetCDF/Zarr/只读断言） | 45 min | 往返单测 |
| 4 | `scripts/ingest_local_era5.py` 读 `/Volumes/Untitled/2010` 小切片 → `data/processed/era5_local.nc` | 45 min | **Test A** 报告（变量/单位/坐标/时间/缺测/范围） |
| 5 | `generators/registry.py` + `base.py` + CLI 骨架 | 30 min | pytest 全绿 |
| 6 | 当日收尾：`docs/PHASE1_REPORT.md`（实际结果 + 未完成项） | 15 min | 报告落盘 |

---

## 13. 交付物映射（任务书 §11）

| # | 要求 | 计划文件 | 阶段 |
|---|---|---|---|
| 1 | 完整源码 | `atmosgen/`、`tests/`、`scripts/` | 1–5 |
| 2 | 可安装 Python 项目 | `pyproject.toml`（`pip install -e .`） | 1 |
| 3 | 五个项目复用说明 | `docs/THIRD_PARTY_AUDIT.md`、`docs/REUSE_MATRIX.md` | ✅ Phase 0 |
| 4 | 依赖与许可证清单 | `docs/DEPENDENCY_COMPATIBILITY.md` + README 许可证节 | ✅ 0 / 5 更新 |
| 5 | 标准配置 | `configs/*.yaml`（demo/stat/physics/hybrid/ingest） | 1–4 |
| 6 | ≥2 个统计生成示例 | `configs/stat_gaussian_copula.yaml`、`configs/stat_ctgan.yaml` | 2 |
| 7 | ≥1 个物理仿真实例 | `configs/physics_advection.yaml` + 输出 | 3 |
| 8 | ≥1 个混合生成示例 | `configs/hybrid.yaml` + 输出 | 4 |
| 9 | 实际生成测试数据 | `output/*/data.nc` | 2–5 |
| 10 | 质量评价报告 | `output/*/quality_report.{json,md}`、`docs/EXPERIMENT_REPORT.md` | 2–5 |
| 11 | 完整运行文档 | `README.md`、`docs/GENERATION_GUIDE.md`、`docs/VALIDATION_GUIDE.md` | 5 |
| 12 | 自动化测试 | `tests/`（pytest） | 1–5 |

文档清单（8 件）：`README.md`、`docs/ARCHITECTURE.md`、`docs/THIRD_PARTY_AUDIT.md` ✅、`docs/REUSE_MATRIX.md` ✅、`docs/DATA_SCHEMA.md` ✅（初稿）、`docs/GENERATION_GUIDE.md`、`docs/VALIDATION_GUIDE.md`、`docs/EXPERIMENT_REPORT.md`。

---

## 14. 假设与需主人确认项（不阻塞开工，按默认执行）

| # | 事项 | 默认假设 | 影响 |
|---|---|---|---|
| A1 | 项目定位为**科研/教学** | 是；SDV BSL 与 RealPDEBench NC 均可合规使用 | 若商用需换栈（Synthcity Apache 可替代部分能力） |
| A2 | 真实数据源使用**本机 ERA5 归档**（`/Volumes/Untitled`，只读） | 是；云水用已有 CDS 样本 + Phase 1 时间盒补测 | 若外置卷不可用，回退到公开 ARCO-ERA5（带宽受限，需换网络） |
| A3 | 演示区域/时段 | 东亚 20–45°N/100–125°E，2020-01-01 起 8 个时次 | 可配置 |
| A4 | 语言 | 文档中文 + 代码/字段英文 | 可调整 |
| A5 | 验收以实际生成数据+指标数值为准 | 是 | 不以目录/代码量计 |

> 以上均为低风险默认；若主人认为 A1/A2 不成立（例如要商用、或必须用本地 ERA5 原始文件），请在开工前说明，计划 §7 的 Phase 2/3 数据路径需相应调整。
