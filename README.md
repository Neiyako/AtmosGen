# AtmosGen — 大气仿真数据生成框架

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-informational)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/tests-140%20passed-success)](https://github.com/Neiyako/AtmosGen/actions)

面向大气科学研究的**仿真数据生成框架**：输入真实大气数据（ERA5），指定生成条件与规模，输出**带统一数据契约**的合成数据，并附带**四类质量验证报告**（统计 / 时间 / 空间 / 物理）。

> **设计立场**：不重新发明生成算法。统计生成用 [SDV](https://github.com/sdv-dev/SDV)，物理仿真用**自研最小求解器**（不 vendor 任何第三方源码），
> 混合模式把二者用物理基底连接。AtmosGen 只负责**数据契约 / IO / 适配层 / 任务编排 / 统一保存 / 质量验证与报告**。

---

## 目录

- [1. 它解决什么问题](#1-它解决什么问题)
- [2. 核心能力](#2-核心能力)
- [3. 安装](#3-安装)
- [4. 快速开始](#4-快速开始)
- [5. 三种生成模式](#5-三种生成模式)
- [6. 数据契约](#6-数据契约)
- [7. 质量验证](#7-质量验证)
- [8. 实测结果](#8-实测结果)
- [9. CLI 参考](#9-cli-参考)
- [10. 配置速查](#10-配置速查)
- [11. 项目结构](#11-项目结构)
- [12. 文档索引](#12-文档索引)
- [13. 数据来源与许可](#13-数据来源与许可)
- [14. 已知限制](#14-已知限制)
- [15. 复现与验证](#15-复现与验证)
- [16. 参与贡献](#16-参与贡献)
- [17. 引用](#17-引用)
- [18. 许可证](#18-许可证)

---

## 1. 它解决什么问题

生成大气"仿真数据"时，常见的三类断裂是：

| 断裂 | AtmosGen 的做法 |
|---|---|
| 真实数据、生成数据、报告三套格式互不相认 | **一份契约**贯穿全流程：真实切片、统计生成、物理场、混合结果都是同一结构（`atmosgen:contract` 内嵌 + `contract.json` 侧车），第三方工具可直接用 xarray 打开 |
| "看起来像"但无法量化 | **四类验证**逐指标给数值/阈值/状态，报告显著列出未运行项；整体状态按 `FAIL > NOT_RUN > CONDITIONAL_PASS > PASS` 聚合，**不把"没跑"当"通过"** |
| 物理不合理的数据被当成成果 | 物理模式自带**解析解验证**（收敛阶、平流位移、质量守恒、CFL）；混合模式逐格点**硬约束筛检**，不合格样本写 `rejected.jsonl` 并统计拒绝率 |

## 2. 核心能力

| 能力 | 说明 | 状态 |
|---|---|---|
| 统一数据契约 | 14 个必需字段、schema 1.0 → 1.1（新增物理追迹量 `c`）、维度/单位/坐标类型校验 | ✅ |
| 真实数据 ingest | 本机 ERA5 归档 → 规范 NetCDF（派生 `tcwv`）、**只读断言**、Test A 报告 | ✅ |
| Mode 1 统计生成 | SDV `gaussian_copula` / `ctgan` / `tvae`，level-wise 方案，条件生成钩子 | ✅ |
| Mode 1' **结构保持型合成** | `analog_blocks`：真实场时序块 × 整域重组，**保住导数型量**（MFC 图案 r 0.89 vs 统计模式 0.00），物理违例 0 | ✅ |
| Mode 2 物理仿真 | 自研 2D advection–diffusion（显式 Euler + 迎风 + 中心扩散），3 种初值 / 3 种边界 | ✅ |
| Mode 3 混合生成 | 物理基底 → 条件生成 → 硬约束筛检 → `rejected.jsonl` | ✅ |
| 四类质量验证 | statistical / temporal / spatial / physical，统一返回结构 + 阈值覆盖 + 状态聚合 | ✅ |
| 报告 | `quality_report.json`（严格 JSON）+ `quality_report.md`（表 + 3 图） | ✅ |
| 实验与复现 | E1/E2/E3 benchmark、Test E（逐位可复现）、Test F（单脚本全链路） | ✅ |
| 可复现性 | 同配置同种子 → **数据逐位一致** | ✅ |

## 3. 安装

```bash
# 1) 环境（Python 3.10+，本项目在 3.11 / macOS arm64 / 8 GB 上实测）
conda create -y -n atmosgen python=3.11
conda activate atmosgen

# 2) 运行时依赖 + 本项目（统计生成需要 sdv/sdmetrics）
pip install numpy pandas xarray netCDF4 zarr dask pyyaml pydantic typer \
            scipy matplotlib psutil pytest
pip install -e .            # 装 atmosgen 包与 atmosgen 命令
pip install -e '.[stat]'    # 需要统计生成时再加 sdv/sdmetrics/scikit-learn
pip install -e '.[dev]'     # 需要开发/测试时再加 pytest

# 3) 验证安装
atmosgen --version          # -> atmosgen 1.0.0
pytest -q                   # -> 140 passed
```

> **Synthcity（可选）**：需独立环境 `atmosgen-synthcity`（Python 3.10 + torch 2.2 + numpy<2），
> 且必须额外 `pip install scipy==1.11.4 opacus==1.4.1`（否则在本机会 dlopen 失败 / 需要 torch≥2.6）。
> 详见 [`docs/DEPENDENCY_COMPATIBILITY.md`](docs/DEPENDENCY_COMPATIBILITY.md) §10。本项目**未启用** Synthcity 作为生成器。

## 4. 快速开始

```bash
cd Atmospheric-Data-Generation

# ① 真实数据 → 规范数据集（Test A；需要本机 ERA5 归档 /Volumes/Untitled）
atmosgen ingest --config configs/era5_ingest.yaml

# ② 生成（三种模式任选）
atmosgen generate --config configs/stat_gaussian_copula.yaml   # Mode 1 统计
atmosgen generate --config configs/physics_advection.yaml      # Mode 2 物理
atmosgen generate --config configs/hybrid.yaml                 # Mode 3 混合

# ③ 质量评价（物理模式无同变量真实参考 → 不传 --real）
atmosgen evaluate --input output/statistical_gaussian_copula_*/data.nc \
                  --real data/processed/era5_local.nc --fail-on never

# ④ 渲染报告
atmosgen report --input output/<run>/quality_report.json --format md \
                --out output/<run>/quality_report_rendered.md
```

产物目录结构：

```
output/<mode>_<model>_<YYYYmmdd-HHMMSS>_<seed>/
├── data.nc                 # 带内嵌契约的规范数据集
├── contract.json           # 契约侧车（含最终文件 SHA-256）
├── metadata.json           # 资源/耗时/峰值 RSS/训练统计/筛检统计
├── environment.json        # 环境与依赖版本快照（可复现性依据）
├── config.yaml             # 本次运行的配置快照
├── quality_report.json     # 四类验证结果（严格 JSON）
├── quality_report.md       # 人类可读报告（含 3 张图）
├── rejected.jsonl          # 混合模式：被物理筛检拒绝的样本（原因+指标+坐标）
└── figures/                # distributions / acf_spectrum / variogram
```

## 5. 三种生成模式

### Mode 1 · 统计生成（SDV）

把真实场展平成 `cell_time` 表格（行 = 一个 `(time, level, lat, lon)` 格点），
**按层**训练单表模型（`gaussian_copula` / `ctgan` / `tvae`），再采样并按栅格顺序重建完整场。

> **为什么按层（level-wise）？** 实测 SDV 2.1 对连续协变量做条件采样时会在**经验分布边界退化**：
> 以 `level=500` 条件采样时温度被压到 `248.0 K ± 0.02 K`（真实 262.3 ± 5.0 K），
> 且 32 448 行耗时 148 s；CTGAN/TVAE 的条件采样是拒绝采样（约 6 s/行）。详见 [`docs/PHASE2_REPORT.md`](docs/PHASE2_REPORT.md) §7。

### Mode 2 · 物理仿真（自研求解器）

求解 `∂c/∂t + u ∂c/∂x + v ∂c/∂y = κ ∇²c`：显式 Euler + 一阶迎风 + 5 点中心扩散。

- 初值：`gaussian_plume`（解析高斯羽流）/ `from_real`（真实场输运）
- 边界：`periodic`（环绕）/ `open`（零梯度）/ `fixed_from_real`（边缘固定）
- 稳定性**双层强制**：CFL、单轴 `C+2r≤1`、**联合 `(Cx+Cy)+2(Rx+Ry)≤1`**、扩散和 `≤0.5`；
  违规直接报错，不静默降步长

### Mode 1' · 结构保持型合成（`analog_blocks`）

单表统计模型能保住**边缘分布**，但保不住**时空结构**：对独立采样的场求散度（云水资源量的核心量
`MFC = -∇·(q·V)`）会把噪声放大到与信号同量级——实测 MFC 图案相关 0.002 而幅度比 0.885
（"量级像、逐格全错"）。`analog_blocks` 改为**重组真实结构**：连续时序块按计划重排 + 整域取源，
实测 MFC 图案相关 **0.894**、空间图案 0.739、lag-1 ACF 0.708、饱和违例 0%。
输出是真实场的重组而非独立样本，契约以 `independence=recombined` 与
`novelty.exact_copy_cell_fraction` **如实披露**。

### Mode 3 · 混合生成（物理基底 + 条件生成 + 筛检）

物理求解器把真实 `tcwv` 输运成物理基底 → SDV 以该基底为**已知列**条件生成其余变量
（已知列逐位保留，不允许被改写）→ 逐格点做 §5.4 硬约束筛检 → 不合格样本写 `rejected.jsonl`，
交付数据集中以 NaN 遮罩（并在契约里声明遮罩比例，物理验证据此区分"遮罩"与"坏值"）。

## 6. 数据契约

所有产出共享同一契约（`docs/DATA_SCHEMA.md`）：

```jsonc
{
  "schema_version": "1.1",
  "variables": ["t", "q", "u", "v", "tcwv"],
  "coordinates": { "time": {...}, "level": {...}, "latitude": {...}, "longitude": {...} },
  "units": {"t": "K", "q": "kg kg**-1", "tcwv": "kg m**-2"},
  "source": {"kind": "real|synthetic|derived", "dataset": "...", "sha256": "...", "readonly": true},
  "generation_method": "ingest|statistical|physics|hybrid",
  "model_name": "sdv.GaussianCopulaSynthesizer",
  "model_version": "2.1.0",
  "random_seed": 42,
  "initial_conditions": {...}, "boundary_conditions": {...}, "physical_parameters": {...},
  "generation_config": {"samples": {"unit": "cell_time", "rows_generated": 32448, ...}},
  "quality_metrics": {"screening_before": {...}, "screening_after": {...}},
  "metadata": {"created_at": "...", "elapsed_seconds": ..., "peak_rss_mb": ...}
}
```

- 存储：NetCDF 全局属性 `atmosgen:contract`（JSON 字符串）+ 同目录 `contract.json` 侧车；
- 校验：缺字段 / schema 大版本 / 变量维度 / 单位 / 坐标类型逐项检查，读不到即报错（不做"尽力而为"）；
- 原始数据只读：`data/raw/` 默认拒写（仅 ingest 显式放行）。

## 7. 质量验证

| 类别 | 代表指标 | 默认阈值 |
|---|---|---|
| statistical | column_shapes / column_pair_trends / KS / mean·std 比 / 分位 RMSE / 相关矩阵距离 | shapes ≥0.80、KS ≤0.15、mean 0.85–1.15 |
| temporal | ACF 差 / 一阶差分方差比 / 谱斜率差 / 突变率 / 窗方差比 | ACF ≤0.20、方差比 0.5–2.0 |
| spatial | 半变异函数比 / 梯度 RMS 比 / 邻域差 KS / 空间谱斜率 / 各向异性 / 相关长度 | 比值 0.5–2.0、KS ≤0.20 |
| physical | 有限性 / q 非负 / 云水非负 / 温度范围 / 风速上限 / 饱和约束 / tcwv 范围（+ 质量守恒、平滑度、方程残差） | 180–340 K、150 m/s、q ≤1.1×q_sat、tcwv ≤120 |

状态语义与聚合规则、阈值键名、如何解读"整体 FAIL"，见 [`docs/VALIDATION_GUIDE.md`](docs/VALIDATION_GUIDE.md)。

## 8. 实测结果

> 全部数字来自本机实跑（macOS arm64 / 8 GB / 无独立 GPU），原始 JSON 在 `output/experiments*/summary.json`。

**E1 · 统计模型对比**（同一真实子集、seed 42）

| 模型 | fit s | sample s | column_shapes ↑ | KS ↓ | 统计模块 |
|---|---|---|---|---|---|
| gaussian_copula | 8.1 | 0.5 | **0.953** | **0.087** | 7/8 指标 PASS |
| ctgan | 251.9 | 0.6 | 0.869 | 0.237 | 3/8 PASS |
| tvae | 128.2 | 0.4 | 0.883 | 0.222 | 4/8 PASS |

**E2 · 物理求解器**（`scripts/validate_solver_analytic.py`，6/6 PASS）

| 验证 | 实测 |
|---|---|
| 纯扩散解析解 L2（dx 20/10/5 km） | 1.84e-3 → 4.50e-4 → 1.12e-4，收敛阶 **2.03 / 2.01** |
| 平流质心位移误差 | **1.70%**（<2%） |
| 周期质量守恒 | 最大相对漂移 **2.86e-16** |

**E3 · 真实 / 统计 / 物理 / 混合 四组对比**

| 指标 | statistical | hybrid | 说明 |
|---|---|---|---|
| KS vs real ↓ | **0.087** | 0.250 | 混合模式条件化改变边缘分布 |
| spatial.variogram_ratio ↓ | 16.57 | **9.74** | 物理基底带来空间结构 |
| temporal.diff_var_ratio ↓ | 338.2 | **163.9** | 物理基底带来时间连续性 |
| physical.saturation 违例 | 1066 | **0**（筛检后） | 混合模式筛检拒绝率 **2.01%**（653/32448） |

**诚实声明**：统计模式**不建模**时空结构，其 temporal/spatial 指标按设计 FAIL（计划 §4.1、风险 X2），
这是**已知限制**而非缺陷；混合模式的边缘统计代价同样是真实权衡。报告不做"美化"。

**可复现性**：同配置同种子两次生成 → 数据**逐位一致**（max|diff| = 0）；Test F 全链路 **1m51s PASS**。

## 9. CLI 参考

| 命令 | 作用 | 退出码 |
|---|---|---|
| `atmosgen --version` | 版本 | 0 |
| `atmosgen validate-config <cfg>` | 配置结构 + 跨字段 + 数据级校验 | 0 / 2 |
| `atmosgen ingest --config <cfg>` | 真实数据 → 规范数据集（Test A） | 0 / 2 / 4 |
| `atmosgen generate --config <cfg>` | 生成（statistical / physics / hybrid） | 0 / 2 / 4 |
| `atmosgen evaluate --input <f> [--real <f>]` | 四类验证 → `quality_report.{json,md}` | 0 / 2 / 4（`--fail-on` 控制质量 FAIL 是否算失败） |
| `atmosgen report --input <f> --format md\|json` | 渲染报告 | 0 / 2 |
| `atmosgen list-models` | 已注册模型与实现状态 | 0 |

退出码约定：`0` 成功｜`2` 配置/用法错误｜`3` 当前阶段未实现｜`4` 运行期失败或质量 FAIL。

## 10. 配置速查

```yaml
project: {name: demo, seed: 42, output_dir: ./output}
data:
  input: ./data/processed/era5_local.nc
  variables: [t, q, u, v, tcwv]
  levels: [500, 700, 850]
  subsample: {lat_step: 1.0, lon_step: 1.0}
generation:
  mode: statistical          # statistical | physics | hybrid
  model: gaussian_copula     # 必须与 mode 匹配且已注册
  samples: {unit: cell_time, rows: 20000}
  training: {epochs: 100, batch_size: 500}
physics:                     # mode=physics/hybrid 必填
  equation: advection_diffusion
  grid: {nx: 64, ny: 64, dx_m: 42000.0, dy_m: 42000.0}
  time: {dt_s: 600.0, steps: 96}
  parameters: {u0: 8.0, v0: 2.0, kappa: 100000.0}
  initial_condition: {kind: gaussian_plume, center: [30.0, 110.0], sigma_deg: 1.5}
  boundary_condition: {kind: periodic}
hybrid:                      # mode=hybrid 必填
  physics_component: {enabled: true, output_variable: tcwv}
  statistical_component: {model: gaussian_copula, trained_on: [real, physics]}
  screening: {hard_constraints: true, reject_ratio_limit: 0.2, soft_metrics: [mass_budget]}
validation: {statistical: true, temporal: true, spatial: true, physical: true,
             thresholds_overrides: {ks_max: 0.12}}
output: {format: netcdf, directory: ./output, compression: {zlib: true, complevel: 4}}
```

完整字段与约束：[`docs/GENERATION_GUIDE.md`](docs/GENERATION_GUIDE.md)。

## 11. 项目结构

```
atmosgen/
├── cli.py              # typer 入口（6 个命令）
├── config.py           # pydantic v2 配置模型 + 跨字段校验
├── contract.py         # 数据契约（14 字段、schema 1.1）
├── generation.py       # 三模式统一编排
├── evaluation.py       # 对齐 + 四类验证 + 报告落盘
├── hybrid.py           # 条件生成断言 + 硬约束筛检 + rejected.jsonl
├── runner.py           # run 目录、种子、耗时/峰值 RSS、环境快照
├── data/{io,tabular,selection,preprocess}.py
├── generators/{sdv_adapter,hybrid_adapter}.py, generators/physics/{advection_diffusion,conditions}.py
└── validation/{statistical,temporal,spatial,physical,report}.py
configs/     # demo / ingest / 3 统计 / physics / hybrid
scripts/     # validate_solver_analytic · run_experiments · run_e2e.sh · synthcity_probe
tests/       # 140 个用例（pytest）
docs/        # 架构 / 生成指南 / 验证指南 / 实验报告 / 数据契约 / 审计 / 阶段报告
research/    # Phase 0 证据（环境探测、外部资料核对）
```

## 12. 文档索引

| 文档 | 内容 |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 分层结构、三条数据流、关键决策与代价、扩展点 |
| [`docs/GENERATION_GUIDE.md`](docs/GENERATION_GUIDE.md) | 配置逐段说明、CLI 用法、三模式最小示例、故障处置 |
| [`docs/VALIDATION_GUIDE.md`](docs/VALIDATION_GUIDE.md) | 四类指标/阈值/状态语义、报告解读、已知限制 |
| [`docs/EXPERIMENT_REPORT.md`](docs/EXPERIMENT_REPORT.md) | E1/E2/E3 实验、Test D/E/F |
| [`docs/DATA_QUALITY_FOR_CLOUD_WATER.md`](docs/DATA_QUALITY_FOR_CLOUD_WATER.md) | 合成数据在**云水资源量**等导数型下游计算中的可用性边界（含 MFC 实测） |
| [`docs/PHASE6_REPORT.md`](docs/PHASE6_REPORT.md) | 结构保持型合成（`analog_blocks`）：把导数型量从不可用提升到可用的实测对照 |
| [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md) | 统一数据契约与文件布局 |
| [`docs/THIRD_PARTY_AUDIT.md`](docs/THIRD_PARTY_AUDIT.md) / [`docs/REUSE_MATRIX.md`](docs/REUSE_MATRIX.md) | 五个候选项目的技术审查与复用边界 |
| [`docs/DEPENDENCY_COMPATIBILITY.md`](docs/DEPENDENCY_COMPATIBILITY.md) | 本机依赖实测、Synthcity 时间盒结论 |
| [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | 实施计划（Phase 0–5、测试 A–F） |
| [`docs/PHASE1_REPORT.md`](docs/PHASE1_REPORT.md) … [`docs/PHASE5_REPORT.md`](docs/PHASE5_REPORT.md) | 各阶段实际结果、偏差与未完成项 |

## 13. 数据来源与许可

| 组件 | 许可 | 本项目用法 |
|---|---|---|
| **SDV / SDGym / Copulas** | **BSL 1.1** | 依赖。允许研究/非生产使用；**禁止商业 Synthetic Data Service**，4 年后转 MIT |
| **sdmetrics** | MIT | 依赖（统计质量指标） |
| **Synthcity** | Apache-2.0 | 可选依赖，隔离环境；未启用为生成器 |
| **PDEBench** | MIT | 仅参考（不引入、不 vendor） |
| **RealPDEBench** | **CC BY-NC 4.0** | 仅参考指标定义（不复制代码/数据，非商用） |
| **ERA5** | Copernicus C3S 数据许可 | **不在本仓库分发**；需自行从 CDS 获取 |

本仓库**不 vendor 任何第三方源码**，仅在运行期依赖它们。详见 [`LICENSE`](LICENSE) 与
[`docs/DEPENDENCY_COMPATIBILITY.md`](docs/DEPENDENCY_COMPATIBILITY.md)。

## 14. 已知限制

1. 统计模式不建模时间/空间结构 → temporal/spatial 指标按设计 FAIL（**不是** bug，已在报告中如实标注）。
2. 混合模式条件化会改变剩余变量的边缘分布（时空改善 ↔ 边缘变差的权衡），未做分布重映射等后处理。
3. `equation_residual` 仅支持单层 2D 风场、未按场量级归一化，阈值待定标；多层数据集记 NOT_RUN。
4. 卫星水物质 `crwc/cswc` 仍需 CDS 补测（当前数据只含 `clwc/ciwc`）。
5. Synthcity 仅完成隔离环境可用性探测（集成 PASS / 质量 NOT_RUN），未作为生成器启用。
6. `hybrid.screening.reject_ratio_limit` 目前只记录拒绝率、不强制拦截。
7. `sdmetrics` 非弃用 `QualityReport` 在单表包装下 `Column Pair Trends` 返回 NaN（0.33.0 实测），
   故统计模块沿用带弃用告警的 single_table 实现，升级条件已记录。

## 15. 复现与验证

```bash
pytest -q                                     # 148 passed
python scripts/validate_solver_analytic.py    # 求解器 6 项解析/守恒/稳定性验证
python scripts/run_experiments.py --experiments e1,e2,e3   # benchmark → output/experiments/summary.json
python scripts/quality_fidelity.py --real data/processed/era5_local.nc \
       --runs output/statistical_gaussian_copula_*/data.nc   # 保真度 + MFC 诊断
bash scripts/run_e2e.sh                      # Test F 全链路（需要 /Volumes/Untitled 挂载）
```

阶段结论与独立复核记录（含每轮修复的缺陷清单）：[`docs/PHASE1_REPORT.md`](docs/PHASE1_REPORT.md) ~ [`docs/PHASE5_REPORT.md`](docs/PHASE5_REPORT.md)。

## 16. 参与贡献

欢迎 issue / PR。提交前请确保：

```bash
pytest -q                      # 全绿
pip install -e '.[dev]'        # 开发依赖
```

改动生成逻辑时请同步更新：对应阶段的 `docs/PHASE*_REPORT.md`（实测数值）、`docs/EXPERIMENT_REPORT.md`（实验表）与
`CHANGELOG.md`。涉及契约字段的改动必须提升 `contract.SCHEMA_VERSION` 并在 `docs/DATA_SCHEMA.md` 记录。

## 17. 引用

```bibtex
@software{atmosgen2026,
  title  = {AtmosGen: Atmospheric Synthetic Data Generation with Unified Data Contract and Four-Class Quality Validation},
  author = {{AtmosGen contributors}},
  year   = {2026},
  url    = {https://github.com/Neiyako/AtmosGen},
  version = {1.0.0}
}
```

## 18. 许可证

代码：[MIT](LICENSE)。数据与第三方组件各自遵循其上游许可（见 §13）。
