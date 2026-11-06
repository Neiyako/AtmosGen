# Phase 0 — 五个第三方项目的技术审查

- 审查日期：2026-11-06（Asia/Shanghai）
- 审查方式：5 个仓库浅克隆到 `third_party/_ref/`（只读参考，不修改）+ 本地源码 grep 行号 + GitHub REST API + PyPI JSON 元数据 + 本机真实安装探测（`research/env/`）+ 数据源 Zarr 元数据实测（`research/external/_raw/`）。
- 本文件只写**已核实事实**；未核实内容统一列在 §7，禁止把"看起来应该可以"写成结论。

---

## 0. 结论摘要

| 项目 | 维护状态（2026-11-06） | 许可证 | 本机可用性 | 大气时空数据契合度 | AtmosGen 复用决策 |
|---|---|---|---|---|---|
| SDV | 活跃（最近 push 2026-11-06；PyPI 2.1.0，2026-10-30） | **BSL 1.1**（研究可用，禁止商业 Synthetic Data Service） | 主环境 `atmosgen` 直接 pip 可用（Python 3.11） | 表格/点样本强；场结构需 AtmosGen 侧组织 | **L1 直接复用**：GaussianCopula / CTGAN / TVAE / 条件采样 |
| Synthcity | 低活跃（push 2026-04-21；PyPI 0.2.12，2025-05-08） | Apache-2.0 | 与主环境冲突（torch<2.3、numpy<2），需隔离 Python 3.10 环境 | 有时序 GAN/VAE/Flow，最贴近时空需求 | **L2 隔离环境 Adapter**：TimeGAN / TimeVAE / FourierFlows（装通才算数） |
| SDGym | 活跃（PyPI 0.16.1，2026-11-04） | **BSL 1.1** | 依赖重（google-cloud-compute、boto3 等），不为基准测试单独引入 | 提供 benchmark 协议与指标组织 | **L3 只借协议**；指标实现直接用同源、**MIT** 的 `sdmetrics` |
| PDEBench | 中活跃（push 2026-03-30；PyPI 0.1.0 停留在 2023-08-25） | **MIT** | 主环境不可用（python<3.11、torch~=1.13、numpy<2）；legacy 环境成本高 | 无大气方程；有 1D Advection / 2D shallow water 等可参考离散化 | **L3 借方程与数值参考 + L4 最小自研** 2D advection–diffusion（任务书允许的最小补充） |
| RealPDEBench | 新（ICLR26 Oral；push 2026-11-04） | **CC BY-NC 4.0**（非商用） | Python≥3.10 可 `pip install -e .`，但依赖 torch 与 10 个模型库 | 数据集为 cylinder / fsi / foil / combustion，**非大气** | **L3 只借 sim-to-real 评价方法**；不复制其代码（NC 限制），不接其数据集 |

**一句话技术路线**：SDV 做统计生成主力，Synthcity（隔离环境）做深度时序补充，SDGym 的评测协议用 MIT 的 sdmetrics 落地，PDEBench 提供方程与数值参考（大气相关部分做最小自研），RealPDEBench 只借真实性/分布偏移评价方法。AtmosGen 自己只写：数据契约与 IO、适配层、任务执行、统一保存、质量验证与报告。

**审查中最重要的事实更正**：任务书给出的 RealPDEBench 地址 `github.com/realpdebench/RealPDEBench` 返回 **404**；其真实仓库为 `AI4Science-WestlakeU/RealPDEBench`（ICLR26 Oral，arXiv 2601.01829）。

---

## 1. 审查方法与证据位置

| 证据类型 | 内容 | 位置 |
|---|---|---|
| 源码 | 5 仓库 `--depth 1` 浅克隆，HEAD 见 §1.1 | `third_party/_ref/` |
| 元数据 | GitHub repo API、PyPI JSON（版本/requires_python/依赖 pin） | `research/external/_raw/` |
| 安装实测 | conda 环境创建、pip 安装日志、`pip freeze` | `research/env/logs/`、`research/env/REPORT.md` |
| 数据源实测 | WB2 / ARCO-ERA5 的 `.zmetadata` 变量清单与坐标属性 | `research/external/_raw/*.zmetadata` |
| 外部文档 | 官方 README / 文档页核对 | `research/external/PHASE0_EVIDENCE.md` |

### 1.1 本次审查锁定的提交

```
SDV         549dc94   sdv-dev/SDV           main
synthcity   23f322f   vanderschaarlab/synthcity
SDGym       009a35b   sdv-dev/SDGym
PDEBench    4ff3e3a   pdebench/PDEBench
RealPDEBench 06ba38d  AI4Science-WestlakeU/RealPDEBench
```

### 1.2 复用级别定义（与任务书 §2 原则 A 对齐）

- **L1 直接复用**：pip 依赖，直接调用原项目公开 API，AtmosGen 不写算法代码。
- **L2 Adapter**：原项目 API 可用但环境/数据格式不兼容，AtmosGen 写独立适配层（不改第三方源码）。
- **L3 只借方法/协议**：只参考其论文方法、指标定义、评测协议，不引入依赖、不复制代码。
- **L4 最小自研**：五个项目确实没有的能力（如大气相关方程），写最小必要代码并单独验证。

---

## 2. SDV（sdv-dev/SDV）— 统计生成

### 2.1 仓库与许可证事实

| 项 | 值 | 证据 |
|---|---|---|
| 仓库 | https://github.com/sdv-dev/SDV | GitHub API |
| 默认分支 / 审查 HEAD | main / `549dc94` | `third_party/_ref/SDV` |
| 最近 push | 2026-11-06 | GitHub API |
| Star | 约 3.6k | GitHub API |
| PyPI 最新版 | **2.1.0**（2026-10-30 上传） | PyPI JSON |
| Python 要求 | `>=3.10,<3.15` | PyPI `requires_python` |
| 关键依赖 pin | `pandas<3`；numpy 按 Python 版本分档（3.13 需 ≥2.1） | PyPI `requires_dist` |
| 许可证 | **Business Source License 1.1**（Licensor: DataCebo, Inc.） | `third_party/_ref/SDV/LICENSE` 首行 |
| 克隆内版本号 | `2.1.1.dev0` | `SDV/sdv/__init__.py:9` |

**许可证原文关键句**（`SDV/LICENSE`）：

> Additional Use Grant: You may make use of the Licensed Work, and derivatives of the Licensed Work, **provided that you do not use the Licensed Work, or derivatives of the Licensed Work, for a Synthetic Data Service.**
>
> A "Synthetic Data Service" is a **commercial offering** that allows third parties … to access the functionality of the Licensed Work …
>
> Change Date: Change date is **four years** from release date. … Change License: **MIT License**

**对 AtmosGen 的实际含义**：
- 科研/教学/内部研究使用（non-production）**允许**；
- 不得把 SDV 能力包装成面向第三方的商业"Synthetic Data Service"；
- 4 年后对应版本自动转 MIT；
- AtmosGen **不 vendor、不复制** SDV 源码，只通过 pip 作为可选依赖（`atmosgen[sdv]`），许可证随依赖声明传递。
- 另核实：SDV 的底层依赖 **Copulas 亦为 BSL 1.1**（`sdv-dev/Copulas`，Additional Use Grant 禁 "Synthetic Data Creation Service"），作为间接依赖同样只用于非生产/科研场景。

### 2.2 可复用 API（已在本仓库源码中核对）

| 能力 | 入口 | 证据 |
|---|---|---|
| 元数据自动推断 | `SingleTableMetadata().detect_from_dataframe(data)` | `SDV/sdv/metadata/_single_table.py:964` |
| 高斯 Copula 生成 | `GaussianCopulaSynthesizer` | `SDV/sdv/single_table/copulas.py` |
| 深度生成（GAN） | `CTGANSynthesizer` / `CopulaGANSynthesizer` | `SDV/sdv/single_table/ctgan.py`、`copulagan.py` |
| 深度生成（VAE） | `TVAESynthesizer` | `SDV/sdv/single_table/copulas.py`（导出） |
| 拟合 | `fit(data)` | `SDV/sdv/single_table/base.py:736` |
| 无条件采样 | `sample(num_rows)` | `base.py:1207` |
| **条件采样** | `sample_from_conditions(conditions)` / `sample_remaining_columns(known_columns=df)` | `base.py:1426`、`base.py:1512` |
| 约束 | `add_constraints` / `validate_constraints` | `base.py:459`、`base.py:533` |
| 模型持久化 | `save(filepath)` | `base.py:774` |
| 序列表格生成器 | `PARSynthesizer`（`sdv/sequential/par.py:49`，依赖 deepecho） | 可作为时序备选 |

**关键限制（源码层面）**：
- SDV 的设计对象是"行 × 列"的表格；**不理解** `(time, level, lat, lon)` 的物理维度。时空结构必须由 AtmosGen 通过「样本单元定义 + 协变量列 + 条件采样」组织（见 IMPLEMENTATION_PLAN §4）。
- 2.x 的条件采样 API 与 1.x 的 `sample_conditions` 不同，适配层需按版本分支（实现时以实际安装版本 2.1.x 为准）。

### 2.3 与大气数据的契合度

- 适合：多变量联合分布（t/q/u/v/clwc 同时出现）、点样本统计增强、条件生成（给定 tcwv/温度生成云水）。
- 不适合：直接生成空间场纹理。Gaussian Copula 只能学到列间相关，学不到二维空间协方差；空间/时间连续性需靠"协变量 + 条件采样 + 物理后处理"补足（这在设计上被明确接受，并由质量验证如实报告）。
- 结论：**L1 直接复用**，但必须由 AtmosGen 承担样本组织与验证。

---

## 3. Synthcity（vanderschaarlab/synthcity）— 深度生成

### 3.1 仓库与许可证事实

| 项 | 值 | 证据 |
|---|---|---|
| 仓库 | https://github.com/vanderschaarlab/synthcity | GitHub API |
| 审查 HEAD | `23f322f` | `third_party/_ref/synthcity` |
| 最近 push | 2026-04-21（`pushed_at`，含非默认分支）；**main 分支最后 commit 为 2025-06-17**，实际主开发已停滞约 5 个月 | GitHub API + clone |
| Star | 约 693 | GitHub API |
| PyPI 最新版 | **0.2.12**（2025-05-08） | PyPI JSON |
| Python 要求 | `>=3.9` | PyPI `requires_python` |
| 许可证 | **Apache-2.0** | `third_party/_ref/synthcity/LICENSE` |
| 关键依赖 pin | `torch<2.3,>=2.1`；`numpy<2.0`；`pandas>=2.1`；`networkx<3`；`lifelines<0.30`；`opacus`；`decaf-synthetic-data` | PyPI `requires_dist` |

**兼容性判断**：与主环境（Python 3.11 / numpy 2.x / torch 最新）**直接冲突** → 只能独立环境（Python 3.10 + torch 2.2.x + numpy<2）。这是 L2 Adapter 的典型场景：不修改其源码、不为其降级主环境。

### 3.2 可复用 API（源码核对）

目录结构为 `src/synthcity/`（不是顶层 `synthcity/`，注意与旧资料不同）：

| 能力 | 入口 | 证据 |
|---|---|---|
| 插件工厂 | `Plugin.get(name, **kwargs)`；`Plugin().fit(dataloader)`；`.generate(count)` | `synthcity/src/synthcity/plugins/core/plugin.py:695`、`:172`、`:274` |
| 数据容器 | `DataLoader(data_type="time_series", static_features=..., temporal_features=...)` | `src/synthcity/plugins/core/dataloader.py` |
| 时序模型 | `TimeGANPlugin`、`TimeVAEPlugin`、`FourierFlowsPlugin` | `plugins/time_series/plugin_timegan.py:30`、`plugin_timevae.py:25`、`plugin_fflows.py:31` |
| 通用深度模型 | `CTGANPlugin`、`TVAEPlugin`、`TabDDPMPlugin`、`BayesianNetworkPlugin`、`NFlowPlugin` 等 | `plugins/generic/` |
| 生成结果形状 | 时序数据按 3D 张量组织（样本 × 时间步 × 特征） | `dataloader.py` 的 `data_type="time_series"` 分支 |

### 3.3 与大气数据的契合度

- 时序模型（TimeGAN/TimeVAE）天然接受"多元时间序列"，最接近大气时间演化需求；但它把多变量当"时间步 × 特征"，不理解 `(lat, lon)` 二维场。可把"空间格点"当作特征维度（降采样后 8×8=64 个格点 × 变量），代价是维度和训练成本上升。
- 8GB 内存 + 纯 CPU：TimeGAN 训练一个小规模序列（如 48 时间步 × 6 变量 × 4×4 格点）可以尝试，但必须时间盒；不得为了"跑通"而伪造结果。
- 结论：**L2 隔离环境 Adapter**。若隔离环境安装失败或训练不可行，记录 NOT_RUN 并以 SDV 的 CTGAN/TVAE 满足"两种不同原理"的最低要求（Gaussian Copula vs 深度生成），不强行接入。

---

## 4. SDGym（sdv-dev/SDGym）— Benchmark

### 4.1 事实

| 项 | 值 | 证据 |
|---|---|---|
| 审查 HEAD / PyPI | `009a35b` / **0.16.1**（2026-11-04） | clone + PyPI JSON |
| 许可证 | **BSL 1.1**（与 SDV 同属 DataCebo） | `third_party/_ref/SDGym/LICENSE` |
| Python | `>=3.10,<3.15` | PyPI |
| 重依赖 | `google-cloud-compute`、`boto3/botocore`、`appdirs`、`compress-pickle`、`openpyxl` 等 | PyPI `requires_dist` |
| 核心入口 | `sdgym.benchmark_single_table(synthesizers=..., custom_synthesizers=..., additional_datasets_folder=..., compute_quality_score=...)` | `SDGym/sdgym/benchmark.py:1239` |
| 自定义生成器 | `sdgym.synthesizers.BaseSynthesizer`（`class BaselineSynthesizer(abc.ABC)`） | `SDGym/sdgym/synthesizers/base.py:23` |
| 内置可比较模型 | GaussianCopula / CTGAN / CopulaGAN / TVAE / RealTabFormer 等 | 同文件 docstring |

### 4.2 评价实现落在哪里

SDGym 本身是"基准调度器 + 数据下载 + 指标编排"，真正的指标实现来自 **SDMetrics**：

| 项 | 值 | 证据 |
|---|---|---|
| PyPI / Python | **sdmetrics 0.33.0** / `>=3.10,<3.15` | PyPI JSON |
| 许可证 | **MIT**（`third_party/_ref/SDV` 同一组织、独立仓库 `sdv-dev/SDMetrics`） | LICENSE 首行 = MIT |
| 关键类 | `QualityReport`（列形状 / 列对趋势）、`DiagnosticReport`（覆盖度等） | 官方文档 + PyPI 元数据 |

**决策**：不安装 SDGym（避免 GCP/AWS 重依赖与 BSL 传播），**直接使用 MIT 的 sdmetrics 做统计质量评价**，并用 SDGym 的 benchmark 协议（统一数据集、统一模型、统一指标、记录训练时间）作为实验设计参考。这在任务书 §3.3 中是被明确允许的路径（"允许使用其评价思想，通过 AtmosGen 自己的少量适配代码完成评价"）。

**重要 API 变更（实测 sdmetrics 0.33.0）**：`sdmetrics.reports.single_table.QualityReport` 已被官方标记 deprecated（源码首行 `_warn_deprecated_report('single table quality', 'QualityReport')`，import 时抛 `FutureWarning`）；应使用 **`from sdmetrics.reports import QualityReport, DiagnosticReport`**（统一报告 API）。另实测存在 `sdmetrics.timeseries`（`TimeSeriesEfficacyMetric`/`TimeSeriesDetectionMetric`，用 LSTM 判别真实 vs 合成）与 `sdmetrics.single_column`/`column_pairs` 指标集（`KSComplement`、`StatisticSimilarity`、`CorrelationSimilarity`、`ContinuousKLDivergence` 等），可直接复用到时序质量与列级质量评价，**无需自研这些指标**。

### 4.3 与大气数据的契合度

SDGym 面向表格基准（adult、census 等），与大气场无关；只借"评测协议"。

---

## 5. PDEBench（pdebench/PDEBench）— 物理仿真

### 5.1 事实

| 项 | 值 | 证据 |
|---|---|---|
| 审查 HEAD | `4ff3e3a` | `third_party/_ref/PDEBench` |
| 最近 push / Star | 2026-03-30 / 约 1.2k | GitHub API |
| PyPI | **0.1.0**，2023-08-25（多年未更新） | PyPI JSON |
| Python 要求 | `>=3.9,<3.11` | PyPI `requires_python` |
| 关键依赖 | `numpy<2`、`torch~=1.13.0`、`torchvision~=0.14.1`、`hydra-core`、`h5py`、`deepxde`、`pyro-ppl` | `PDEBench/pyproject.toml:21-33` |
| 许可证 | **MIT**（`LICENSE.txt`） | `third_party/_ref/PDEBench/LICENSE.txt` |

### 5.2 已实现的方程（README 核对）

`pdebench/data_gen/` 下有：1D Advection、1D Burgers、1D Reaction-Diffusion、1D Diffusion-Sorption、2D Diffusion-Reaction（`gen_diff_react.py`）、2D Shallow Water（`gen_radial_dam_break.py`）、2D Darcy、2D Compressible Navier-Stokes、2D Incompressible NS（`gen_ns_incomp.py`）。

**与大气相关的部分**：
- 1D Advection：标量平流，可作为水汽/示踪物输送的数值参考；
- 2D Shallow Water：地球物理流体动力学方程，但它是自由表面浅水方程，不是大气热力学/水物质方程；
- **没有** advection–diffusion、没有湿大气热力学、没有云微物理。

### 5.3 复用判断

- 直接 `pip install pdebench` 不可行：Python<3.11 与主环境冲突；其数据生成脚本是研究代码（需 `.env`、`hydra` cfg、legacy torch），接管成本高于收益。
- 数据格式可参考：README 描述"先生成 numpy 数组 → `Data_Merge.py` 合并为 HDF5，字段为物理量数组（如 vx, vy, density, pressure）"。
- **决策：L3 借方程/离散化与 HDF5 组织方式；Mode 2 核心自研最小 2D advection–diffusion 求解器（L4）**，理由：
  1. 任务书明确允许"如果原项目没有合适的大气方程，可以补充最少量的物理模型代码"；
  2. 自研代码可控、可验证（对高斯羽流解析解做收敛性/精度测试），比接管 legacy 研究代码更符合"以实际生成效果验收"；
  3. 不修改、不复制 PDEBench 代码；文档中注明方程来源与差异。

---

## 6. RealPDEBench（AI4Science-WestlakeU/RealPDEBench）— 真实性验证

### 6.1 事实与地址更正

| 项 | 值 | 证据 |
|---|---|---|
| 任务书地址 | `github.com/realpdebench/RealPDEBench` → **404 Not Found** | GitHub API 实测 |
| 真实仓库 | https://github.com/AI4Science-WestlakeU/RealPDEBench | GitHub search + README |
| 论文 | *RealPDEBench: A Benchmark for Complex Physical Systems with Paired Real-World and Simulated Data*，ICLR26 Oral，arXiv 2601.01829 | README 头部 |
| 审查 HEAD / push / Star | `06ba38d` / 2026-11-04 / 约 137 | clone + GitHub API |
| 安装 | `pip install -e .`，**Python ≥ 3.10** | README |
| 许可证 | **CC BY-NC 4.0（非商用）** | `third_party/_ref/RealPDEBench/LICENSE` 首行 |
| 数据集 | `cylinder`、`fsi`、`controlled_cylinder`、`foil`、`combustion`（配对真实测量 + 数值仿真） | README §Overview |
| 基线模型 | U-Net、FNO、CNO、WDNO、DeepONet、MWT、GK-Transformer、Transolver、DPOT、DMD | README |
| 评测指标（9 项） | RMSE、MAE、Rel L₂、R²、Update Ratio、fRMSE、FE、KE、MVPE | README §Overview + `realpdebench/utils/metrics.py` |

**代码事实**（逐行核对 `realpdebench/utils/metrics.py`）：
- `eval_metrics(pred, target, c, batch_size=None)` 的 docstring 明写输入形状为 **`torch.Tensor [b, t, h, w, c']`**（4D 场 + 通道），与 AtmosGen 的 `(time, level, lat, lon)` 结构不同，需转换；
- 实际返回 **13 个量**：`rmse, mae, rel_l2_error, r2, ke_error, f_error, low_f_error, mid_f_error, high_f_error, rel_low/mid/high_f_error, freq_error`；README/网页宣传的 **MVPE 在 `eval_metrics` 中并不存在**（只在论文/网页），引用前必须读论文原文；
- `ke_error` 依赖 u/v 分量假设（`c < 2` 时无意义）；
- docstring 写明 "Adapted from https://github.com/pdebench/PDEBench/blob/main/pdebench/models/metrics.py"——即 **RealPDEBench 的指标本身继承自 MIT 的 PDEBench**（AtmosGen 因而可以合法参考同一指标定义而无需接触 NC 代码）。

### 6.2 复用判断

- 数据集（流体、燃烧）**与大气无关**，直接套用不成立；
- 但"真实测量 vs 数值仿真配对 + 分布偏移/谱误差/能量误差"的评测思想可迁移：AtmosGen 的 Sim-to-Real 评价 = 真实 ERA5 vs 生成数据，采用其指标定义中的子集（RMSE、MAE、R²、Rel L₂、fRMSE、KE）；
- **许可证约束**：CC BY-NC 4.0 禁止商业使用，且要求署名。为避免许可污染：
  - **不复制**其任何源码；
  - 只参考指标定义（指标公式本身是通用数学定义）；
  - 在文档中注明方法来源；
- 决策：**L3 只借方法**，不引入依赖。

---

## 7. 真实数据源审查（Test A 的前提）

> 本节在 2026-11-06 晚做了重要更正：原计划使用匿名 GCS Zarr（ARCO-ERA5）作为主数据源，但**实测 GCS 下载速度仅 ~83 KB/s**（`curl -r 0-1000000` 耗时 12 s，2026-11-06 实测），而 ARCO 每个 chunk = 1 时次 × 37 层 × 全球（85 MB 压缩 / 154 MB 解压），完整切片需数 GB 传输，**在 8 GB 本机带宽条件下不可行**。已改用**本机真实 ERA5 归档**为主数据源。

### 7.1 本机 ERA5 归档（主数据源，已核实）

| 项 | 实测值 |
|---|---|
| 路径 | `/Volumes/Untitled/{2010,2011,2012,2013}/`（外置卷，**只读使用**） |
| 文件命名 | `ERA5_025dg_YYYYMMDD_<variable>.nc` |
| 覆盖 | 每年 184 天（**5 月 1 日 – 10 月 31 日**），逐日 24 个时次；4 年 |
| 变量 | 仅 4 个：`specific_humidity`(q)、`temperature`(t)、`u_component_of_wind`(u)、`v_component_of_wind`(v) |
| 维度（实测） | `time=24, level=37, latitude=321, longitude=441`（0.25°，lat −10…70°N，lon 50…160°E） |
| 层 | 37 个气压层：1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,300,350,400,450,500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000 hPa |
| 单位（文件 attrs 实测） | q: `kg kg**-1`；t: `K`；u/v: `m s**-1` |
| 单文件 | ≈251 MB（4 年 × 184 天 × 4 变量 ≈ 2,944 文件，约 700 GB） |
| 读取 | 本地磁盘 + netCDF4 子集读取；**无需网络**；只读打开即可 |

**意义**：Test A（真实 ERA5 读取）与全部统计实验的数据前提已解决，且不依赖外网；`data/raw/` 只需存**小切片**（如 4 个时次 × 3 层 × 25°×25°），不复制原始大文件。

### 7.2 云水变量（clwc/ciwc/crwc/cswc）

- 本机缺乏这四个变量的大范围数据。已核实可用的真实样本：`/Users/apple/Desktop/era5数据dbas/data/downloads/reanalysis-era5-pressure-levels/2024/20240101_multi_fdb02e.nc`（CDS `reanalysis-era5-pressure-levels` 下载），实测含 **`clwc`、`ciwc`**，`dims=(valid_time=1, pressure_level=3, latitude=61, longitude=141)`，单位 `kg kg**-1`（青藏高原区域、3 层、1 时次）。
- 用户机器存在 `~/.cdsapirc`（仅做存在性检查，不读取内容），且既有项目 DBAS 留有可用的 CDS 拉取配置与成本分析（`era5数据dbas/configs/fetch/era5_clwc_ciwc.yaml`；实测 CDS 单请求成本上限 60000，1 天全 37 层 24 时次 ≈ 10656，5 天 ≈ 53280 可过——见其 README/config 注释）。
- **crwc/cswc 尚未获取** → 计划：Phase 1 用 CDS 拉小切片（1–2 天、区域 [70,50,-10,160]、37 层、2 变量）；若队列/成本不可接受，标记 NOT_RUN，绝不伪造。
- 云水样本与主归档**区域/时间不同**，因此不能混进同一个规范数据集；作为独立小数据集（`data/processed/era5_cloud_sample.nc`）验证契约与物理约束路径。

### 7.3 tcwv（整层可降水量）

- 本机归档没有 2D 的 `tcwv`。两个诚实选项：
  1. **派生**：由 q 垂直积分 `tcwv = (1/g)∫q dp`（37 层覆盖 1000→1 hPa），契约标 `source.kind="derived"` 与公式，**不冒充实测 tcwv**；与真实 tcwv 的偏差需在报告中说明（37 层近似）。
  2. 用 CDS 下载 2D `total_column_water_vapour`（小区域/时段，成本低）。
- Phase 1 先实现选项 1（无网络依赖），Phase 2 视 CDS 可用性补选项 2 并对比偏差。

### 7.4 公开远程源（备选/元数据核对）

| 源 | 实测（2026-11-06） | 结论 |
|---|---|---|
| ARCO-ERA5 压力层 `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` | 277 变量；含 t/q/u/v/clwc/ciwc/tcwv；chunk=[1,37,721,1440]，单 chunk 85 MB 压缩；**GCS 下载 83 KB/s** | 元数据已核对；**带宽不可用**，仅留作将来（换网络/代理）备选 |
| ARCO-ERA5 模型层 `...model-level-1h-0p25deg.zarr-v1` | 18 变量，四云水全有，`hybrid`=137 层 | 同上，备选 |
| ARCO 粗网格 `ar/1959-2022-6h-64x32_...` / `240x121_...` | chunk 友好（[100,13,64,32] / [8,13,240,121]），但无云水变量 | 潜在快速远程备选（无云水） |
| WeatherBench2 ERA5 | 32 个物理变量，无云水 | 备选 |

### 7.5 数据源决策（写入计划）

1. **主数据**：本机 `/Volumes/Untitled/` ERA5 归档（t/q/u/v）→ 小切片 → 统一契约；
2. **云水**：DBAS 已有 CDS 小样本（clwc/ciwc）；crwc/cswc + 更大范围云水 → Phase 1 用 CDS 补（时间盒，失败即 NOT_RUN）；
3. **tcwv**：由 q 派生（标注 derived）或 CDS 获取；
4. 原始数据一律只读；`data/raw/` 只存小切片（<50 MB），不复制 700 GB 归档。

---

## 8. 关键风险与合规结论

| # | 风险 | 等级 | 结论与对策 |
|---|---|---|---|
| R1 | SDV/SDGym/Copulas 均为 BSL 1.1 | 中 | 研究/教学/内部使用可用；不 vendor、不包装成商业 Synthetic Data Service（或 Benchmarking Service / Creation Service）；作为可选 extra；4 年后转 MIT。文档中明示。 |
| R2 | RealPDEBench 为 CC BY-NC 4.0 | 中 | 不复制代码、不商用；只借方法并在文档署名。若未来商用需替换。 |
| R3 | Synthcity 依赖与主环境冲突 | 中 | 独立 conda 环境（py3.10 + torch 2.2 + numpy<2）适配；失败则 NOT_RUN，不降级主环境。 |
| R4 | PDEBench 依赖不可用（py<3.11, torch 1.13） | 高（对其直接复用） | 放弃直接复用，L3+L4；自研求解器需对解析解验证。 |
| R5 | 云水变量（crwc/cswc）需 CDS 下载，且队列/成本受限 | 中 | 用已有 CDS 小样本先验证 clwc/ciwc 路径；crwc/cswc 时间盒获取，失败标 NOT_RUN；不伪造。 |
| R6 | 8GB 内存/无 GPU/远程 GCS 带宽极低（实测 83 KB/s） | 高 | 已改本机 ERA5 归档为主数据源；分块读取、小规模实验、CPU-only、峰值内存记录。 |
| R7 | SDV 列间相关≠空间协方差 | 中 | 设计上由"协变量+条件采样+物理筛检"补足；质量验证如实报告时空指标，不粉饰。 |
| R8 | 第三方仓库持续变动 | 低 | 锁定 §1.1 的 commit；安装版本写入 `pip freeze` 与数据契约 `metadata.env`。 |

---

## 9. 未核实项（禁止当作已知）

| # | 项 | 计划核实方式 |
|---|---|---|
| U1 | Synthcity 0.2.12 在 macOS arm64 + Python 3.10 + torch 2.2 的完整安装/训练可行性 | 隔离环境实测（Phase 2，最晚 Phase 2 结束前给出 PASS/NOT_RUN 结论） |
| U2 | SDV 2.1.x `sample_from_conditions` / `sample_remaining_columns` 的实际行为与性能 | Phase 2 适配层单测 + 小样本实测 |
| U3 | ARCO-ERA5 实际切片读取耗时、网络稳定性、切片文件大小 | Phase 1 `research/env` 实测 |
| U4 | CTGAN/TVAE 在 CPU/8GB 下的训练时长与质量 | Phase 2 实验记录（时间+内存+指标） |
| U5 | RealPDEBench 各指标的精确定义细节（Update Ratio、MVPE） | 只在其论文/文档可读时引用；否则不使用该指标 |
| U6 | ERA5 CDS 的 crwc/cswc 获取流程（需 key） | 写入文档，不作为本地验收项 |

---

## 10. 对任务书两个示例假设的修正

1. **RealPDEBench 地址**：任务书写的 `realpdebench/RealPDEBench` 不存在，真实仓库是 `AI4Science-WestlakeU/RealPDEBench`。
2. **"五个项目都有可直接复用的接口"不成立**：SDV/SDGym 可用，Synthcity 需隔离环境，PDEBench 只能参考（无大气方程），RealPDEBench 只能借方法（非大气数据 + NC 许可）。AtmosGen 的工程价值恰在这四个"不成立"的缝隙里。
