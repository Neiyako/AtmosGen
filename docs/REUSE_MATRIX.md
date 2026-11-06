# 复用矩阵（REUSE MATRIX）

- 输入：`docs/THIRD_PARTY_AUDIT.md` 中已核实的事实（2026-11-06）
- 原则：L1 直接复用 > L2 Adapter > L3 只借方法 > L4 最小自研。**L4 只出现在五个项目确实没有、且对生成质量有实际影响的环节。**

## 1. 能力 → 来源 → 决策 矩阵

| # | AtmosGen 需要的能力 | 首选来源 | 复用级别 | AtmosGen 侧组件 | 不采用时的退路 |
|---|---|---|---|---|---|
| C1 | 多变量统计分布学习（Gaussian Copula） | SDV `GaussianCopulaSynthesizer` | **L1** | `generators/sdv_adapter.py` | 无（这是主算法之一） |
| C2 | 深度表格生成（GAN/VAE） | SDV `CTGANSynthesizer` / `TVAESynthesizer` | **L1** | `generators/sdv_adapter.py` | Synthcity `CTGANPlugin`/`TVAEPlugin`（L2） |
| C3 | 深度时序生成（TimeGAN / TimeVAE / FFlow） | Synthcity `TimeGANPlugin`、`TimeVAEPlugin`、`FourierFlowsPlugin` | **L2**（隔离环境） | `generators/synthcity_adapter.py` | 标记 NOT_RUN；由 C1/C2 满足"≥2 种原理" |
| C4 | 条件生成（给定 tcwv/t 等生成其余变量） | SDV `sample_from_conditions` / `sample_remaining_columns` | **L1** | `generators/sdv_adapter.py` + `hybrid.py` | 无条件生成 + 物理筛检（质量下降，需如实报告） |
| C5 | 统计质量指标（列形状、列对趋势、覆盖度） | `sdmetrics`（MIT）`QualityReport` / `DiagnosticReport` | **L1** | `validation/statistical.py` | 自实现 KS/相关（L4，仅兜底） |
| C6 | Benchmark 协议（统一数据集/模型/指标/性能记录） | SDGym 的 benchmark 设计 | **L3** | `scripts/benchmark.py` + `validation/report.py` | 无（协议参考即可，不需依赖） |
| C7 | PDE 数值求解（advection / advection–diffusion） | PDEBench 的方程与离散化参考 | **L3 + L4** | `generators/physics/advection_diffusion.py` | 无（PDEBench 无该方程） |
| C8 | 初始条件 / 边界条件 / 物理参数配置 | 参考 PDEBench 的 cfg/HDF5 组织 | **L3 + L4** | `generators/physics/*.py` + YAML `physics:` 段 | 无 |
| C9 | 物理仿真数据保存（HDF5/NetCDF） | 参考 PDEBench 字段组织 | **L3 + L4** | `data/io.py`（NetCDF 统一） | Zarr |
| C10 | Sim-to-Real / 分布偏移 / 谱与能量误差指标 | RealPDEBench 指标定义（RMSE/MAE/R²/RelL2/fRMSE/KE） | **L3** | `validation/physical.py`、`validation/spatial.py` | 用 PDEBench MIT 指标定义（同源） |
| C11 | NetCDF/Zarr 分块读写、惰性计算 | `xarray` + `netCDF4` + `zarr` + `dask` + `gcsfs` | **L1**（通用库） | `data/io.py` | 无 |
| C12 | 统一数据契约与元数据 | **无第三方**：AtmosGen 核心自研 | **L4** | `contract.py` + `docs/DATA_SCHEMA.md` | — |
| C13 | YAML 配置 + 校验 + CLI | `pydantic` + `typer`（通用库） | **L1** | `config.py`、`cli.py` | argparse |
| C14 | 报告（JSON/Markdown/图） | 自研最小实现 + `matplotlib` | **L4** | `validation/report.py` | — |
| C15 | 混合生成编排（物理 → 条件生成 → 筛检） | 无现成实现（组合 C7/C4/C5） | **L4（仅编排）** | `hybrid.py` | — |

## 2. 项目 × 能力 覆盖矩阵

图例：**●** 本项目直接提供成熟实现；**◐** 需要 Adapter/格式转换；**○** 只提供方法参考；**—** 无关或不可用。

| 能力 | SDV | Synthcity | SDGym | PDEBench | RealPDEBench |
|---|---|---|---|---|---|
| 多变量统计生成 | ● | ●（generic） | — | — | — |
| 深度生成（GAN/VAE/Flow/Diffusion） | ●（CTGAN/TVAE/CopulaGAN） | ●（CTGAN/TVAE/DDPM/NFlow/TimeGAN/TimeVAE/FFlow） | — | — | — |
| 时序生成 | ◐（PAR，需 deepecho） | ●（time_series 三类） | — | — | — |
| 条件生成 | ● | ◐（constraints / generate 条件） | — | — | — |
| 统计质量指标 | ◐（通过 sdmetrics） | ◐（有 evaluator） | ●（编排 sdmetrics） | — | — |
| Benchmark 协议 | — | — | ● | — | — |
| 数值 PDE 求解 | — | — | — | ●（8+ 方程） | ●（配对仿真数据） |
| 大气物理方程 | — | — | — | ○（无大气专用方程） | — |
| 时空场生成 | ○（表格模型，需组织） | ◐（时序模型，需张量化） | — | ●（场仿真） | ●（场数据） |
| Sim-to-Real 评价 | — | — | ◐（跨数据集比较） | ◐（指标实现，MIT） | ●（9 指标协议） |
| 大气数据格式 | — | — | — | — | — |

## 3. "不构建"清单（防止过度工程）

明确**不做**以下事情（任务书原则 A / 最终决策标准的直接落实）：

1. 不实现 GAN/VAE/Copula/Diffusion 算法本身；
2. 不做通用插件框架、不做 Web UI、不做数据库、不做多用户/权限系统；
3. 不做统一"实验管理平台"；benchmark 用一个脚本 + JSON 结果即可；
4. 不接管 PDEBench 的训练/数据生成栈（legacy python<3.11）；
5. 不安装 SDGym 本体（重依赖 + BSL），只用其协议与 MIT 的 sdmetrics；
6. 不接 RealPDEBench 数据集（非大气 + NC 许可）；
7. 不为 Synthcity 修改主环境依赖（隔离环境，失败即记录 NOT_RUN）；
8. 不引入未经验证的"物理后处理"来假装提升物理一致性（修正必须独立验证）。

## 4. 许可证传播清单

| 依赖 | 许可证 | 分发方式 | 对交付的影响 |
|---|---|---|---|
| SDV（可选 extra） | BSL 1.1 | pip 依赖，不 vendor | 研究用途 OK；不得包装为商业合成数据服务；交付物中不包含其源码 |
| sdmetrics | MIT | pip 依赖 | 可直接使用与引用 |
| Synthcity（可选 extra，隔离环境） | Apache-2.0 | pip 依赖，不 vendor | 可商用/可再分发（保留 NOTICE） |
| PDEBench | MIT | **不引入**，只参考方程 | 文档注明参考；不复制代码 |
| RealPDEBench | CC BY-NC 4.0 | **不引入代码/数据** | 只参考方法并署名；交付物不含其代码 |
| ERA5 数据（本机归档 / ARCO-ERA5 / WB2） | Copernicus 产品许可（署名、无担保） | 数据不随包分发（只读引用本机卷/由用户脚本获取） | README 注明数据来源与获取方式 |
| xarray/netCDF4/zarr/dask/gcsfs/pydantic/typer/scipy/scikit-learn/matplotlib | BSD/Apache/MIT 等宽松 | pip 依赖 | 正常 |

## 5. 决策溯源

| 决策 | 依据 |
|---|---|
| SDV 用 L1 而非 L2 | 主环境实测可装（`research/env/`），API 稳定且公开 |
| Synthcity 用 L2 | `requires_dist` 明示 `torch<2.3`、`numpy<2.0`，与主环境冲突（`docs/DEPENDENCY_COMPATIBILITY.md`） |
| SDGym 用 L3 | BSL + `google-cloud-compute` 等重依赖；评价实现同源但在 MIT 的 sdmetrics 中 |
| PDEBench 用 L3+L4 | `requires_python <3.11`、`torch~=1.13`，且无大气方程 |
| RealPDEBench 用 L3 | `CC BY-NC 4.0`；数据集为流体/燃烧 |
| 数据契约自研 L4 | 五个项目均无大气时空统一契约；任务书 §4 明确要求 AtmosGen 自主完成 |
