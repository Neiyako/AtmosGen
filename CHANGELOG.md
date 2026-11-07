# Changelog

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。所有重要变更记录于此。

## [Unreleased]

- 新增 `scripts/quality_fidelity.py`：合成数据保真度诊断（边缘 KS / 时序相关 / 空间图案相关 /
  lag-1 ACF / 饱和违例率，以及**导数型量** MFC 的图案相关与幅度比），面向云水资源量等下游计算的可用性评估。
- 新增 `docs/DATA_QUALITY_FOR_CLOUD_WATER.md`：合成数据在云水资源量计算中的可用性边界（机理 + 实测 + 改造路线）。
- 新增 `tests/test_quality_fidelity.py`（8 用例）；测试总数 148。

## [1.0.0] — 2026-11-07

首个**正式版**：Phase 0–5 全部交付（契约 / IO / 配置 / CLI / 统计生成 / 物理仿真 / 混合生成 / 四类质量验证 / 实验与交付物）。

### 新增

- **统一数据契约**（schema 1.1）：14 个必需字段，内嵌 `atmosgen:contract` + `contract.json` 侧车；
  schema 1.1 新增物理追迹变量 `c`（Mode 2 浓度场）。
- **数据 IO**：`data/io.py` NetCDF4/Zarr 原子写、只读打开、`data/raw` 写保护、契约校验；
  `data/tabular.py` 场 ↔ 表双向转换；`data/selection.py` 时间/层/空间子集；
  `data/preprocess.py` 本机 ERA5 归档 ingest（派生 `tcwv`，Test A 报告）。
- **配置与 CLI**：`atmosgen validate-config / ingest / list-models / generate / evaluate / report`；
  pydantic v2 跨字段校验（含 CFL/扩散稳定性、样本单元语义、source 字段按 kind 白名单）。
- **Mode 1 统计生成**（SDV 2.1）：`gaussian_copula` / `ctgan` / `tvae`，**level_wise** 方案
  （每层一个模型，规避连续协变量条件采样在经验分布边界的退化）。
- **Mode 2 物理仿真**（自研）：2D advection–diffusion（显式 Euler + 一阶迎风 + 5 点中心扩散），
  `gaussian_plume` / `from_real` 初值，`periodic` / `open` / `fixed_from_real` 边界，
  稳定性双层校验（CFL、单轴 C+2r、**联合 (Cx+Cy)+2(Rx+Ry)≤1**、扩散和 ≤0.5）。
- **Mode 3 混合生成**：物理基底 → 条件生成（`sample_remaining_columns`，已知列 bit 级保留）→
  逐格点硬约束筛检 → `rejected.jsonl` → 契约 `quality_metrics.screening_before/after`。
- **四类质量验证**：`statistical`（sdmetrics + 自研）、`temporal`、`spatial`、`physical`
  （硬约束 + 软指标），统一返回结构与状态聚合（FAIL > NOT_RUN > CONDITIONAL_PASS > PASS）。
- **报告**：`quality_report.json`（严格 JSON）/ `quality_report.md`（表格 + 3 图）。
- **实验与工具**：`scripts/validate_solver_analytic.py`（6 项解析/守恒/稳定性 + Test C）、
  `scripts/run_experiments.py`（E1/E2/E3 benchmark → `summary.json`）、
  `scripts/run_e2e.sh`（Test F 全链路）、`scripts/synthcity_probe.py`（隔离环境时间盒）。
- **文档**：`README` + `ARCHITECTURE` / `GENERATION_GUIDE` / `VALIDATION_GUIDE` / `EXPERIMENT_REPORT` /
  `DATA_SCHEMA` / `THIRD_PARTY_AUDIT` / `REUSE_MATRIX` / `DEPENDENCY_COMPATIBILITY` /
  `IMPLEMENTATION_PLAN` + 阶段报告 PHASE1–PHASE5。

### 已知限制（详见各阶段报告）

- 统计模式**不建模**时间/空间结构（时空指标按设计 FAIL）；层间相关不建模。
- 混合模式条件化会改变剩余变量的边缘分布（时空改善 ↔ 边缘变差的权衡）。
- `equation_residual` 仅支持单层 2D 风场、未按场量级归一化，阈值待定标。
- Synthcity 仅做隔离环境可用性探测（集成 PASS / 质量 NOT_RUN），未启用为生成器。
- `crwc/cswc` 仍需 CDS 补测；`hybrid.screening.reject_ratio_limit` 未强制。
