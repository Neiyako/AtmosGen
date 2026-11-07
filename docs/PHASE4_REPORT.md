# Phase 4 报告 · 混合生成（物理基底 + 条件生成 + 物理筛检）

- 日期：2026-11-07
- 依据：[`docs/IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §4.3、§5.4、§7 Phase 4、§9 E3；[`docs/PHASE3_REPORT.md`](PHASE3_REPORT.md)
- 结论：**Phase 4 交付项完成**。混合链路按计划 §4.3 的六步实现：物理基底 → 条件生成 → 物理硬约束筛检 → `rejected.jsonl` → 修正前/后指标分离 → 四组对比（E3）；真实数据端到端跑通，拒绝率 2.01%（653/32448，全部为超饱和），筛检后同批数据 0 违例。

---

## 1. 一句话结果

`atmosgen generate --config configs/hybrid.yaml` 用物理求解器把真实 tcwv 平流/扩散成物理基底（26×26 @1°、16 时次），再让按层训练的 SDV 模型**以该基底为已知列**条件生成 t/q/u/v（`sample_remaining_columns`，known 列逐行 bit 级保留），最后逐格点做 §5.4 硬约束筛检：**32448 个样本中 653 个超饱和（2.01%）被拒绝**并写入 `rejected.jsonl`，拒绝单元在数据集中以 NaN 遮罩；重新筛检通过单元得到 **0 违例**（"修正后"指标）。单次运行约 7 分钟（8 GB / CPU）。

## 2. 六步链路（对应计划 §4.3）

| 步骤 | 实现 | 证据 |
|---|---|---|
| ① 物理基底 | `AdvectionDiffusionGenerator` 以 `from_real(tcwv)` 为初值，平流/扩散得到输运场；求解器变量 `c` 在混合模式映射为配置的 `physics_component.output_variable`（tcwv） | `hybrid_adapter.py::sample`；`metadata.json` 的 `physics_diagnostics` |
| ② 统计模型训练 | 与 Mode 1 相同的 level-wise SDV（`data` 段 + `subsample`），每层一个模型 | `hybrid_adapter.py::fit` → `SDVGenerator.fit` |
| ③ 条件生成 | 每个层的模型以物理基底（扁平化后逐格点）作为 known 列调用 `sample_remaining_columns`；**known 列必须原样返回**，否则抛 `HybridError`（防止静默改写物理基底） | `hybrid.conditional_generate`（含 bit 级断言） |
| ④ 物理硬约束筛检 | `hybrid.screen_samples`：finite / q_nonneg / cloud_nonneg / t_range / wind_range / saturation(Tetens, 1.1×) / tcwv_range，**逐样本**给原因 | `hybrid.py`；`quality_metrics.screening_before` |
| ⑤ 拒绝记录 | `rejected.jsonl` 每行 = 行号 + reasons + 样本值 + time/level/lat/lon；拒绝率入契约；拒绝单元以 NaN 遮罩（`mask_applied: true`） | `run_dir/rejected.jsonl`（653 行）；`atmosgen_screening` 语义见 `VALIDATION_GUIDE` §6 |
| ⑥ 修正前/后指标 | 契约 `quality_metrics.screening_before / screening_after` 分开记录；after 只统计通过单元且重新跑筛检（0 违例） | `quality_report.json` / `metadata.json` |

## 3. 真实数据实测（run: `output/hybrid_hybrid_<ts>_42`）

| 指标 | 值 |
|---|---|
| 网格 / 时次 | 26×26 @1°（与真实 1° 子网格逐点对齐）、16 时次（索引式映射，见偏差 P2） |
| 总样本（cell_time） | 32 448 |
| 筛检前拒绝 | **653（2.01%）**，原因分布：`supersaturated` 653（无其他违例） |
| 筛检后重新校验 | 0 违例（31 795 通过单元） |
| 物理基底 mass 漂移 | 2.9e-16（周期边界，求解器诊断） |
| 数据变量 | t/q/u/v（4D）+ tcwv（3D，物理基底，符合 schema） |

## 4. E3 四组对比（真实 / 统计 / 物理 / 混合）

完整数值与判定见 [`docs/EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) E3 节与 `output/experiments/summary.json`；定性结论：

1. **物理模式（Mode 2）**：物理类别 PASS（质量守恒 ~1e-16、无负值/非有限），但 statistical/temporal/spatial 无同变量真实参考 → NOT_RUN；它输出的是示踪浓度，不是气象要素场。
2. **统计模式（Mode 1）**：边缘统计 7/8 PASS、时空指标全面 FAIL（计划预设），且**饱和违例 3.3%（1066/32448）**。
3. **混合模式（Mode 3）**：以物理基底替换 tcwv 后，物理筛检拒绝 2.01% 并保证交付数据 0 物理违例；时空指标相对统计模式有可测改善（如 `spatial.variogram_ratio`、`spatial.gradient_rms_ratio`、`temporal.diff_var_ratio` 均下降），但**边缘统计指标变差**（`ks`、`mean_ratio`、`column_shapes`），原因是条件化改变了剩余变量的边缘分布、且物理基底的 tcwv 分布与真实分布不同（扩散平滑导致）。这是 Mode 3 的真实权衡，不是实现缺陷。
4. **结论**：混合模式的价值在"物理一致性 + 部分时空结构"，代价是边缘统计拟合度；若追求边缘一致性应使用 Mode 1，若追求物理一致应使用 Mode 2/筛检后的 Mode 3。报告不预设哪种模式"更好"。

## 5. 偏差决策

| # | 事项 | 实际做法 | 理由/证据 |
|---|---|---|---|
| P1 | 条件化的物理基底在层间共享 | 同一 2D 物理基底同时作为三个层模型的 known 列 | tcwv 是整层积分量（schema 规定 3D，不得伪造 4D），层间共享物理上一致；t/q/u/v 仍按层分别生成 |
| P2 | 时间轴 | 混合输出沿用真实数据前 16 个时次标签（索引式映射），物理演化步长 dt_s 独立 | 求解器稳定步长（CFL）与真实 6 h 步长无法同时满足（u0=8 m/s 时 CFL=1.8）；标签映射使 E3 可比，契约 notes 明确说明不构成物理时钟对应 |
| P3 | 拒绝 vs 修正 | 拒绝单元**遮罩为 NaN**而不是删除（保持规范网格），并写 `rejected.jsonl`；契约声明 `mask_applied` | 规则网格不能有洞；物理验证据此区分"筛检遮罩"与"坏值"（`VALIDATION_GUIDE` §6） |
| P4 | 筛检规则阈值 | 复用 `validation/physical.DEFAULT_THRESHOLDS`（t 180–340 K、风 150 m/s、tcwv 120、饱和 1.1×） | 与四类验证口径一致，避免"生成用一套、验证用另一套" |
| P5 | 条件采样实现 | 直接用 SDV `sample_remaining_columns`（每层一个调用） | Phase 2 实测：已知列逐行 bit 级保留、行序一致、速度约 100 行/s；32k 样本约 7 分钟 |

## 6. 未完成 / NOT_RUN（如实标注）

- **E4 Sim-to-Real**：无下游任务与真实观测对照 → NOT_RUN（不硬凑指标）
- **`equation_residual` 阈值定标**：仍为 INFO（未归一化，见 `PHASE3_REPORT` §7）
- **条件生成边缘分布代价**：未做分布重映射（quantile mapping）或条件增强；作为 Phase 5 之后的可选工作记录在案
- **`hybrid.screening.reject_ratio_limit`**：配置项已解析但**未强制**（当前只记录拒绝率，超过 0.2 不报错）；如需硬门槛应改配置校验（记录为已知缺口）
- **E3 的完整指标表**：数值见 EXPERIMENT_REPORT（由 `scripts/run_experiments.py` 生成）

## 7. Phase 5 开工清单

1. `scripts/run_experiments.py` 汇总 E1/E2/E3（统一条件、质量、耗时、内存）→ `output/experiments/summary.json`
2. Test D：四类质量报告汇总到 `docs/EXPERIMENT_REPORT.md`
3. Test E：固定种子二次运行一致性（数值级对比，非 bitwise）
4. Test F：单脚本真实全链路 `scripts/run_e2e.sh`
5. 文档 8 件套补齐（ARCHITECTURE / GENERATION_GUIDE / VALIDATION_GUIDE / EXPERIMENT_REPORT / README）
6. 全量测试与覆盖率说明、依赖与许可证清单复核

## 8. 复现命令

```bash
export PATH=/opt/miniconda3/envs/atmosgen/bin:$PATH
cd "Atmospheric Data Generation"
atmosgen generate --config configs/hybrid.yaml
ls output/hybrid_hybrid_*/{data.nc,contract.json,metadata.json,rejected.jsonl,quality_report.json}
atmosgen evaluate --input output/hybrid_hybrid_*/data.nc --real data/processed/era5_local.nc --overwrite --fail-on never
```
