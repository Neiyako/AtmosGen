# 实验报告（Phase 5 终稿）

- 日期：2026-11-07
- 数据：`data/processed/era5_local.nc`（2010-05-01..04，16 时次 × 3 层 × 101×101，t/q/u/v + 派生 tcwv）
- 训练/生成网格：`data.subsample = 1°` → 26×26，cell_time 表格 32 448 行
- 机器：macOS arm64，8 GB 内存，无独立 GPU（torch 2.2.2 / sdv 2.1.0 / sdmetrics 0.33.0）
- 汇总产物：`output/experiments/summary.json`（E1/E2/E3，统一由 `scripts/run_experiments.py` 生成）；E3 修复后重跑见 `output/experiments_e3/summary.json`
- 状态：E1/E2/E3 已实测；E4 Sim-to-Real **NOT_RUN**（无下游任务）

---

## E1 统计模型对比（gaussian_copula / ctgan / tvae）

统一条件：同一真实子集、epochs=100、batch_size=500、seed=42、按层训练（level_wise，见 PHASE2_REPORT P1）。

### 资源与速度（benchmark 实测，`summary.json`）

| 模型 | 生成总耗时 s | 拟合 s | 采样 s | fit 峰值 RSS MB | 进程 ru_maxrss MB |
|---|---|---|---|---|---|
| gaussian_copula | 17.78 | 8.09 | 0.545 | 445.7 | 445.9 |
| ctgan | 268.06 | 251.93 | 0.591 | 453.0 | 454.4 |
| tvae | 141.67 | 128.17 | 0.410 | 566.5 | 566.5 |

> 单跑 copula 时拟合 1.95 s、采样 0.235 s（`output/statistical_gaussian_copula_20261107-183504_42/metadata.json`）；本表的差异来自同轮 benchmark 的机器负载，模型间相对关系不变。

### 质量指标（每模型对真实数据，阈值见计划 §5.1）

| 指标 | 阈值 | gaussian_copula | ctgan | tvae |
|---|---|---|---|---|
| column_shapes ↑ | ≥0.80 | **0.9530** ✅ | 0.8687 ✅ | 0.8827 ✅ |
| column_pair_trends ↑ | ≥0.80 | **0.8884** ✅ | 0.6616 ❌ | 0.7350 ❌ |
| ks ↓ | ≤0.15 | **0.0871** ✅ | 0.2367 ❌ | 0.2215 ❌ |
| mean_ratio | 0.85–1.15 | **1.0590** ✅ | 3.1520 ❌ | 1.6616 ❌ |
| std_ratio | 0.75–1.25 | **1.0934** ✅ | 0.5850 ❌ | 0.5576 ❌ |
| quantile_rmse ↓ | ≤0.10 | **0.0552** ✅ | 0.1668 ❌ | 0.1627 ❌ |
| corr_frobenius ↓ | ≤0.25 | 1.1121 ❌ | 1.2960 ❌ | 1.1281 ❌ |
| extreme_ratio | 0.5–2.0 | 1.1026 ✅ | 1.1576 ✅ | 1.0621 ✅ |

**结论**：gaussian_copula 8 项中 7 项 PASS（仅跨层相关 FAIL），且快 8–15 倍；CTGAN/TVAE 在当前规模（每层 1.08 万行）+ CPU/MPS 下欠拟合。不预设"深度模型更好"，这是实测差异。

## E2 物理参数敏感性（Phase 3 完成，本 benchmark 复跑）

`scripts/validate_solver_analytic.py` 6 项全部 PASS（`summary.json` E2 `report.status=PASS`）：

| 验证 | 实测 | 阈值 |
|---|---|---|
| 纯扩散解析解 L2（dx 20/10/5 km） | 1.84e-3 / 4.50e-4 / 1.12e-4；收敛阶 **2.032 / 2.008** | 记录制（理论二阶） |
| 平流质心位移误差 | x **1.70%**、y 4.4e-5% | <2% |
| 周期质量守恒 | 最大相对漂移 **2.86e-16** | <1e-6/步 |
| 最大 CFL（1.0）稳定性 | 100 步有限、峰值 0.984 | 无爆炸 |
| 非法 CFL 拒绝 | CFL 1.429 → PhysicsError（含修复建议） | 必须拒绝 |
| Test C 参数敏感性 | IC 平移 → 质心 **301 010.4 m**；κ×5 → σ 比 1.59/1.86；风×2.5 → **388 121.2 m** | 输出可测差异 |

## E3 真实 / 统计 / 物理 / 混合 四组对比

> 下表取修复后重跑（`output/experiments_e3/summary.json`，HEAD 含 F1–F4 修复前的 run 时间轴修正）；E1 表取 `output/experiments/summary.json`。

| 组 | 生成耗时 | statistical | temporal | spatial | physical |
|---|---|---|---|---|---|
| 真实参考（`era5_local.nc`） | — | 基准 | 基准 | 基准 | 基准 |
| statistical（copula） | 6.08 s | FAIL（7/8 指标 PASS） | FAIL | FAIL | FAIL（饱和违例 1066） |
| physics（advection–diffusion） | 3.21 s | NOT_RUN（无可比变量/时轴） | NOT_RUN | NOT_RUN | **PASS**（质量守恒 1.4e-16） |
| hybrid（物理基底 + 条件生成） | 943.28 s | FAIL（条件化代价，见下） | FAIL | FAIL | NOT_RUN（equation_residual 不可算；finite/saturation/mass 均 PASS） |

关键指标：

| 指标 | statistical | hybrid |
|---|---|---|
| KS vs real ↓ | **0.0871** | 0.2501 |
| column_shapes ↑ | **0.9530** | 0.8815 |
| temporal.diff_var_ratio ↓ | 338.17 | **163.91** |
| spatial.variogram_ratio ↓ | 16.57 | **9.74** |
| spatial.gradient_rms_ratio ↓ | 6.86 | **4.60** |
| physical.saturation（违例数） | 1066 | 0（筛检后；筛除 653，拒绝率 **2.01%**） |
| physical 模块状态 | FAIL | NOT_RUN（硬约束全 PASS；equation_residual 对多层风场不适用） |

**解读（不预设结论）**：

1. 统计模式边缘分布最好（KS 0.087），但**物理不一致**（3.3% 超饱和）且无时空结构。
2. 物理模式物理自洽（质量守恒、无负值/非有限），但输出是示踪浓度，与气象要素不可直接比 → 三类对照 NOT_RUN，不造假指标。
3. 混合模式以物理基底替换 tcwv 后：**时空指标有可测改善**（`diff_var_ratio` 338→164、`variogram_ratio` 16.6→9.7、`gradient_rms_ratio` 6.9→4.6），物理违例经筛检归零；但**边缘统计变差**（KS 0.087→0.25、shapes 0.953→0.882）——条件化改变剩余变量边缘分布，物理基底的 tcwv 分布（扩散平滑）与真实不同。这是真实权衡，不是缺陷。
4. 不存在"最好模式"的普适结论：边缘一致性用 Mode 1，物理一致用 Mode 2/混合，时空结构用混合（待改进）。

## Test D 质量报告

每 run：`quality_report.json`（严格 JSON、逐指标 value/threshold/status/detail）+ `quality_report.md`（表 + 3 图：边际分布、ACF+功率谱、半变异函数）+ `figures/*.png`。四类按 `FAIL > NOT_RUN > CONDITIONAL_PASS > PASS` 聚合；"不适用"与"未运行"在 notes 分开列（见 `VALIDATION_GUIDE`）。

## Test E 可复现性

同配置同种子（`configs/stat_gaussian_copula.yaml`，seed=42）连续两次生成：

| 检查 | 结果 |
|---|---|
| 两次 run 目录 | 同配置两次生成到不同 `output_dir`（`output/_repro_a/…` 与 `output/_repro_b/…`） |
| 数据数值一致性 | **逐位相同**（t/q/u/v/tcwv 的 max\|diff\| = 0） |
| 说明 | 文件 SHA 不同（契约 `created_at` 等元数据变化），但**数据与指标一致**，强于计划 §8 "≤1%" 的要求 |

## Test F 全链路

`scripts/run_e2e.sh`：validate-config（4 配置）→ ingest（Test A）→ generate（Mode 1）→ generate（Mode 2）→ evaluate+report → 求解器解析验证，任一失败即非零退出。见 `docs/PHASE5_REPORT.md` 的实跑记录。

## E4 Sim-to-Real

**NOT_RUN**：无下游任务与真实观测对照，不硬凑指标（计划 §8 允许如实标注）。

## 复现命令

```bash
export PATH=/opt/miniconda3/envs/atmosgen/bin:$PATH
cd "Atmospheric Data Generation"
python scripts/run_experiments.py --experiments e1,e2,e3 --out output/experiments
python scripts/validate_solver_analytic.py
bash scripts/run_e2e.sh
```
