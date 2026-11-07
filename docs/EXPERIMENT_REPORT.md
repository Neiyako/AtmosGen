# 实验报告（Phase 2 初稿）

- 日期：2026-11-07
- 数据：`data/processed/era5_local.nc`（2010-05-01..04，16 时次 × 3 层 × 101×101，t/q/u/v + 派生 tcwv）
- 训练/生成网格：`data.subsample = 1°` → 26×26，表格 16×3×26×26 = **32 448 行**（cell_time）
- 机器：macOS arm64，8 GB 内存，无独立 GPU（torch 2.2.2 / sdv 2.1.0）
- 状态：**E1 已完成**；E2–E4 属 Phase 3–5，标 NOT_RUN，本文件后续续写。

---

## E1 统计模型对比（gaussian_copula / ctgan / tvae）

统一条件：同一真实子集、同一训练超参（epochs=100、batch_size=500，ctgan 另有 discriminator_steps=1）、
seed=42、按层训练（每层一个模型，见 `docs/PHASE2_REPORT.md` 偏差 P1）。

### 1. 资源与速度（实测）

| 模型 | 拟合耗时 s | 采样耗时 s | fit 峰值 RSS MB | 进程 ru_maxrss MB | 备注 |
|---|---|---|---|---|---|
| gaussian_copula | **1.95** | **0.235** | 440.8 | 445.9 | 最快，主实验默认 |
| ctgan | 136.95 | 0.484 | 453.0 | 454.4 | 最慢（约 70× copula） |
| tvae | 63.08 | 0.269 | 566.5 | 566.5 | 介于两者之间 |

三个模型都远低于计划 §9 的 "单模型 ≤20 min" 预算；CTGAN 采样阶段 RSS 采样值偏低（119.6 MB），
以 `process_max_rss_mb` 为准（进程生命周期峰值，不随阶段重置）。

### 2. 质量指标（真实 vs 生成，阈值见计划 §5.1）

| 指标 | 阈值 | gaussian_copula | ctgan | tvae |
|---|---|---|---|---|
| column_shapes（越高越好） | ≥0.80 | **0.9530** ✅ | 0.8687 ✅ | 0.8827 ✅ |
| column_pair_trends（越高越好） | ≥0.80 | **0.8884** ✅ | 0.6616 ❌ | 0.7350 ❌ |
| ks（越低越好） | ≤0.15 | **0.0871** ✅ | 0.2367 ❌ | 0.2215 ❌ |
| mean_ratio（范围） | 0.85–1.15 | **1.0590** ✅ | 3.1520 ❌ | 1.6616 ❌ |
| std_ratio（范围） | 0.75–1.25 | **1.0934** ✅ | 0.5850 ❌ | 0.5576 ❌ |
| quantile_rmse（越低越好） | ≤0.10 | **0.0552** ✅ | 0.1668 ❌ | 0.1627 ❌ |
| corr_frobenius（越低越好） | ≤0.25 | 1.1121 ❌ | 1.2960 ❌ | 1.1281 ❌ |
| extreme_ratio（范围） | 0.5–2.0 | 1.1026 ✅ | 1.1576 ✅ | 1.0621 ✅ |

**统计模块结论**：gaussian_copula **7/8 项 PASS**，仅 `corr_frobenius` FAIL；CTGAN 3/8 项 PASS；
TVAE 4/8 项 PASS。整体状态按计划 §5.5 "最低状态" 聚合为 FAIL（非 bug，见分析 §3）。

### 3. 时空指标（真实 vs 生成，阈值见计划 §5.2/§5.3）

| 指标 | 阈值 | gaussian_copula | ctgan | tvae |
|---|---|---|---|---|
| acf_diff | ≤0.20 | 0.5025 ❌ | 0.4948 ❌ | 0.5038 ❌ |
| diff_var_ratio | 0.5–2.0 | 338.17 ❌ | 280.67 ❌ | 236.05 ❌ |
| spectral_slope_diff | ≤0.50 | 1.8526 ❌ | 1.9191 ❌ | 1.8293 ❌ |
| variogram_ratio | 0.5–2.0 | 16.57 ❌ | 14.62 ❌ | 16.30 ❌ |
| gradient_rms_ratio | 0.5–2.0 | 6.86 ❌ | 5.89 ❌ | 6.39 ❌ |
| neighbor_ks | ≤0.20 | 0.3638 ❌ | 0.3683 ❌ | 0.3586 ❌ |
| timeseries_efficacy | — | NOT_RUN | NOT_RUN | NOT_RUN |

**结论**：统计模型的 cell_time 独立采样**不具备**时间/空间连续性，全部时空指标 FAIL ——
这正是计划 §4.1（"时空连续性不承诺自动成立"）与风险 X2 预设的结果，不是实现缺陷；
不得在报告中把它写成 PASS。`timeseries_efficacy`（sdmetrics LSTM 判别）在 Phase 2 默认
NOT_RUN（`validation/temporal.py::timeseries_efficacy` 已实现、可显式调用，Phase 4/5 再启用）。

### 4. 分析与不预设结论的差异

1. **gaussian_copula 在 8 项统计指标上全面优于两个深度模型**，且快 35–70 倍。原因是本题表格是
   单峰、强相关、每层仅有 10 816 行的连续数值表，Copula 的统计假设（近似多元高斯 copula）成立；
   CTGAN/TVAE 在 100 epochs、CPU/MPS、每层万行规模下欠拟合（mean/std 偏差明显），这不是
   "深度模型一定差" 的普适结论，只是当前数据规模与算力下的实测结果。
2. **corr_frobenius 三模型全部 FAIL**，且数值接近（1.11–1.30）：该指标比较的是"跨层变量相关
   矩阵"（t@500 与 t@700 等的相关），而按层训练在构造上就不建模层间相关，因此必然 FAIL。
   这是 Phase 2 方案的可解释结果（见 PHASE2_REPORT 偏差 P1）。
3. CTGAN 的 `mean_ratio=3.15` 由个别列主导（均值接近 0 的 u/v 在比率指标下会被放大），
   逐列细节见各 run 的 `quality_report.json` 的 `detail` 字段，报告不隐藏该口径缺陷。

### 5. Synthcity 隔离环境时间盒（Phase 2 交付项）

- 环境：`atmosgen-synthcity`（Python 3.10.22，numpy 1.26.4，torch 2.2.2，synthcity 27 个插件可用）
- 结论：**PASS（集成）**。`Plugins().get("ctgan", n_iter=5)` 在 200 行小表上 fit 0.63 s、
  generate 0.03 s，产出 200 行；但 n_iter=5 未调参时边缘分布质量差（最大相对均值偏差 67.2、
  t 的生成 std 退化到 5.7e-14），**质量结论 NOT_RUN（未调参，不作为模型对比依据）**。
- 证据：`research/env/logs/p4_synthcity_install.log`、`research/env/logs/p4_synthcity_probe.{log,json}`
- 依赖兼容性记录：见 `docs/DEPENDENCY_COMPATIBILITY.md` §Synthcity 时间盒结论（含两次失败与修复）

## E2 物理参数/IC 敏感性（Phase 3 完成）

求解器先通过解析/守恒/稳定性验证（`scripts/validate_solver_analytic.py`全 PASS，证据 `research/physics/solver_validation_report.json`）：

| 验证 | 实测 | 阈值 |
|---|---|---|
| 纯扩散解析解 L2 误差（dx 20/10/5 km） | 1.84e-3 / 4.50e-4 / 1.12e-4，收敛阶 **2.032 / 2.008** | 记录制（理论二阶） |
| 平流质心位移误差 | x **1.70%**（120 km 位移）、y 4.4e-5% | <2% |
| 周期质量守恒 | 最大相对漂移 **2.86e-16**（/步同量级） | <1e-6/步 |
| 最大 CFL（1.0）稳定性 | 100 步有限、峰值 0.984、无增长 | 无爆炸 |
| 非法 CFL 拒绝 | CFL 1.429 → PhysicsError（报错含修复建议） | 必须拒绝 |

> 过程修复：2D 扩散稳定性判据原写错（κ·dt/min(dx,dy)²≤0.5 允许每轴 0.4、总 0.8 → 细网格爆炸，先被收敛阶=-170 抓出），已改为 von Neumann 总和判据 κ·dt·(1/dx²+1/dy²)≤0.5（详见 PHASE3_REPORT §3）。

Test C 参数敏感性（base：u0=8、v0=2、κ=1e5、σ=1.5°、中心 [30,110]，save 24 场）：

| 场景 | 输出差异 vs base（可量化） |
|---|---|
| IC 中心 [32,112] | 质心位移 **301 010.4 m**（=平移距离，符合恒风平流的线性）；σ 比几乎不变 |
| κ=5e5（×5） | σ_x 比 **1.588**、σ_y 比 **1.860**（扩散 σ∝√(2κt)） |
| u0=20, v0=−4 | 质心位移 **388 121.2 m**（角位移=|Δv|·T），σ_x 比 5.199（风切变致展宽） |

Mode 2 端到端 run：`output/physics_advection_diffusion_20261107-201137_42/`——Solver CFL_x=0.114、扩散数=0.068、mass 漂移 2.86e-16；`evaluate`（无 --real）→ `physical=PASS`（finite/c_nonneg/mass 全 PASS，equation_residual RMS **5.45e-07** INFO），statistical/temporal/spatial NOT_RUN（无真实同网格参考，不硬凑）。

**接入 physical 验证发现（Phase 2 数据回评）**：统计模型 run 的 `saturation` FAIL（copula 32448 点中 **1066 点** q>1.1×q_sat，即 3.3% 超饱和组合）→ 这是 Phase 4 混合模式物理筛检（plan §4.3）的直接动机。

## E4 Sim-to-Real

**NOT_RUN**：Phase 2 没有下游任务（无真实观测对照的预测任务），不硬凑指标；按计划 §8 标 NOT_RUN。

## 复现命令

```bash
export PATH=/opt/miniconda3/envs/atmosgen/bin:$PATH
cd "Atmospheric Data Generation"
atmosgen generate --config configs/stat_gaussian_copula.yaml
atmosgen generate --config configs/stat_ctgan.yaml
atmosgen generate --config configs/stat_tvae.yaml
for run in output/statistical_*; do
  atmosgen evaluate --input "$run/data.nc" --real data/processed/era5_local.nc
done
```
