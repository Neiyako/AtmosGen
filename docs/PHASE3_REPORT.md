# Phase 3 报告 · 物理仿真（2D advection–diffusion + 解析解验证）

- 日期：2026-11-07
- 依据：[`docs/IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §7 Phase 3 验收表、§4.2、§5.4；决策 D7（最小自研求解器）、D8（RealPDEBench 只取指标定义）
- 结论：**Phase 3 交付项全部完成**。求解器通过 6 项解析/守恒/稳定性验证（`scripts/validate_solver_analytic.py` 全 PASS）；Mode 2 端到端产出契约化 NetCDF；`validation/physical.py` 11 项物理指标接入 `evaluate` 四类聚合（24 用例全绿）；Test C 参数敏感性给出可量化差异表。回归 `pytest -q` → **129 passed**。

---

## 1. 交付物对照（计划 §7 Phase 3）

| 验收项 | 状态 | 证据 |
|---|---|---|
| advection–diffusion 求解器（Euler + 迎风 + 中心扩散） | ✅ | `atmosgen/generators/physics/advection_diffusion.py`；`conditions.py`（gaussian_plume / from_real IC，periodic / open / fixed_from_real BC） |
| 解析解收敛（3 分辨率，记录误差与收敛阶） | ✅ | L2 相对误差 **1.84e-3 → 4.50e-4 → 1.12e-4**（dx 20/10/5 km），观测收敛阶 **2.032 / 2.008**（理论二阶） |
| 平流位移测试（误差 <2%） | ✅ | 恒风 u=10 m/s、T=12 000 s：质心位移误差 **x 1.70% / y 4.4e-5%**（<2% 阈值） |
| 质量守恒（<1e-6/步） | ✅ | 周期边界 240 步：最大相对漂移 **2.86e-16**（每步同量级） |
| CFL/扩散数校验在求解器与配置双层生效 | ✅ | `check_stability()` + `config.PhysicsConfig`；非法参数被拒（见 §3 修正①） |
| 最大 CFL 下无数值爆炸 | ✅ | CFL=1.0、纯平流 100 步：有限、峰值 0.984（无增长） |
| Test C：改 IC/参数 → 输出可测差异 | ✅ | §2 表（质心位移 301 km / 1.79 km / 388 km，σ 比 1.59 等） |
| 物理验证（§5.4 硬约束 + 软指标） | ✅ | `validation/physical.py` 11 指标 + `evaluate` 四类聚合；24 用例 |
| Mode 2 端到端（NetCDF + 契约 + 报告） | ✅ | `output/physics_advection_diffusion_20261107-201137_42/`：`data.nc`（24 场）+ `contract.json` + `metadata.json` + `quality_report.{json,md}` |

解析验证的证据文件：`output/solver_validation_*/report.json`（副本存档 `research/physics/solver_validation_report.json`）。

## 2. Test C · 参数/IC 敏感性（base: u0=8, v0=2, κ=1e5, 高斯羽流 σ=1.5°, 中心 [30,110]）

| 场景 | 变化 | 质心位移 vs base | std_x / std_y 比 | mass 漂移 |
|---|---|---|---|---|
| base | — | — | 1 / 1 | 0 |
| shifted_ic | IC 中心 → [32,112]（+301 km 对角） | **301 010.4 m** | 1.0212 / 1.0000 | ~0 |
| strong_diffusion | κ → 5e5（×5） | 1 792.9 m | **1.5882 / 1.8602** | ~0 |
| strong_wind | u0→20, v0→−4（×2.5/−2） | **388 121.2 m** | 5.1986 / 1.1118 | ~0 |

解读：IC 平移 → 质心线性移动（位移=平移距离，误差 0.3%）；κ 加倍扩散 → 两个方向的展宽比 ≈ √5/…（实测 1.59/1.86，符合扩散 σ ∝ √(2κt)）；风场变化 → 质心位移按 u·T 缩放。输出差异均可量化（Test C ✅）。

## 3. 过程中发现并修正的两个真 bug（如实记录）

| # | 问题 | 根因 | 修复 |
|---|---|---|---|
| 修正① | 细网格 dx=5 km 数值爆炸（L2 误差 5.59e+50） | **2D 扩散稳定性判据写错**：只查 κ·dt/min(dx,dy)² ≤ 0.5，而显式 2D 中心扩散的 von Neumann 条件是 κ·dt·(1/dx²+1/dy²) **总和** ≤ 0.5（方格即为每轴 ≤0.25）；原判据允许每轴 0.4（总 0.8），必然爆 | `check_stability` 与 `config.PhysicsConfig` 同步改为总和判据；收敛测试按 0.4/0.5 余量取 dt。该错误先被**收敛阶测试抓出**（负收敛阶 -170），验证流程有效 |
| 修正② | 平流位移测试误差 138% | 位移 360 km 超过周期域半宽 256 km → 羽流绕回，质心测量失去意义 | 测试改为 T=12 000 s（120 km 位移，域内）；**不是**求解器 bug，是测试设计与周期边界语义不符 |

## 4. 物理验证模块（`atmosgen/validation/physical.py`）

- 硬约束：`finite / q_nonneg / cloud_nonneg / t_range / wind_range / saturation(Tetens) / tcwv_range / c_nonneg`，任一违例 → 该项 FAIL；
- 软指标：`mass_budget`（周期总质量漂移，期望 ≤5%）、`smoothness`（棋盘格翻转率 vs real ≤2×）、`equation_residual`（`∂c/∂t+u∂c/∂x+v∂c/∂y−κ∇²c` 离散残差 RMS，Phase 4 设阈值）；
- 聚合语义（plan §5.4/§5.5）：**变量缺失 → 不适用（不参与状态聚合，仍显著列出）**；**软指标 INFO 不判死**；核心状态变量（t/q/u/v/c）全部缺失的数据集 → 整体 NOT_RUN（不得误判 PASS）；
- `equation_residual` 支持 u/v 来自数据集字段或**契约物理参数 u0/v0**（Mode 2 常量风场语境，plan §5.4 原文"需 u,v,κ"）；限定均匀时间步（Mode 2 快照调度已改为等间隔以满足该项）；
- 实测（Mode 2 run）：finite/c_nonneg/mass 全 PASS（mass 1.43e-16），equation_residual **RMS=5.45e-07**（INFO，Phase 4 设阈值）；
- Tetens 已知限制（超出水面适用域的低温点按"非法气压/不适用"记录，不静默放宽）——详见模块 docstring 与 teammate 交付说明。

## 5. 对 Phase 2 统计 run 的新发现（桥接 Phase 4）

接入 physical 模块后重评三个统计 run：**saturation FAIL（如 gaussian_copula：32448 点中 1066 点 q > 1.1×q_sat(t,p)，占 3.3%）**。这不是 bug：统计模型按层学习 (t,q) 联合分布，但在 1.1 倍饱和阈值口径下确实产生超饱和组合——这正是计划 §4.3（Mode 3）"物理硬约束筛检 + rejected.jsonl" 的设计动机。Phase 4 的筛检将以该指标为准并在报告中区分"修正前/修正后"。

## 6. 偏差决策（延续 PHASE1/2 格式）

| # | 事项 | 实际做法 | 理由/证据 |
|---|---|---|---|
| P1 | 契约 schema 1.0 → **1.1** | 新增可选物理追迹变量 `c`（(time,latitude,longitude)，单位 `1`，category=tracer）；旧数据（1.0）读取不受影响 | Mode 2 输出的是浓度场，DATA_SCHEMA 9 个气象变量没有它的位置；按 schema 规则"新增可选 → minor"，`docs/DATA_SCHEMA.md` 已补记 |
| P2 | `fields < steps+1` 快照调度 | `fields < steps+1` 时改为**等间隔**快照（最多 fields 张），并在 diagnostics 警告末段未持久化 | 原 linspace 舍入产生不均匀 dt（2400/2400/3000…s），使任何"均匀时钟"检查失效且物理上时间轴与 dt_s 不符 |
| P3 | 时间轴 | Mode 2 使用合成时钟起点（`2000-01-01T00`），`from_real` 时取真实数据首时次；契约 `initial_conditions.details` 记录 | 物理解没有现实日历含义；度量只关心 Δt |
| P4 | 质心/统计量的米制换算 | 等距圆柱近似（dy=111 320 m/deg，dx=111 320·cos(lat) m/deg） | 有限区域误差可忽略（<1°）；解析解在同一换算下定义，自洽 |

## 7. 未完成 / NOT_RUN（如实标注）

- `hybrid`（Mode 3 / Phase 4）：仍 `atmosgen generate` → exit 3 NOT_RUN
- E3（混合 vs 统计 vs 物理）、E4（Sim-to-Real）：Phase 4/5
- physical 的 `equation_residual` 阈值：Phase 4 定标后写死
- 三层 (level>1) 气压层上的 residual：实现为单层语义，多层 → NOT_RUN（已记录）
- `equation_residual` 未按场量级归一化（单位 c·s⁻¹）：Phase 4 定阈值前必须无量纲化（已在模块 docstring 记录）

## 8. Phase 4 开工清单

1. `atmosgen/hybrid.py`：物理基底（tcwv 或 c）→ 条件生成其余变量 → 硬约束筛检链路（数据合并：需要与云水样本或本数据集跨域组合的决策）
2. sat/sat 屏检 `validation/physical` 的饱和指标复用：拒绝样本写 `rejected.jsonl`（原因+指标），统计拒绝率入契约 quality_metrics
3. `hybrid` 生成器注册表接通 + `atmosgen generate --config configs/hybrid.yaml`
4. 四组对比（真实/统计/物理/混合）同表 E3
5. `equation_residual` 无量纲化与阈值定标

## 9. 复现命令

```bash
export PATH=/opt/miniconda3/envs/atmosgen/bin:$PATH
cd "Atmospheric Data Generation"
python scripts/validate_solver_analytic.py        # 6 项解析/守恒/稳定性 + Test C
atmosgen generate --config configs/physics_advection.yaml
atmosgen evaluate --input output/physics_advection_diffusion_*/data.nc \
                  --overwrite --fail-on never     # physical=PASS
pytest -q                                         # 129 passed
```
