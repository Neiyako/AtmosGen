# Phase 2 报告 · 统计生成（SDV）+ 统计/时序/空间评价

- 日期：2026-11-07
- 依据：[`docs/IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §7 Phase 2 验收表、§4.1、§5.1–5.3、§6.1/§6.2；D3/D4/D6；风险 X1/X2
- 结论：**Phase 2 交付项全部完成**（Test B 通过；E1 完成；Synthcity 时间盒 PASS-集成 / 质量 NOT_RUN）。生成质量按实测如实报告：统计边缘分布 7/8 指标 PASS，**时序/空间指标按计划预设 FAIL**（不是实现缺陷，不允许写成 PASS）。

---

## 1. 一句话结果

`atmosgen generate --config configs/stat_gaussian_copula.yaml` 在 **1.95 s 拟合 + 0.235 s 采样**内产出 `output/statistical_gaussian_copula_20261107-183504_42/data.nc`（16 时次 × 3 层 × 26×26，32 448 行 → 完整场，契约 14 字段齐全）；`atmosgen evaluate --real data/processed/era5_local.nc` 给出 8 项统计指标中 **7 项 PASS**（仅跨层相关 `corr_frobenius` FAIL）、时序与空间指标 FAIL（计划 §4.1/X2 预设）。同一链路跑通 CTGAN（137 s）与 TVAE（63 s）。`pytest -q` **96 passed**（Phase 2 时点；Phase 5 终态 140）。

## 2. Phase 2 验收表（对照计划 §7）

| 验收项 | 状态 | 证据 |
|---|---|---|
| 场↔表转换可逆（允许数值精度误差） | ✅ PASS | `tests/test_tabular.py`：t/q/u/v 逐值精确；3D tcwv 按层平均重建、1 ULP 内（atol=1e-12）；行序 (time,level,lat,lon) 语义测试；`tests/test_selection.py` |
| SDV Adapter（≥2 模型；条件生成） | ✅ PASS | `atmosgen/generators/sdv_adapter.py` 三模型；`tests/test_sdv_adapter.py` 7 用例；条件生成入口 `sample_conditional()` 已实现（Phase 4 用），其局限见偏差 P1 |
| 统计质量（sdmetrics + 自定义） | ✅ PASS | `atmosgen/validation/statistical.py` 8 指标；真实数据实测见 §3 |
| 时序/空间质量 v1 | ✅ PASS | `temporal.py` 5 指标 + `timeseries_efficacy`（默认 NOT_RUN）；`spatial.py` 6 指标；实测见 §4 |
| generate / evaluate / report CLI | ✅ PASS | `tests/test_cli.py::test_generate_statistical_end_to_end` 端到端（generate→evaluate→report） |
| 实验 E1 初稿 | ✅ PASS | [`docs/EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) |
| Synthcity 隔离环境探测（时间盒） | ✅ PASS（集成）/ ⚠️ 质量 NOT_RUN | `research/env/logs/p4_synthcity_*.{log,json}`；见 §6 |
| **Test B**：生成→保存→重读一致 | ✅ PASS | `tests/test_sdv_adapter.py::test_test_b_roundtrip_save_and_reread`（逐值 `assert_array_equal` + 契约重读）；真实数据产物 `output/statistical_*/data.nc` + `contract.json` 侧车 |

Agent/模块测试规模：`tests/` 共 **96 passed / 12.0 s**（契约 10、IO 9、配置 14、ingest 8、registry 5、CLI 12、tabular 9、selection 4、SDV adapter 7、validation 18）。

## 3. 统计质量（真实 vs 生成，阈值见计划 §5.1）

| 指标 | 阈值 | gaussian_copula | ctgan | tvae |
|---|---|---|---|---|
| column_shapes ↑ | ≥0.80 | **0.953** ✅ | 0.869 ✅ | 0.883 ✅ |
| column_pair_trends ↑ | ≥0.80 | **0.888** ✅ | 0.662 ❌ | 0.735 ❌ |
| ks ↓ | ≤0.15 | **0.087** ✅ | 0.237 ❌ | 0.222 ❌ |
| mean_ratio | 0.85–1.15 | **1.059** ✅ | 3.152 ❌ | 1.662 ❌ |
| std_ratio | 0.75–1.25 | **1.093** ✅ | 0.585 ❌ | 0.558 ❌ |
| quantile_rmse ↓ | ≤0.10 | **0.055** ✅ | 0.167 ❌ | 0.163 ❌ |
| corr_frobenius ↓ | ≤0.25 | 1.112 ❌ | 1.296 ❌ | 1.128 ❌ |
| extreme_ratio | 0.5–2.0 | 1.103 ✅ | 1.158 ✅ | 1.062 ✅ |

- `gaussian_copula`：**7/8 PASS**；`corr_frobenius` 三模型均 FAIL——该指标比较跨层相关矩阵，而按层训练在构造上不建模层间相关（见 P1）。
- 统计模块整体状态按计划 §5.5「最低状态」聚合为 FAIL；这是阈值口径的如实结果，不隐藏。
- 各 run 的 `quality_report.md` 现已包含计划 §5.5 要求的图（`figures/distributions.png`、`figures/acf_spectrum.png`、`figures/variogram.png`，每 run 3 张）。

## 4. 时序 / 空间质量（计划 §5.2 / §5.3）

| 指标 | 阈值 | copula | ctgan | tvae |
|---|---|---|---|---|
| acf_diff ↓ | ≤0.20 | 0.503 ❌ | 0.495 ❌ | 0.504 ❌ |
| diff_var_ratio | 0.5–2.0 | 338.2 ❌ | 280.7 ❌ | 236.1 ❌ |
| spectral_slope_diff ↓ | ≤0.50 | 1.853 ❌ | 1.919 ❌ | 1.829 ❌ |
| variogram_ratio | 0.5–2.0 | 16.57 ❌ | 14.62 ❌ | 16.30 ❌ |
| gradient_rms_ratio | 0.5–2.0 | 6.86 ❌ | 5.89 ❌ | 6.39 ❌ |
| neighbor_ks ↓ | ≤0.20 | 0.364 ❌ | 0.368 ❌ | 0.359 ❌ |
| timeseries_efficacy | — | NOT_RUN | NOT_RUN | NOT_RUN |

结论：cell_time 独立采样**没有**时间/空间连续性；该结果与计划 §4.1（"时空连续性不承诺自动成立"）和风险 X2 一致。不得写成 PASS，也不得靠混入真实数据"美化"。

## 5. 资源实测（`metadata.json`，后台 50 ms 采样 + ru_maxrss 交叉核对）

| 模型 | fit s | sample s | fit 峰值 RSS MB | 进程 ru_maxrss MB |
|---|---|---|---|---|
| gaussian_copula | 1.95 | 0.235 | 440.8 | 445.9 |
| ctgan | 136.95 | 0.484 | 453.0 | 454.4 |
| tvae | 63.08 | 0.269 | 566.5 | 566.5 |

均在计划 §9 预算（单模型 ≤20 min）内；无 OOM（8 GB 机器）。

## 6. Synthcity 隔离环境时间盒（D6）

- 环境：`atmosgen-synthcity`（Python 3.10.22 / numpy 1.26.4 / torch 2.2.2 / synthcity 27 插件）
- 安装期两次失败与修复：scipy 1.16 wheel 在该 macOS 上 dlopen 失败（`__DATA/__thread_bss` 报错）→ 降 `scipy==1.11.4`；opacus ≥1.5 需要 torch≥2.6 的 `nn.RMSNorm` → 降 `opacus==1.4.1`。详见 [`docs/DEPENDENCY_COMPATIBILITY.md`](DEPENDENCY_COMPATIBILITY.md) §10。
- Probe 结论：**集成 PASS**（import / 27 插件 / `ctgan` fit 0.63 s + generate 0.03 s / 200 行）；**质量 NOT_RUN**（`n_iter=5` 未调参，最大相对均值偏差 67.2，t 生成 std 退化 5.7e-14）——不作为模型对比依据，也不启用为默认生成器。
- 证据：`research/env/logs/p4_synthcity_install.log`、`p4_synthcity_probe.log`、`p4_synthcity_probe.json`、`scripts/synthcity_probe.py`

## 7. 偏差决策（按 PHASE1_REPORT §6 格式）

| # | 事项 | 实际做法 | 理由/证据 |
|---|---|---|---|
| P1 | 条件生成方案 | 采用 **level_wise**（每层一个模型），条件协变量采样仅保留 API 供 Phase 4 | 实测 SDV 2.1：连续协变量条件采样在经验分布边界退化——用 `sample_remaining_columns(level=500 等)` 时 t 被压到 248.0 K ± 0.02 K（真实 262.3 ± 5.0），且 32 448 行耗时 148 s；CTGAN/TVAE 条件采样是拒绝采样（约 1 接受/100 次、约 6 s/行）。两条路都不适合全网格 |
| P2 | `generation.covariates` | 配置中的 hour_sin/hour_cos/level/lat_norm/lon_norm 照常校验并写入契约 `covariates_configured`，但 Phase 2 只有 level 作为分组维度真正参与 | 同上；契约 `covariates_not_used` 明确列出未用项，不假装修建了条件结构 |
| P3 | sdmetrics QualityReport API（D4 偏差） | 使用 `sdmetrics.reports.single_table.QualityReport`（弃用告警） | 实测 sdmetrics 0.33.0 非弃用 `sdmetrics.reports.QualityReport` 是 multi-table 口径：单表包装 `{"tables": {...}}` 时 `Column Pair Trends` 的 `Synthetic Correlation` 全为 NaN（探针见报告 §7 说明）。升级条件：sdmetrics 修复该路径后切换 |
| P4 | Synthcity 依赖 pin（D6） | 隔离环境必须额外 pin `scipy==1.11.4`、`opacus==1.4.1` | 实测安装日志（两次失败原文见 DEPENDENCY_COMPATIBILITY §10） |
| P5 | 总体质量状态 | 统计/时序/空间全部按实测报 FAIL（copula 统计 7/8 PASS，模块仍按最低项聚合 FAIL） | 计划 §5.5 聚合口径 + X2 预设；不因"大部分 PASS"就改总体状态 |
| P6 | 变量集 | 仅 t/q/u/v/tcwv；clwc/ciwc 等云水变量未参与 | 主数据集无云水（Phase 1 结论）；hybrid（Phase 4）需先建合并数据集 |
| P7 | `samples.rows` 语义 | 作为下限：实际生成量 = max(配置 rows, 目标网格行数)，契约同时记录 rows_requested / rows_generated | 重建完整场必须填满网格；避免"20000 样本 = 20000 场"的语义错误（计划 D12） |
| P8 | `evaluate` 退出码 | 新增 `--fail-on fail`（默认）：质量 FAIL → 退出 4；`--fail-on never` → 总是 0 | 计划 §6.2 要求"退出码 0/非 0 语义明确"；独立复核 D1 指出 FAIL 时退出 0 会让 CI 放行 |
| P9 | 报告互操作与覆盖 | ① 非有限浮点写为 JSON `null`（严格解析器可读；`NaN` 非法）② `physical` 显式 NOT_RUN 块（可区分"未实现"与"通过"）③ 每 run 3 张图 | 独立复核 D2/D4/D6；计划 §5.5 要求"四类最低状态 + NOT_RUN 显著列出"、md 含图 |

## 8. 未完成 / NOT_RUN（如实标注）

- **physical 验证**（计划 §5.4）：Phase 3 实现，当前 `validation/` 无物理约束模块
- **timeseries_efficacy**：函数已实现但 Phase 2 默认关闭（sdmetrics LSTM 判别，成本高），Phase 4/5 启用
- **E2/E3/E4**：Phase 3/4/5；E4 Sim-to-Real 无下游任务 → NOT_RUN
- **crwc/cswc**：需 CDS 时间盒下载，仍未获取（Phase 1 遗留）
- 源归档 >64 MiB 文件的全文件 SHA-256：仍只做 size+mtime 指纹（Phase 1 P4）
- CTGAN/TVAE 超参未调优（epochs=100 固定）：X1 风险，实测质量低于 Copula；未做调参追平（不预设结论）
- 采样峰值 RSS 为阶段采样值，进程级峰值见 ru_maxrss 列

## 9. Phase 3 开工清单

1. `generators/physics/advection_diffusion.py`：显式 Euler + 一阶迎风 + 5 点中心扩散；CFL/扩散稳定性检查（配置已支持，见 `config.py::PhysicsConfig._check_cfl`）
2. `scripts/validate_solver_analytic.py`：纯扩散对高斯解析解，3 个分辨率记录收敛阶；平流位移测试（误差 <2%）；周期边界质量守恒（<1e-6/步）
3. IC/BC 实现：`gaussian_plume` / `from_real`；`periodic` / `open` / `fixed_from_real`
4. `validation/physical.py`：硬约束（有限/非负/温度/风速/饱和/tcwv 范围）+ 软指标（质量守恒/方程残差/稳定性/平滑性）
5. `atmosgen generate --config configs/physics_advection.yaml` 端到端（替换当前 NOT_RUN 分支）+ Test C（改 IC/参数 → 输出可测差异）
6. 更新 EXPERIMENT_REPORT E2 / README / 新增 PHASE3_REPORT

## 11. 独立验收（子代理复核）与修复记录

Phase 2 完成后由独立子代理（只读复核，不采信仓库文字）复核，首轮判定 **CONDITIONAL_PASS**：核心主张（96 passed、Test B、契约字段、level_wise 逐层放置、copula 指标与资源数值）全部现场复现成立；指出 3 个中级缺陷与 3 个次要问题。本轮已全部修复并回归：

| ID | 问题 | 修复 | 证据 |
|---|---|---|---|
| D1 | `evaluate` 在 `overall_status=FAIL` 时仍退出 0，与 README/计划退出码约定矛盾 | 新增 `--fail-on fail`（默认）：FAIL → 退出 4；`--fail-on never` → 0 | `atmosgen evaluate …` → `DEFAULT_FAIL_ON_EXIT=4`；`--fail-on never` → 0；单测 `test_evaluate_fail_status_exits_4` |
| D2 | `quality_report.json` 含裸 `NaN`（来自 `timeseries_efficacy.value`），不是合法 JSON | `runner.write_json` 递归把 NaN/Inf 转 `null`，并 `allow_nan=False` 强制校验 | 严格 JSON 解析（`parse_constant` 拦截）对 3 个 run 全部 OK |
| D3 | §3 表 ctgan/tvae 的 `extreme_ratio` 数字与产物不符（写作 1.167/1.274，实测 1.1576/1.0621） | 按实测修正 §3 与 EXPERIMENT_REPORT 表；全表 24 格重新核对 24/24 一致 | `quality_report.json` 的 `statistical.metrics.extreme_ratio.value` |
| D4 | 计划 §5.5 要求 md 含图，原实现无图 | `evaluation._write_figures()`：分布对比 / ACF+功率谱 / 半变异函数，每 run 3 张 PNG，md 追加图链接 | 3 个 run 的 `figures/` 与 `quality_report.md` 实测 |
| D5 | 场↔表往返单测用 nlevel=2，"精确"结论只在二进制可除尽时成立 | 测试改用 nlevel=3（生产同构）：t/q/u/v 严格相等，tcwv `atol=1e-12` | `tests/test_tabular.py::test_roundtrip_reconstructs_fields_exactly` |
| D6 | 报告只有 three 类模块，`physical` 无 NOT_RUN 块，"未实现"与"通过"无法区分 | `evaluation` 显式写入 `physical: NOT_RUN` 块 + `coverage` 字段；renderer 支持 physical 小节 | `quality_report.json` 的 `physical.status=NOT_RUN`、`coverage.not_implemented` |

复核同时确认无缺陷的项：`aggregate_status` 不会把 CONDITIONAL_PASS/NOT_RUN 升级为 PASS（12 组探针 12/12 符合规格）；evaluate 重跑与原始报告逐键一致（仅 `meta.environment.timestamp` 不同）。

## 10. 复现命令

```bash
export PATH=/opt/miniconda3/envs/atmosgen/bin:$PATH
cd "Atmospheric Data Generation"
pip install -e . --no-deps
atmosgen ingest --config configs/era5_ingest.yaml --overwrite
atmosgen generate --config configs/stat_gaussian_copula.yaml
atmosgen generate --config configs/stat_ctgan.yaml
atmosgen generate --config configs/stat_tvae.yaml
for run in output/statistical_*; do
  atmosgen evaluate --input "$run/data.nc" --real data/processed/era5_local.nc --overwrite --fail-on never
  atmosgen report --input "$run/quality_report.json" --format md --out "$run/quality_report_rendered.md"
done
pytest -q
```
