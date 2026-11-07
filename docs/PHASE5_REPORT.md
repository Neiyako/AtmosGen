# Phase 5 报告 · Benchmark、四类报告、可复现性与交付

- 日期：2026-11-07
- 依据：[`docs/IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §7 Phase 5、§8 测试 A–F、§9 实验设计、§11–13 交付物映射
- 结论：**Phase 5 交付完成**。E1/E2/E3 已实测并汇总（`output/experiments/summary.json`）、四类质量报告带图落盘、Test E 达到**逐位复现**（强于 ≤1% 要求）、Test F 单脚本全链路 **PASS**（1m51s）、`pytest` **137 passed**、文档 8 件套齐全。

---

## 1. 交付物对照（计划 §13）

| # | 要求 | 产物 | 状态 |
|---|---|---|---|
| 1 | 完整源码 | `atmosgen/`（20 个模块）、`tests/`（11 个测试文件）、`scripts/` | ✅ |
| 2 | 可安装 Python 项目 | `pyproject.toml`；`pip install -e . --no-deps` → `atmosgen --version` = 0.1.0 | ✅ |
| 3 | 五个项目复用说明 | `docs/THIRD_PARTY_AUDIT.md`、`docs/REUSE_MATRIX.md` | ✅（Phase 0） |
| 4 | 依赖与许可证清单 | `docs/DEPENDENCY_COMPATIBILITY.md`（含 Synthcity 时间盒 §10）+ README 许可证节 | ✅ |
| 5 | 标准配置 | `configs/{demo,era5_ingest,era5_cloud_sample,stat_gaussian_copula,stat_ctgan,stat_tvae,physics_advection,hybrid}.yaml` | ✅ |
| 6 | ≥2 个统计生成示例 | `stat_gaussian_copula` / `stat_ctgan`（+`stat_tvae`） | ✅ |
| 7 | ≥1 个物理仿真实例 | `configs/physics_advection.yaml` + `output/physics_advection_diffusion_*/` | ✅ |
| 8 | ≥1 个混合生成示例 | `configs/hybrid.yaml` + `output/hybrid_hybrid_*/`（含 `rejected.jsonl`） | ✅ |
| 9 | 实际生成测试数据 | `output/statistical_*/data.nc`、`output/physics_*/data.nc`、`output/hybrid_*/data.nc` | ✅ |
| 10 | 质量评价报告 | 每 run `quality_report.{json,md}` + `figures/*.png`；汇总 `docs/EXPERIMENT_REPORT.md` | ✅ |
| 11 | 完整运行文档 | `README.md`、`docs/ARCHITECTURE.md`、`docs/GENERATION_GUIDE.md`、`docs/VALIDATION_GUIDE.md` | ✅ |
| 12 | 自动化测试 | `pytest`：**137 passed**（契约 10 / IO 9 / 配置 14 / ingest 8 / registry 5 / CLI 13 / tabular 9 / selection 4 / SDV 7 / physics 10 / hybrid 6 / validation 18 / physical 24） | ✅ |

文档 8 件套：README、ARCHITECTURE、THIRD_PARTY_AUDIT、REUSE_MATRIX、DATA_SCHEMA、GENERATION_GUIDE、VALIDATION_GUIDE、EXPERIMENT_REPORT；另有阶段报告 PHASE1–PHASE5。

## 2. Benchmark 与实验（Test D 汇总）

`python scripts/run_experiments.py --experiments e1,e2,e3` → `output/experiments*/summary.json`（统一记录模型/耗时/内存/指标/状态）。

| 实验 | 结论（数值见 [`EXPERIMENT_REPORT`](EXPERIMENT_REPORT.md)） |
|---|---|
| E1 统计模型对比 | copula 8 项中 7 项 PASS 且比 CTGAN/TVAE 快 8–15×；深度模型在 CPU/万行级欠拟合 |
| E2 物理参数敏感性 | 6/6 解析/守恒/稳定性验证 PASS；收敛阶 2.03/2.01；Test C 差异可量化（301 km / 388 km / σ 比 1.59） |
| E3 四组对比 | 统计=边缘好+物理差+无时空；物理=物理自洽但不可比；混合=时空改善（variogram 16.6→9.7）+ 筛检后 0 物理违例，代价是边缘统计变差 |
| E4 Sim-to-Real | **NOT_RUN**（无下游任务，不硬凑） |

## 3. Test E 可复现性

同配置同种子两次运行（`configs/stat_gaussian_copula.yaml`）→ 数据数值**逐位一致**（max|diff|=0），优于计划 §8 "≤1%" 要求。
说明：文件 SHA 因契约 `created_at` 等元数据而不同（属预期，见 PHASE1_REPORT P6）；环境/版本记录见每 run `environment.json` 与契约 `metadata`。

## 4. Test F 全链路

`bash scripts/run_e2e.sh`：validate-config（4 配置）→ ingest（Test A）→ generate（Mode 1）→ generate（Mode 2）→ evaluate + report（统计用 `--real`、物理不传 `--real` 并给出理由）→ 求解器解析验证；**退出码 0，实测 1m51s**。

## 5. 状态汇总（不虚报）

| 项 | 状态 |
|---|---|
| 四类质量验证 | statistical/temporal/spatial/physical 全部可用并进 evaluate 聚合 |
| 图表 | 每 run 3 图（边际分布、ACF+功率谱、半变异函数） |
| 可复现性 | PASS（逐位一致） |
| 全链路脚本 | PASS |
| 自动化测试 | 137 passed / 37 s |
| E4 Sim-to-Real | NOT_RUN |
| `equation_residual` 阈值 | NOT_RUN（Phase 4+ 定标，见 PHASE3_REPORT §7） |
| `hybrid.screening.reject_ratio_limit` | 配置项已解析但未强制（缺口记录在 PHASE4_REPORT §6） |
| Synthcity 作为生成器 | 不启用（仅环境 PASS，质量 NOT_RUN，见 DEPENDENCY_COMPATIBILITY §10） |

## 6. 关键偏差与已知限制（跨阶段汇总）

| # | 事项 | 说明 | 出处 |
|---|---|---|---|
| P1 | level_wise 统计训练 | 连续协变量条件采样在 SDV 2.1 边界退化；改为每层一个模型 | PHASE2_REPORT §7 |
| P2 | 混合模式条件化代价 | 条件化改变剩余变量边缘分布、物理基底分布与真实不同 → 边缘统计变差；时空指标改善 | PHASE4_REPORT §4、EXPERIMENT_REPORT E3 |
| P3 | 混合时间轴索引映射 | 求解器稳定步长与真实 6 h 步长不可兼得；time 标签沿用真实前 N 个时次（契约注明） | PHASE4_REPORT P2 |
| P4 | 筛检遮罩 NaN 语义 | 拒绝单元以 NaN 保存 + 契约 `mask_applied` 声明；物理验证据此区分"遮罩"与"坏值" | VALIDATION_GUIDE §6 |
| P5 | sdmetrics 弃用 API | 非弃用 multi-table QualityReport 在单表包装下 Pair Trends 全 NaN（实测），沿用 single_table 实现 | PHASE2_REPORT P3 |
| P6 | 文件级 SHA 不可复现 | 契约含 created_at/耗时，属设计；可复现性以数据/指标为准 | PHASE1_REPORT P6、本报告 §3 |
| P7 | 物理模式与真实不可比 | Mode 2 输出示踪浓度 + 合成时钟 → evaluate 三类对照 NOT_RUN，不伪造指标 | EXPERIMENT_REPORT E3 |

## 7. 复现

```bash
export PATH=/opt/miniconda3/envs/atmosgen/bin:$PATH
cd "Atmospheric Data Generation"
pip install -e . --no-deps
pytest -q                                                # 137 passed
python scripts/run_experiments.py --experiments e1,e2,e3 # benchmark + summary.json
python scripts/validate_solver_analytic.py               # 求解器 6 项验证
bash scripts/run_e2e.sh                                  # Test F 全链路
```

产物：`output/experiments*/summary.json`、`output/*/quality_report.{json,md}`、`output/*/figures/*.png`、`output/*/rejected.jsonl`（混合）。
