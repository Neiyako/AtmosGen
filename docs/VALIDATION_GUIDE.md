# 验证指南（VALIDATION_GUIDE）

> 结论先行：`atmosgen evaluate` 用**四类模块**（statistical / temporal / spatial / physical）给出**逐指标数值 + 阈值 + 状态**，
> 报告落盘为 `quality_report.json`（严格合法 JSON）与 `quality_report.md`（含图）。
> 状态取值 `PASS / CONDITIONAL_PASS / NOT_RUN / FAIL / INFO`，整体按 **FAIL > NOT_RUN > CONDITIONAL_PASS > PASS** 聚合；
> "不适用"与"未运行"分开记录，`NOT_RUN ≠ PASS` 是本项目的基本纪律。

---

## 1. 评价入口与数据要求

```bash
# 有真实参考数据（statistical / temporal / spatial 可算）
atmosgen evaluate --input output/<run>/data.nc --real data/processed/era5_local.nc --overwrite --fail-on never

# 无真实参考（只有 physical 可算，其余显式 NOT_RUN）
atmosgen evaluate --input output/physics_*/data.nc --overwrite --fail-on never

# 渲染 Markdown（quality_report.json → md，可指定输出路径）
atmosgen report --input output/<run>/quality_report.json --format md --out output/<run>/quality_report_rendered.md
```

| 参数 | 作用 |
|---|---|
| `--input` | 待评价的规范数据集（`data.nc`，必须带内嵌契约） |
| `--real` | 真实参考数据集；缺省时 statistical/temporal/spatial 显式 NOT_RUN（不是静默通过） |
| `--config` | generation 配置，用于读取 `validation.thresholds_overrides` |
| `--out` | 报告输出目录（默认与 `--input` 同目录） |
| `--overwrite` | 覆盖已有 `quality_report.json` |
| `--fail-on fail\|never` | 默认 `fail`：整体 FAIL 时退出码 4；`never`：只写报告、始终退出 0 |

对齐规则：`--real` 会按合成数据的 time/level/latitude/longitude **精确坐标 + 容差**重采样（`evaluation.align_real_to_synthetic`）；
无法对齐时**报错退出**（退出码 4），绝不拿错位的真实数据算指标。

---

## 2. 统计分布一致性（`statistical`）

对每个变量的**逐层列**（如 `t@500`、`q@850`）计算，取最差值作为该指标值。

| 指标键 | 方法 | 默认阈值 | 判据 |
|---|---|---|---|
| `column_shapes` | sdmetrics QualityReport · Column Shapes（KSComplement/TVComplement 均分） | ≥ 0.80 | 越高越好 |
| `column_pair_trends` | sdmetrics · Column Pair Trends（CorrelationSimilarity） | ≥ 0.80 | 越高越好 |
| `mean_ratio` | 逐列 `mean_gen / mean_real` | 0.85–1.15 | 范围 |
| `std_ratio` | 逐列 `std_gen / std_real` | 0.75–1.25 | 范围 |
| `quantile_rmse` | q05/q25/q50/q75/q95 归一化 RMSE | ≤ 0.10 | 越低越好 |
| `ks` | `scipy.stats.ks_2samp` 逐列 KS 统计量 | ≤ 0.15 | 越低越好 |
| `corr_frobenius` | 变量间相关矩阵的 Frobenius 距离 / √n | ≤ 0.25 | 越低越好 |
| `extreme_ratio` | P1/P99 分位数的比值 | 0.5–2.0 | 范围 |

**已知口径问题**：`mean_ratio` 在真实均值接近 0 的列（如高层 u/v）上会被放大，读取 detail 的逐列数值判断，不要只看聚合值。

> sdmetrics 现状：`sdmetrics.reports.single_table.QualityReport` 可用但带弃用告警；非弃用的 `sdmetrics.reports.QualityReport`
> 在单表包装（`{"tables": {...}}`）下 `Column Pair Trends` 的 Synthetic Correlation 全为 NaN（0.33.0 实测），
> 因此 Phase 2 起沿用 single_table 实现，升级条件见 `docs/PHASE2_REPORT.md` 偏差 P3。

---

## 3. 时间连续性（`temporal`）

沿 time 轴对 (variable × level × cell) 序列计算，抽样 ≤200 条序列（固定种子 42），取中位数/最差值。

| 指标键 | 方法 | 默认阈值 | 判据 |
|---|---|---|---|
| `acf_diff` | lag 1–6 的平均 \|ΔACF\| | ≤ 0.20 | 越低越好 |
| `diff_var_ratio` | `var(Δx_gen)/var(Δx_real)` | 0.5–2.0 | 范围 |
| `spectral_slope_diff` | 沿时间 rfft → log-log 斜率差 | ≤ 0.50 | 越低越好 |
| `jump_extra` | gen 5σ 超阈比例 − real 基线 | ≤ 0.01 | 越低越好 |
| `window_var_ratio` | 8 步窗均值方差比（gen/real 中位数） | ≤ 2.0 | 越低越好 |
| `timeseries_efficacy` | sdmetrics LSTM 判别（可选，`validation.temporal.timeseries_efficacy`） | — | 默认 NOT_RUN，Phase 4/5 视需要显式调用 |

> 判读纪律：cell_time 统计生成**不建模时间相关**，本模块 FAIL 是计划预设结果（`IMPLEMENTATION_PLAN` §4.1、风险 X2），
> 不是 bug；物理模式输出则应显著优于统计模式（时间演化由求解器决定）。

---

## 4. 空间连续性（`spatial`）

对每个时次 × 层的 2D 场计算，抽样 ≤50 个场（固定种子 123）。

| 指标键 | 方法 | 默认阈值 | 判据 |
|---|---|---|---|
| `variogram_ratio` | 经验半变异函数 γ(h)（lag 1–5）比值 | 0.5–2.0 | 范围 |
| `gradient_rms_ratio` | 空间梯度 RMS 比 | 0.5–2.0 | 范围 |
| `neighbor_ks` | `x − mean(4邻域)` 的 KS | ≤ 0.20 | 越低越好 |
| `spatial_spectral_slope` | 径向平均功率谱 log-log 斜率差 | ≤ 0.30 | 越低越好 |
| `anisotropy_diff` | x/y 方向梯度能量比的对数差 | ≤ 0.30 | 越低越好 |
| `corr_length` | 自相关降到 1/e 的距离比 | 0.5–2.0 | 范围 |

---

## 5. 物理合理性（`physical`）

### 5.1 硬约束（任一违例 → 该 metric FAIL）

| 指标键 | 判据 | 默认阈值 |
|---|---|---|
| `finite` | 无 NaN/Inf（已声明筛检遮罩的 NaN 除外，见 §6） | 坏值数 = 0 |
| `q_nonneg` | `q ≥ 0` | 负值数 = 0 |
| `cloud_nonneg` | `clwc/ciwc/crwc/cswc ≥ 0` | 负值数 = 0 |
| `t_range` | `t_min ≤ t ≤ t_max` | 180–340 K（`t_min`/`t_max`） |
| `wind_range` | `|u|,|v| ≤ wind_max` | 150 m/s（`wind_max`） |
| `saturation` | `q ≤ factor × q_sat(t, p)`，Tetens 公式 | factor = 1.10（`saturation_factor_max`） |
| `tcwv_range` | `0 ≤ tcwv ≤ tcwv_max` | 120 kg m⁻²（`tcwv_max`） |
| `c_nonneg` | 物理追迹浓度 `c ≥ 0`（物理模式） | 负值数 = 0 |

### 5.2 软指标（记录数值，不直接判死）

| 指标键 | 方法 | 期望 |
|---|---|---|
| `mass_budget` | 周期边界下总质量相对漂移 | ≤ 5%（`mass_drift_max`）；超限降级 CONDITIONAL_PASS |
| `smoothness` | 棋盘格翻转率（gen/real 比） | ≤ 2×（`smoothness_factor_max`）；无 real 时 INFO |
| `equation_residual` | `∂c/∂t+u∂c/∂x+v∂c/∂y−κ∇²c` 离散残差 RMS（INFO，Phase 4/5 定阈值） | 记录数值 |

> `equation_residual` 支持两种风场来源：数据集中的 `u`/`v` 字段，或契约 `physical_parameters.u0/v0`（Mode 2 常量风场语境），
> 要求**均匀时间步**（hybrid 的快照调度已保证等间隔）。

---

## 6. 状态语义与聚合

| 状态 | 含义 |
|---|---|
| `PASS` | 指标达到阈值 |
| `CONDITIONAL_PASS` | 软指标超期望但记录明确（不判死） |
| `NOT_RUN` | 未计算：输入残缺（如缺 level、缺变量）或数据集不适用；必须写原因 |
| `FAIL` | 硬约束违例或指标越界 |
| `INFO` | 仅记录、无阈值（如无 real 时的 smoothness） |

**整体状态聚合**（`validation/report.py::aggregate_status`，从低到高）：

```
FAIL  >  NOT_RUN  >  CONDITIONAL_PASS  >  PASS
```

- 模块状态 = 该模块内各指标聚合；报告整体 = 四类模块中最低状态。
- `INFO` 不参与聚合（软指标记录不断案），但会出现在 notes 中。
- **"不适用" vs "未运行"**：变量在本数据集中不存在（如统计数据集没有云水变量）→ 该指标 NOT_RUN 且**不参与聚合**（notes 列明）；
  所需输入存在但无法计算（level 单位不可识别、形状不一致、全 NaN）→ NOT_RUN 且**参与聚合**（会让整体至少为 NOT_RUN）。
- 核心状态变量（t/q/u/v/c）全部缺失的数据集 → physical 整体 NOT_RUN（不因剩余可算指标而误判 PASS）。

**筛检遮罩**：Mode 3 输出中被物理筛检拒绝的单元以 NaN 保存，同时在 `atmosgen_screening.mask_applied=true` 中声明；
此时 `finite` 允许 NaN（计为"筛检遮罩"），但 Inf 永远算坏值。未声明遮罩的数据集出现 NaN → `finite` FAIL。

---

## 7. 报告结构

```jsonc
{
  "meta": {"input": "...", "real": "...", "contract": {...}, "threshold_overrides": {}, "environment": {...}},
  "statistical": {"status": "...", "metrics": {"<key>": {"value", "threshold", "comparison", "status", "detail"}}, "notes": []},
  "temporal":    {...},
  "spatial":     {...},
  "physical":    {...},
  "overall_status": "FAIL",
  "coverage": {"implemented": [...], "not_implemented": []},
  "figures": ["figures/distributions.png", "figures/acf_spectrum.png", "figures/variogram.png"]
}
```

- 所有 `value` 均可回溯到 detail 中的逐列/逐场数值；
- 非有限数值写为 JSON `null`（严格解析器可读）；
- 图表：边际分布对比、ACF + 功率谱、半变异函数（每 run 3 张）。

---

## 8. 阈值覆盖

`configs/*.yaml` 的 `validation.thresholds_overrides` 支持覆盖上表阈值；键名集合见 `atmosgen/config.py::THRESHOLD_LIMITS`（含取值范围校验）。

```yaml
validation:
  statistical: true
  temporal: true
  spatial: true
  physical: true
  thresholds_overrides:
    ks_max: 0.12
    saturation_factor_max: 1.05
    mass_drift_max: 0.03
```

未知键会作为自定义阈值原样传递（`validate-config` 打印 warning），界外值会被 `validate-config` 拒绝（退出码 2）。

---

## 9. 已知限制

1. **sdmetrics 弃用 API**：见 §2 说明；升级条件写进 `docs/PHASE2_REPORT.md` P3。
2. **Tetens 适用范围**：水面饱和公式在 -40~+50 °C 较可靠；`t_range` 放宽到 180–340 K 时，冰点以下会高估饱和水汽（饱和检查偏宽松）；
   极端高温低压点会因 `p−0.378e_s ≤ 0` 被判"非法气压"并排除。
3. **saturation 只支持单气压坐标**（hPa/mb/Pa；`model_level` → NOT_RUN，不猜单位）。
4. **equation_residual 未按场量级归一化**（单位 c·s⁻¹），Phase 4/5 定阈值前必须无量纲化；且要求均匀 dt、单层语义。
5. **smoothness 的语义**：符号翻转率对正定场（如 c、q）恒为 0，只用于捕捉棋盘格伪迹，不是通用平滑度。
6. **mass_budget 未做面积/气压加权**（对 c/tcwv/q 直接全场求和），仅作相对漂移诊断。
7. **时空指标抽样**：temporal ≤200 条序列、spatial ≤50 个场，固定种子；极端局部异常可能被抽样遗漏。
8. **整体 FAIL 不等于"全部指标失败"**：例如统计模式 `statistical` 7/8 指标 PASS 时，仅 `corr_frobenius`（跨层相关）FAIL
   即导致模块 FAIL——这是计划 §5.5 的聚合口径，读报告时请同时看逐指标状态。
9. **NaN 遮罩语义**只在声明 `atmosgen_screening.mask_applied` 时成立；手工注入 NaN 仍按缺失/坏值处理。
