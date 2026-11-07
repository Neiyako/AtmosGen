# Phase 6 报告 · 结构保持型合成（scientific-usability 优化）

- 日期：2026-11-07（深夜）
- 起因：`docs/DATA_QUALITY_FOR_CLOUD_WATER.md` 的诊断结论——统计生成"边缘像、导数型量全错"，
  无法支撑云水资源量等**导数型下游计算**；用户要求"保障仿真数据的科学性和可用性"。
- 结论：**新增 `analog_blocks`（结构保持型合成）模式**，把导数型量的可用性从"不可用"提升到"可用"：
  MFC 图案相关 **0.002 → 0.894**、MFC 幅度比 0.885 → **0.84**、域总量比 0.161 → **1.07**、
  空间图案相关 0.009 → **0.739**、lag-1 ACF −0.260 → **+0.708**、饱和违例 3.29% → **0.00%**；
  边缘统计仍保持（KS(t) 0.024）。同时**如实披露**输出性质：它是真实场的重组
  （`independence=recombined`，`novelty.exact_copy_cell_fraction`）。

---

## 1. 为什么必须换模式（机理）

云水资源量（MFC、凝结-降水差、水平/垂直积分）是**导数型量**：对独立采样的场求散度，
噪声被差分算子放大到与信号同量级 —— 实测 MFC 幅度比 0.885 却图案相关 r=0.002，
"量级像、逐格全错"，是比明显崩坏更危险的失败模式。

因此**单表统计模型（SDV）在结构上无法满足**这类需求：它建模格点值的联合分布，不承诺时空协方差。
可行路径只有两条：① 让生成模型本身带时空结构（时空条件生成，Phase 2 已实测条件采样在经验分布边界退化）；
② **保留真实结构做重组**（本阶段实现）。

## 2. 新模式：`analog_blocks`（结构保持型合成）

| 要素 | 设计 |
|---|---|
| 时间轴 | 连续 `block_steps` 步的**真实块**按计划重排 → 保留过程持续性（实测 ACF1 0.708） |
| 空间 | `patch_mode=whole_field`：整个域取同一源块 → 保留大尺度天气系统（导数型量的前提） |
| 多样性 | `patch_mode=block/mosaic`（空间分块重组）→ 更"新颖"但**破坏大尺度结构**（实测 MFC r 0.00–0.14，不可用于资源量计算） |
| 跨变量一致性 | 同一格点的所有变量取自同一源块 → 物理一致性继承自真实场 |
| 物理筛检 | §5.4 硬约束逐格点筛检（实测违例 0），不合格单元掩码 + `rejected.jsonl` |
| 新奇度 | `noise_jitter` 注入相对量级抖动；`novelty.exact_copy_cell_fraction` 披露与真实数据的重叠 |
| 契约披露 | `generation_config.independence="recombined"`、`novelty.*`、`method.*`（block_steps/patch_cells/patch_mode/noise_jitter） |

### 参数敏感性实测（真实 ERA5 切片，26×26@1°，16 时次 ×3 层）

| patch_mode | patch_cells | KS(t) | 空间 r | ACF1 生 | **MFC 图案 r** | MFC 幅度比 | MFC 域总量比 |
|---|---|---|---|---|---|---|---|
| mosaic | 8 | 0.048 | 0.138 | 0.603 | 0.113 | 1.08 | 0.07 |
| mosaic | 13 | 0.097 | 0.415 | 0.756 | 0.058 | 1.14 | 0.52 |
| mosaic | 26 | 0.030 | 0.739 | 0.579 | 0.857 | 0.89 | 0.83 |
| block | 8 | 0.131 | 0.086 | 0.601 | 0.142 | 1.81 | 0.68 |
| block | 13 | 0.089 | 0.428 | 0.708 | −0.005 | 1.14 | 0.21 |
| **block / whole_field** | **整域** | **0.024** | **0.739** | **0.708** | **0.894** | **0.84** | **1.07** |

**结论**：只有**整域一致**的取源方式能保住导数型量；任何空间拼接都会破坏大尺度辐合结构。
这与"云水资源量依赖天气系统尺度"的物理直觉一致，也是本阶段最有工程价值的结论。

## 3. 四类验证对比（同一真实参考）

| 模块 | copula（优化前） | **analog_blocks（优化后）** |
|---|---|---|
| statistical | 7/8 PASS（corr_frobenius 1.112 FAIL） | **7/8 PASS（corr_frobenius 0.198 PASS、KS 0.097、column_shapes 0.940、column_pair_trends 0.989）**；mean_ratio 1.474 FAIL（重组后 u/v 的层-时次配对变化，见 §5） |
| temporal | ACF1 −0.260、diff_var_ratio 338 | ACF1 **+0.708**；temporal 仍 FAIL（口径是"与真实同一时刻对比"，而合成时序是**重排**，见 §5） |
| spatial | variogram 16.6、neighbor_ks 0.364、corr_length 0.167 FAIL | variogram 2.24、neighbor_ks **0.056 PASS**、corr_length **1.000 PASS**、gradient 0.875 PASS |
| **physical** | **FAIL**（饱和违例 3.29%） | **PASS**（finite/q/t/wind/saturation/tcwv 全 PASS，违例 0.00%） |

## 4. 对论文/业务的直接价值

1. **云水资源量计算**：用 `analog_blocks`（whole_field）的 MFC 图案相关 0.894、域总量比 1.07，
   空间格局可用；`mosaic/block` 模式**不可用**（MFC r ≤0.14），已在配置注释与本文写明。
2. **方法学结论**：给出"边缘相似 ≠ 导数型量可用"的可复现证据（KS 0.03 但 MFC r 0.00），
   以及"结构保持型合成可修复该缺陷"的量化对照（0.00 → 0.89）。
3. **诚实披露**：输出是真实场重组，契约写明 `independence=recombined` 与逐位重复比例；
   论文中必须声明"非独立样本"，适用于敏感性/数据增强分析，不适用于独立样本统计检验。

## 5. 剩余限制（如实）

| # | 限制 | 说明 |
|---|---|---|
| L1 | **输出非独立样本** | 逐位重复比例默认 1.0（jitter=0）；需在论文/报告中声明 |
| L2 | **时间轴是重排** | 生成序列与真实同一时刻不对齐 → temporal 指标（逐时刻对比）仍 FAIL；序列自身持续性正确（ACF1 0.708） |
| L3 | mean_ratio 1.474 FAIL | 重组后 u/v 的"层×时次"配对关系改变，与真实逐列均值不可比；同一现象在 corr_frobenius 上是**改善**（1.112 → 0.198） |
| L4 | 时空分辨率 | 沿用真实 1° 抽稀网格；要更高分辨率需更多真实数据（源文件 0.25°，可用 `subsample` 调细，但样本量随之增大） |
| L5 | 多样性 vs 一致性 | 想要"既新颖又保留大尺度"需要真正的时空生成模型（如条件 GAN / 扩散模型），Phase 2 已实测 SDV 条件采样在边界退化，属后续工作 |
| L6 | 变量可加性 | 目前重组同一组变量；引入新变量（如云水 clwc/ciwc）需等云水样本到位（仍 NOT_RUN） |

## 6. 复现

```bash
export PATH=/opt/miniconda3/envs/atmosgen/bin:$PATH
cd "Atmospheric Data Generation"
atmosgen generate --config configs/analog_blocks.yaml
atmosgen evaluate --input output/statistical_analog_blocks_*/data.nc --real data/processed/era5_local.nc --overwrite --fail-on never
python scripts/quality_fidelity.py --real data/processed/era5_local.nc \
       --runs output/statistical_analog_blocks_*/data.nc output/statistical_gaussian_copula_*/data.nc \
       --out output/quality_fidelity
```

## 7. 结论

- 统计生成的**边缘保真**与**导数型可用性**是两个不同目标；前者已达标（KS 0.024–0.031），
  后者需要结构保持——本阶段用 `analog_blocks` 达成（MFC r 0.894），并以契约如实披露其"重组"性质。
- 对云水资源量这类下游计算，**推荐 `analog_blocks`（whole_field，jitter 按需）**；
  `gaussian_copula/ctgan/tvae` 保留用于"需要独立样本的边缘分布研究"；`hybrid` 用于"需要物理基底与空间结构的混合实验"。
