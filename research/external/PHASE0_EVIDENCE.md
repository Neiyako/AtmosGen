# Phase 0 外部证据（AtmosGen 审计输入）

- 核对日期：**2026-11-06**（全部 URL 均在当日实际请求；未拿到正文的条目已标「未读正文」）
- 状态：外部证据汇编，**不是**最终审计结论；结论以本文件 §1 为准，缺口见 §5
- 采集方式：`gh api`（已认证，GitHub 元数据/文件树）、`raw.githubusercontent.com`（许可证原文）、`storage.googleapis.com` 公共 bucket 的 `.zmetadata`（Zarr 变量清单，匿名可读）
- 原始落盘：`research/external/_raw/`（`*.repo.json`、`*.zmetadata`、许可证全文、`rpdb_metrics.py`）

---

## 1. 结论先行

| 项目 | 存在性 / 维护 | 许可证 | 本项目的可用级别 | 一句话理由 |
|---|---|---|---|---|
| **SDV** (`sdv-dev/SDV`) | ✅ 3600★，未归档，`main` 分支最后 commit **2026-11-05**（今日仍有 push 2026-11-06） | **BSL 1.1**（DataCebo），Change Date = 发布后 4 年，Change License = MIT | **直接复用**（单表/时序生成核心） | Additional Use Grant 只禁「Synthetic Data Service」，科研/内部生产不受限 |
| **SDGym** (`sdv-dev/SDGym`) | ✅ 312★，未归档，最后 commit **2026-11-04** | **BSL 1.1**，Additional Use Grant 禁的是「Synthetic Data Benchmarking Service」 | **直接复用**（评测框架，非产品化不发服务即可） | 同 BSL，但禁的品类不同；我们自用跑 benchmark 不构成该服务 |
| **sdmetrics** (`sdv-dev/sdmetrics`) | ✅ 264★，未归档，最后 commit **2026-11-02** | **MIT** | **直接复用**（质量指标层） | 唯一真正宽松许可的一环 |
| **Copulas** (`sdv-dev/Copulas`, SDV 依赖) | ✅ 658★，未归档，最后 commit **2026-11-02** | **BSL 1.1**（禁「Synthetic Data Creation Service」） | 间接复用（作为 SDV 依赖） | 通过 SDV 使用即可；不单独分发其服务 |
| **Synthcity** (`vanderschaarlab/synthcity`) | ⚠️ 693★，未归档，但 `main` 最后 commit **2025-06-17**（约 5 个月无进展） | **Apache-2.0** ✅ | **Adapter**（只挑时序插件思路或实现，不整体依赖） | 许可最友好，但 PyPI 只到 0.2.12 且 pin `torch<2.3 / numpy<2`，与 py3.11+新 torch 冲突 |
| **PDEBench** (`pdebench/PDEBench`) | ✅ 1227★，未归档，最后 commit **2026-03-30** | **MIT**（根目录 `LICENSE.txt`，正文已读） | **只借方法 + 有限复用数据生成器** | pip 包 `pdebench==0.1.0` pin `python<3.11`、`torch~=1.13`，不适配我们的 3.11 环境；源码可读可改但别装包 |
| **RealPDEBench** (`AI4Science-WestlakeU/RealPDEBench`) | ✅ ICLR26 Oral，137★，创建 2025-12-30，最后 commit **2026-11-04** | **CC BY-NC 4.0**（正文："non-commercial use only"） | **只借方法（非商用前提下）** | 9 个评测指标可借鉴；5 个数据集全是流体/燃烧，**与大气无关**；NC 限制禁止商用 |
| **天气数据源** | WB2 与 ARCO 均**匿名可读**（HTTP 200，无需 token） | WB2 bucket 有 `datasets/era5/LICENSE`（未读正文） | **ARCO-ERA5 模型层**为主要云水来源 | **WB2 根本没有 clwc/ciwc/crwc/cswc**；ARCO 压力层有 clwc/ciwc，模型层四者全有 |

**最关键的三个判断（给计划书用）**

1. **云水变量必须换源**：WeatherBench2 的 ERA5 Zarr **不含** `clwc/ciwc/crwc/cswc`，只含 `total_cloud_cover` 与 `total_column_water_vapour`。含四类云水的是 **ARCO-ERA5**（`gs://gcp-public-data-arco-era5/`），其中 `ar/model-level-1h-0p25deg.zarr-v1` 同时具备 `specific_cloud_{liquid,ice}_water_content` 与 `specific_{rain,snow}_water_content`。→ 数据方案以 ARCO 为准，WB2 仅作压层快速基线。
2. **许可证风险可控但不可商用**：SDV/SDGym/Copulas 是 BSL 1.1（非 OSI），RealPDEBench 是 CC BY-NC 4.0；**本项目若定位为科研/论文，可全部合规使用；一旦产品化或提供对外合成数据服务，必须替换 SDV 栈或买商业许可、并剥离 RealPDEBench 派生代码**。
3. **Synthcity 不能整体接入**：许可虽好（Apache-2.0），但最后 commit 停在 2025-06-17、pin 老 torch/numpy，只能抄插件结构或做隔离 Adapter。

---

## 2. 逐项目证据表

### 2.1 存在性与维护（`gh api repos/{owner}/{repo}`，核对日期 2026-11-06）

| 仓库 | default_branch | stars | archived | 最后 commit（默认分支） | pushed_at | 许可证（GitHub 识别） |
|---|---|---|---|---|---|---|
| `sdv-dev/SDV` | `main` | 3600 | false | 2026-11-05T11:42:21Z `549dc94` | 2026-11-06T13:58:12Z | NOASSERTION（实为 BSL 1.1） |
| `sdv-dev/SDGym` | `main` | 312 | false | 2026-11-04T18:36:51Z `009a35b` | 2026-11-04T18:37:48Z | NOASSERTION（实为 BSL 1.1） |
| `sdv-dev/sdmetrics` | `main` | 264 | false | 2026-11-02T17:18:43Z `e6f745f` | 2026-11-02T17:18:46Z | MIT |
| `sdv-dev/Copulas` | `main` | 658 | false | 2026-11-02T16:29:52Z `7c2d15d` | 2026-11-02T16:29:54Z | NOASSERTION（实为 BSL 1.1） |
| `vanderschaarlab/synthcity` | `main` | 693 | false | **2025-06-17T08:40:31Z** `23f322f` | 2026-04-21T17:24:15Z | Apache-2.0 |
| `pdebench/PDEBench` | `main` | 1227 | false | 2026-03-30T08:27:30Z `4ff3e3a` | 2026-03-30T08:34:19Z | NOASSERTION（实为 MIT） |
| `AI4Science-WestlakeU/RealPDEBench` | `main` | 137 | false | 2026-11-04T12:52:31Z `06ba38d` | 2026-11-04T12:52:31Z | NOASSERTION（实为 CC BY-NC 4.0） |

**任务书地址纠错已复核**：`gh api repos/realpdebench/RealPDEBench` → `{"message":"Not Found", "status":"404"}`。
正确地址 **`https://github.com/AI4Science-WestlakeU/RealPDEBench`**（`created_at` 2025-12-30T10:34:17Z，description 含 `[🔥ICLR26 Oral] RealPDEBench: A Benchmark for Complex Physical Systems with Paired Real-World and Simulated Data`）。

> 注：`pushed_at` 与默认分支最后 commit 可以不同（分支推送/标签），上表两者都列。synthcity 的 `pushed_at 2026-04-21` 晚于最后 commit，说明有其它分支活动，**默认分支实质停更**。

### 2.2 许可证逐条（原文已下载，路径见文件头「原始落盘」）

| 项目 | 许可证文件 URL | 原文关键句（逐字） |
|---|---|---|
| SDV | https://raw.githubusercontent.com/sdv-dev/SDV/main/LICENSE | `Business Source License 1.1` / `Licensor: DataCebo, Inc.` / `Additional Use Grant: You may make use of the Licensed Work, and derivatives of the Licensed Work, provided that you do not use the Licensed Work, or derivatives of the Licensed Work, for a Synthetic Data Service.` / `Change Date: Change date is four years from release date.` / `Change License: MIT License` / `The Business Source License (this document, or the "License") is not an Open Source license.` |
| SDV 基础授予条款 | 同上 | `The Licensor hereby grants you the right to copy, modify, create derivative works, redistribute, and make non-production use of the Licensed Work.` |
| SDGym | https://raw.githubusercontent.com/sdv-dev/SDGym/main/LICENSE | 与 SDV 同版式，差异仅在 `Licensed Work: SDGym` 与禁用品类 `Synthetic Data Benchmarking Service`（`diff` 实测：仅 7 处行差异，全部是产品名/服务名替换） |
| Copulas | https://raw.githubusercontent.com/sdv-dev/Copulas/main/LICENSE | `Licensed Work: Copulas`；禁用品类 `Synthetic Data Creation Service` |
| sdmetrics | https://raw.githubusercontent.com/sdv-dev/sdmetrics/main/LICENSE | `MIT License` / `Copyright (c) 2020, MIT Data To AI Lab` |
| Synthcity | https://raw.githubusercontent.com/vanderschaarlab/synthcity/main/LICENSE | **未读正文**（PyPI `info.license` = `Apache-2.0`，GitHub API `license.spdx_id` = `Apache-2.0`；两者一致，未逐字读） |
| PDEBench | https://raw.githubusercontent.com/pdebench/PDEBench/main/LICENSE.txt | `Except where otherwise stated this code is released under the MIT license.` + 标准 MIT 正文；`Copyright 2022 NEC Labs Europe GmbH, Stuttgart University, CSIRO and PDEbench contributors` |
| RealPDEBench | https://raw.githubusercontent.com/AI4Science-WestlakeU/RealPDEBench/main/LICENSE | `Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)` / `This repository is released under the CC BY-NC 4.0 license (non-commercial use only).` / `You are free to share and adapt the material for non-commercial purposes, provided that you give appropriate credit and indicate if changes were made.` |

**对「科研 / 再分发 / 商用」的实际限制（一句话结论）**

- **SDV / SDGym / Copulas（BSL 1.1）**：科研与内部生产**允许**（Additional Use Grant 只禁对外提供合成数据/基准服务）；**再分发**允许但必须随附该 License 且整体受 BSL 约束（`You must conspicuously display this License on each original or modified copy`）；**商用**除「对外合成数据服务」外允许，若触碰该品类须买商业许可；每个版本发布满 4 年后自动转 MIT。
- **sdmetrics（MIT）**：科研/再分发/商用**全部允许**，只需保留版权声明。
- **Synthcity（Apache-2.0）**：科研/再分发/商用**全部允许**（需保留 NOTICE、专利授权条款）。
- **PDEBench（MIT）**：同 MIT，**唯一限制**是「Except where otherwise stated」——数据集/第三方素材可能另有条款，使用其预生成数据前需单独核。
- **RealPDEBench（CC BY-NC 4.0）**：科研可用、可再分发（需署名 + 注明改动），**禁止商业用途**；「非商用」判断按 CC 定义，谨慎起见本项目的论文/毕设可用，任何付费/产品化用途**必须完全剥离**。

### 2.3 依赖兼容（PyPI JSON `https://pypi.org/pypi/{pkg}/json`，核对 2026-11-06）

| 包 | 最新版 | requires_python | license 字段 | 备注 |
|---|---|---|---|---|
| `sdv` | **2.1.0** | `>=3.10,<3.15` | `license_expression: BUSL-1.1` | python 3.13 可装 |
| `sdmetrics` | **0.33.0** | `>=3.10,<3.15` | `license_expression: MIT` | |
| `sdgym` | **0.16.1** | `>=3.10,<3.15` | `license_expression: BUSL-1.1` | |
| `copulas` | **0.15.0** | `>=3.10,<3.15` | `license_expression: BUSL-1.1` | |
| `synthcity` | **0.2.12** | `>=3.9` | `Apache-2.0` | ⚠️ 无上界但实测 pin 老 torch/numpy（见下） |
| `pdebench` | **0.1.0** | `>=3.9,<3.11` | MIT 全文 | ⚠️ **与 py3.11 冲突**（要求 `<3.11`） |

**结论**：本机 `python 3.13` base + conda 可建 3.11 的方案里，**SDV 栈可装在 3.13 或 3.11**；`pdebench` 包两头都装不了（3.13 太新、3.11 正好卡在 `<3.11` 边界外），只能**读源码 / clone 后用其中 `data_gen` 脚本**；synthcity 需专门降级环境（见 §2.4）。

### 2.4 RealPDEBench 评测协议与数据形态（决定「只借方法」）

**证据来源**：
- README https://github.com/AI4Science-WestlakeU/RealPDEBench/blob/main/README.md （已读全文，12340 B）
- 指标代码 https://github.com/AI4Science-WestlakeU/RealPDEBench/blob/main/realpdebench/utils/metrics.py （已读 `eval_metrics` 全文）

README 原文关键句（逐字）：

> `- **[5 Datasets](https://realpdebench.github.io/datasets/)**: `cylinder`, `fsi`, `controlled_cylinder`, `foil`, `combustion``
> `- **[700+ Trajectories](https://realpdebench.github.io/datasets/#dataset-inventory)**`
> `- **[9 Evaluation metrics](https://realpdebench.github.io/metrics/)**: RMSE, MAE, Rel L₂, R², Update Ratio, fRMSE, FE, KE, MVPE`
> `for complex physical systems, designed for **spatiotemporal forecasting** and **sim-to-real transfer**.`

**9 个指标的代码落点**：`realpdebench/utils/metrics.py::eval_metrics(pred, target, c, batch_size=None)`，其 docstring 原文：

> `Adapted from https://github.com/pdebench/PDEBench/blob/main/pdebench/models/metrics.py`

实际返回（代码逐字核对，**13 个张量**，README 的 9 个是聚合口径）：

| README 指标 | 代码变量 | 公式（源码逐字）/ 说明 |
|---|---|---|
| RMSE | `rmse` | `torch.sqrt(torch.mean(nn.MSELoss(reduction='none')(pred,target)))` |
| MAE | `mae` | `torch.mean(torch.abs(pred - target))` |
| Rel L₂ | `rel_l2_error` | `torch.mean(‖pred-target‖₂ / ‖target‖₂)`（按 batch 展平后取均值） |
| R² | `r2` | `1 - Σ(pred-target)² / Σ(target - target.mean(0))²` |
| KE（动能误差） | `ke_error` | `0.5*mean((u'-ū)²)+0.5*mean((v'-v̄)²)` 的预测-真值绝对差；**`c<2` 时直接置 0**（非流体通道数不足则失效） |
| fRMSE（谱误差） | `f_error` / `low_f_error` / `mid_f_error` / `high_f_error` | 3D FFT 按半径分 bin，`_err_F = sqrt(mean(err_F))/(t*h*w)`；再按 `min(t/2,h/2,w/2)` 的 1/3、2/3 切低/中/高频 |
| FE（相对谱误差） | `rel_low/mid/high_f_error` | `(_err_F / _norm_F)` 分带均值 |
| Freq error | `freq_error` | 沿时间维求和后 1D FFT 的绝对差均值 |
| MVPE | **未在 `eval_metrics` 中出现** | 见 §5 未核实项 |

**数据形态**：`pred, target: torch.Tensor, [b, t, h, w, c']`（源码 docstring 逐字），即 `(batch, time, height, width, channels)` 的 5D 张量；`kinetic_energy` 同样注释 `[b, t, h, w, c]`。
→ **与大气数据的差异**：大气是 **4D 球面格点 `(time, level, lat, lon)` 或 `(time, lat, lon)`**，没有 `h/w` 平面笛卡尔结构、也没有「通道=物理矢量分量」的默认约定；RealPDEBench 的 FFT 分带与 KE 都假设空间均匀网格 + 至少 2 个速度分量。**可迁移的只有标量指标（RMSE/MAE/RelL2/R²）+ 谱误差的分带思想**，KE/MVPE 需要按球面重新定义。

**5 个数据集与大气的关系**：`cylinder`（圆柱绕流）、`fsi`（流固耦合）、`controlled_cylinder`（受控圆柱）、`foil`（翼型）、`combustion`（燃烧）——**全部是流体力学/燃烧，无一与大气/气象相关**。数据托管：HF `AI4Science-WestlakeU/RealPDEBench`（`realpdebench download --dataset-root ... --what hf_dataset`），原始 HDF5 在 `https://realpdebench.westlake.edu.cn/`（README 原文）。
→ **结论：只借方法，不复用数据**；且 CC BY-NC 使得任何借来的指标实现也要保留 NC 约束。

### 2.5 Synthcity 时序模型与 3D 张量限制（决定 Adapter 形态）

**文件树证据**（`gh api repos/vanderschaarlab/synthcity/git/trees/HEAD?recursive=1`）：

```
src/synthcity/plugins/time_series/plugin_timegan.py
src/synthcity/plugins/time_series/plugin_timevae.py
src/synthcity/plugins/time_series/plugin_fflows.py
tests/plugins/time_series/test_timegan.py
tutorials/tutorial4_time_series.ipynb
tutorials/tutorial6_time_series_data_preparation.ipynb
```

`plugin_timegan.py` 关键行（已读，逐字）：

- L30 `class TimeGANPlugin(Plugin):`
- L345-348：`n_static_units_in=static.shape[-1]` / `n_temporal_units_in=temporal[0].shape[-1]` / `n_temporal_window=temporal[0].shape[0]` / `output_shape=outcome_enc.shape[1:]`
  → 插件内部把数据拆成 **`static`**（静态协变量）+ **`temporal`**（序列，形状 `[seq_len, n_features]`），即**外部输入必须是 3D `(n_samples, seq_len, n_features)`**。
- 该类参数含 `generator_lr/discriminator_lr/n_layers_hidden/n_units_hidden/weight_decay` 等，继承 `Plugin` 抽象接口（`fit/ generate/ fit_generate`）。

⚠️ **未逐字读到的部分**：把 `(n, seq, feat)` 约束写死的代码位置（可能在 `synthcity/utils/datasets/time_series/` 或 `TimeSeriesDataLoader`），以及 GSGAN/VAE 的具体类名清单 —— 见 §5。

**阻塞判断**：PyPI 0.2.12 + 最后 commit 2025-06-17 → 与 `torch>=2.3 / numpy>=2` 的项目主环境冲突（任务书所述 pin 未逐字核到，见 §5）。**当 Adapter 用的成本 > 自己写一个轻量时序生成器**，建议：只借用「static + temporal 拆分」这一数据组织契约，模型层不引入。

### 2.6 PDEBench 方程集合 / data_gen / HDF5（部分核实）

**已核实**（`gh api repos/pdebench/PDEBench/git/trees/HEAD?recursive=1`，路径逐字）：

- 数据生成脚本：`pdebench/data_gen/data_gen_NLE/{AdvectionEq,BurgersEq,...}/`，含 `advection_exact_Hydra.py`、`advection_multi_solution_Hydra.py`、`burgers_Hydra.py`、`burgers_multi_solution_Hydra.py`
- Hydra 配置：`pdebench/data_gen/configs/{diff-react.yaml, diff-sorp.yaml, ns_incomp.yaml, radial_dam_break.yaml, mode/{debug,default,slurm}.yaml}`
- Advection 参数扫描配置：`.../AdvectionEq/config/args/beta{1e-1,1e0,1e1,2e-1,2e0,4e-1,4e0}.yaml`、`.../config/multi/config2D.yaml`
- 另有 `pdebench/data_download/config/config.yaml`（预生成数据下载入口）

⚠️ **2D shallow water 方程是否在 PDEBench 中：未核实**（我未在已抓取的文件树片段里见到 `shallow`/`swe` 路径——**不能据此断言"没有"**，树没抓全）。

### 2.7 sdmetrics 指标族（部分核实）

- `sdmetrics/reports/single_table/quality_report.py` 已读（974 B 头部），实际 import：
  ```python
  from sdmetrics.reports.base_report import BaseReport
  from sdmetrics.reports.single_table._properties import ColumnPairTrends, ColumnShapes
  ```
  → **QualityReport 由两个 property 组成：`ColumnShapes`、`ColumnPairTrends`**（对应「形状」与「列间关系」两族），这是逐字证据。
- ⚠️ `DiagnosticReport` 的 property 清单与两族下**具体指标名**（如 `BoundaryAdherence`、`CategoryAdherence`、`KeyUniqueness`、`Coverage` 等）**未逐字核到** → 见 §5。

---

## 3. 可复用 API 清单（带证据状态）

| 项目 | API（类/函数） | 文档/源码链接 | 证据状态 |
|---|---|---|---|
| SDV | `GaussianCopulaSynthesizer` | https://docs.sdv.dev/sdv | ⚠️ 待逐字核（见 §5） |
| SDV | `CTGANSynthesizer` / `TVAESynthesizer` | https://docs.sdv.dev/sdv | ⚠️ 待逐字核 |
| SDV | 条件采样 `sample_conditions(conditions, num_rows)` | https://docs.sdv.dev/sdv | ⚠️ 待逐字核（方法名待确认） |
| SDV | metadata 自动推断 `SingleTableMetadata.detect_from_dataframe(df)` | https://docs.sdv.dev/sdv | ⚠️ 待逐字核 |
| SDGym | 自定义 synthesizer 注册（`synthesizers` dict / `create_synthesizers`） | https://github.com/sdv-dev/SDGym | ⚠️ 待逐字核 |
| sdmetrics | `QualityReport`（property: `ColumnShapes` + `ColumnPairTrends`） | https://github.com/sdv-dev/sdmetrics/blob/main/sdmetrics/reports/single_table/quality_report.py | ✅ 已读源码头部 |
| sdmetrics | `DiagnosticReport` | https://github.com/sdv-dev/sdmetrics/blob/main/sdmetrics/reports/single_table/diagnostic_report.py | ⚠️ 待逐字核 |
| Synthcity | `TimeGANPlugin`（`src/synthcity/plugins/time_series/plugin_timegan.py`，L30） | 同上仓库路径 | ✅ 已读源码 |
| Synthcity | `TimeVAEPlugin` / `FFLOWS` 插件（文件存在） | `.../plugin_timevae.py`、`.../plugin_fflows.py` | ✅ 文件存在，类名未逐字核 |
| Synthcity | `Plugin` 基类（`fit / generate / fit_generate`） | `src/synthcity/plugins/core/plugin.py` | ⚠️ 待逐字核 |
| PDEBench | `eval_metrics` 实现被 RealPDEBench 直接引用为上游 | https://github.com/pdebench/PDEBench/blob/main/pdebench/models/metrics.py | ✅ README/docstring 引用 |
| RealPDEBench | `eval_metrics(pred, target, c, batch_size=None)` → 13 个张量 | https://github.com/AI4Science-WestlakeU/RealPDEBench/blob/main/realpdebench/utils/metrics.py | ✅ 已读全文 |
| RealPDEBench | `kinetic_energy(x)`，`x: [b,t,h,w,c]` | 同上 L15-L21 | ✅ 已读全文 |
| RealPDEBench | `probe_diagnostic(pred, target, d, center_x, center_y, sub_s_real, ...)` | 同上 L133 | ✅ 已读签名 |

---

## 4. 真实大气数据源方案

### 4.1 候选 A：WeatherBench2 ERA5（GCS，匿名）

- Bucket（已实测匿名 200）：`https://storage.googleapis.com/weatherbench2/datasets/era5/`
- 该前缀下共 **23 个** 数据集目录（`storage/v1/b/weatherbench2/o?prefix=datasets/era5/&delimiter=/` 实测 JSON 解析），含：
  - `1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2`
  - `1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr`
  - `1959-2023_01_10-6h-1440x721.zarr`（等
- **变量实测**（拉取 `.zmetadata` 并 `jq` 列举，35 个数组）：

  ```
  10m_u_component_of_wind 10m_v_component_of_wind 2m_temperature
  angle_of_sub_gridscale_orography anisotropy_of_sub_gridscale_orography
  geopotential geopotential_at_surface high_vegetation_cover lake_cover
  lake_depth land_sea_mask latitude level longitude low_vegetation_cover
  mean_sea_level_pressure sea_ice_cover sea_surface_temperature
  slope_of_sub_gridscale_orography soil_type specific_humidity
  standard_deviation_of_filtered_subgrid_orography
  standard_deviation_of_orography surface_pressure temperature time
  toa_incident_solar_radiation total_cloud_cover total_column_water_vapour
  total_precipitation type_of_high_vegetation type_of_low_vegetation
  u_component_of_wind v_component_of_wind vertical_velocity
  ```

  → **已验证不含** `clwc / ciwc / crwc / cswc`（逐个 `jq 'has()'` 检查，全部 false）；`tcwv` 以全称 `total_column_water_vapour` 存在。
  → 这独立印证了 Lead 的判断（Lead 说 36 个变量，我实测 35 个 `.zarray`，差异见 §5 计数口径）。

**最小切片读法**（需 `pip install xarray zarr gcsfs`）：
```python
import xarray as xr
ds = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
                  consolidated=True)
sub = ds["total_column_water_vapour"].sel(time="2020-01-01", latitude=slice(60,20), longitude=slice(100,140))
print(sub.shape, sub.attrs.get("units"))
```
匿名最小验证（不装库，只证明可匿名读）：
```bash
curl -s -m 30 -o /dev/null -w "%{http_code}\n" \
  "https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2/.zmetadata"
# 实测：200
```

### 4.2 候选 B：ARCO-ERA5（GCS，匿名）★推荐

- Bucket（实测匿名 200）：`https://storage.googleapis.com/gcp-public-data-arco-era5/`
- `ar/` 前缀下共 **19 个** 数据集目录，关键三个：
  - `ar/full_37-1h-0p25deg-chunk-1.zarr-v3/`（压力层 37 层，**变量最全**，132785 B `.zmetadata`）
  - `ar/model-level-1h-0p25deg.zarr-v1/`（**模型层**，85585 B `.zmetadata`）
  - `ar/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr/`

**压力层 `full_37-...zarr-v3` 实测含**（从 190+ 变量中筛出与云水/水汽相关的）：

| ERA5 短名 | ARCO 中的全名 | 是否存在 |
|---|---|---|
| `tcwv` | `total_column_water_vapour` | ✅ |
| `clwc` | `specific_cloud_liquid_water_content` | ✅ |
| `ciwc` | `specific_cloud_ice_water_content` | ✅ |
| — | `total_column_cloud_liquid_water` / `total_column_cloud_ice_water` / `total_column_rain_water` / `total_column_snow_water` / `total_column_supercooled_liquid_water` | ✅（柱积分量） |
| `crwc` | `specific_rain_water_content` | ❌ **压力层无** |
| `cswc` | `specific_snow_water_content` | ❌ **压力层无** |

**模型层 `model-level-1h-0p25deg.zarr-v1` 实测含（16 个变量，逐字）**：

```
divergence  fraction_of_cloud_cover  geopotential  hybrid  latitude  longitude
ozone_mass_mixing_ratio  specific_cloud_ice_water_content
specific_cloud_liquid_water_content  specific_humidity
specific_rain_water_content  specific_snow_water_content
temperature  time  u_component_of_wind  v_component_of_wind
vertical_velocity  vorticity
```

→ **四个云水变量 clwc/ciwc/crwc/cswc 全部命中**（全名对应关系：`specific_cloud_liquid_water_content`=clwc、`specific_cloud_ice_water_content`=ciwc、`specific_rain_water_content`=crwc、`specific_snow_water_content`=cswc），空间维度是 **`hybrid`（模型层 137 层）** 而非 `level`。
→ **这是本项目 PWV/云水资源方向唯一现成的四云水公开源**。

**最小切片读法**：
```python
import xarray as xr
# 压力层（含 clwc/ciwc + tcwv，适合压层训练）
ds_p = xr.open_zarr("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
                    consolidated=True)
# 模型层（含四云水，注意 vertical 维叫 hybrid）
ds_m = xr.open_zarr("gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1",
                    consolidated=True)
print(list(ds_m.data_vars))
```
匿名最小验证（实测命令与返回）：
```bash
curl -s -m 30 -o /dev/null -w "%{http_code}\n" \
  "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1/.zmetadata"
# 实测：200（v2 路径为 404，只有 v1）
curl -s -m 30 -o /dev/null -w "%{http_code}\n" \
  "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/.zmetadata"
# 实测：200
```

### 4.3 推荐与取舍

| 方案 | 云水变量 | 分辨率 | 读取成本 | 适用 |
|---|---|---|---|---|
| **ARCO-ERA5 压力层 v3** | clwc/ciwc + tcwv（无 crwc/cswc） | 0.25° × 37 层 × 1h | 小（Zarr 分块，可任意切片） | **主训练源**（压层、时次连续） |
| **ARCO-ERA5 模型层 v1** | **四云水全有** | 0.25° × 137 hybrid 层 × 1h | 大（137 层） | 云水专项研究、可选子集 |
| WeatherBench2 ERA5 | ❌ 仅 `total_cloud_cover`/`total_column_water_vapour` | 0.25° × 37 层 × 1h | 最小 | 仅作快速基线/压层对照，**不能作为云水主源** |

**建议**：主源 = ARCO-ERA5 压力层；云水专项 = ARCO 模型层（先只取 137 层里 20–60 层的湿层子集，控制内存；本机 8GB 内存必须分块 + 单时次切片）。WB2 只用于和已发表 WB2 基线（如 Pangu/GraphCast 复现）对齐。

---

## 5. 不确定 / 未核实清单

> 以下条目**没有**拿到正文或未逐字核对，计划书里引用前必须补。按优先级排序。

| # | 未核实项 | 为什么重要 | 建议补法 |
|---|---|---|---|
| 1 | **RealPDEBench 的 MVPE 指标定义**：`eval_metrics` 返回的 13 个量里没有名为 MVPE 的项；README 列了 9 个含 MVPE | 决定"借方法"时具体能借几个 | 读论文 https://realpdebench.github.io/metrics/ 或 `realpdebench/utils/` 其它文件（我未抓全） |
| 2 | **PDEBench 是否有 2D shallow water（`swe`）方程**：已抓文件树片段里未见，但树未抓全 | 大气相关方程的首选（浅水方程≈原始方程简化） | `gh api .../git/trees/HEAD?recursive=1` 全量 + grep `shallow\|swe` |
| 3 | **PDEBench HDF5 输出字段/形状**：未读到任何 `.h5` 写盘代码 | 决定能否复用其数据生成器 | 读 `pdebench/data_gen/data_gen_NLE/.../*_Hydra.py` 的 `h5py` 调用 |
| 4 | **synthcity 的 `torch<2.3 / numpy<2` pin 原文**：任务书给的 pin 未逐字核到（`pyproject.toml`/`setup.py` 未读） | 决定 Adapter 是否需要独立 conda 环境 | 读 https://raw.githubusercontent.com/vanderschaarlab/synthcity/main/pyproject.toml |
| 5 | **synthcity `(n, seq, feat)` 3D 约束的代码位置**：只从 `plugin_timegan.py` 内部 `temporal[0].shape` 反推，未见到显式断言 | Adapter 输入契约 | 读 `synthcity/utils/datasets/time_series/` 与 `TimeSeriesDataLoader` |
| 6 | **SDV / SDGym 具体类名与方法名**（`GaussianCopulaSynthesizer`、`sample_conditions`、SDGym 自定义 synthesizer 注册入口）：全部来自任务书/记忆，**本次未逐字核** | 直接复用清单的准确性 | 读 `sdv/synthesizers/` 与 SDGym `benchmark.py` |
| 7 | **sdmetrics DiagnosticReport 的 property 与具体指标名** | 决定质量指标层要不要自己写 | 读 `sdmetrics/reports/single_table/diagnostic_report.py` + `_properties/` |
| 8 | **WB2 变量数 35 vs Lead 报的 36**：我实测 `.zarray` 共 35 个（含 `latitude/level/longitude/time` 4 个坐标 → 物理变量 31 个）；差异可能是计数口径不同 | 证据一致性 | 同命令复跑即可（`_raw/wb2_*.zmetadata` 已落盘） |
| 9 | **WB2 bucket 的 `datasets/era5/LICENSE` 内容**：文件存在但未读 | 再分发数据时的合规 | `curl .../datasets/era5/LICENSE` |
| 10 | **ARCO 变量的维度顺序与单位**：只拿到 `.zarray` 的变量名，未读各变量 `.zattrs`（`units`、`long_name`）与 `.zarray` 的 `shape`/`chunks` | 预处理正确性 | 对单个变量 `curl .../<var>/.zattrs` |
| 11 | **Synthcity LICENSE 正文**：只拿到 PyPI/GitHub 两处 `Apache-2.0` 元数据，未逐字读 | 合规留档 | `curl raw.../synthcity/main/LICENSE` |
| 12 | **RealPDEBench HF 数据集的实际变量/字段名**：只从 README 拿到下载命令与场景名 | 若以后要借数据格式 | 读 `realpdebench/data/*_dataset.py` 或 HF dataset card |
| 13 | **PDEBench pip 包 pin 的逐字原文**：`requires_python=<3.11,>=3.9` 已从 PyPI JSON 核实；但 `torch~=1.13`、`hydra-core` 未逐字读 `setup.py` | 装包失败原因定位 | 读 `pdebench/PDEBench/main` 的 `setup.py` / `pyproject.toml` |

---

## 6. 复现本文件的命令（可审计）

```bash
# 仓库元数据（需 gh 登录）
gh api repos/sdv-dev/SDV --jq '{default_branch,stargazers_count,archived,pushed_at,license:.license.spdx_id}'
gh api "repos/sdv-dev/SDV/commits?per_page=1" --jq '.[0].commit.committer.date'
gh api repos/realpdebench/RealPDEBench            # 期望 404 Not Found

# 许可证原文
curl -sL https://raw.githubusercontent.com/sdv-dev/SDV/main/LICENSE
curl -sL https://raw.githubusercontent.com/pdebench/PDEBench/main/LICENSE.txt
curl -sL https://raw.githubusercontent.com/AI4Science-WestlakeU/RealPDEBench/main/LICENSE

# PyPI
curl -s https://pypi.org/pypi/sdv/json | jq -r '.info.version, .info.requires_python, .info.license_expression'

# Zarr 变量清单（无需 token）
curl -s "https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2/.zmetadata" \
  | jq -r '.metadata|keys[]|select(endswith("/.zarray"))|sub("/\\.zarray";"")'
curl -s "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1/.zmetadata" \
  | jq -r '.metadata|keys[]|select(endswith("/.zarray"))|sub("/\\.zarray";"")'
```

- 全部命令在本机（macOS arm64，python 3.13.13）2026-11-06 夜间实测通过；原始响应已存 `research/external/_raw/`。
- 未使用浏览器 CDP（GitHub/GCS 均匿名可达，无需登录态）。

---

## 7. 追加核实（同日 23:52 补，**覆盖/补全前文对应条目**）

### 7.1 sdmetrics：QualityReport 已被官方标记 deprecated（重要）

`sdmetrics/reports/single_table/quality_report.py` 全文仅 35 行，第 10 行逐字：

```python
_warn_deprecated_report('single table quality', 'QualityReport')
```

`DiagnosticReport` / `QualityReport` 同目录（`sdmetrics/reports/single_table/{diagnostic_report.py, quality_report.py}`）。
`_properties/` 下实际有 **8 个模块**（逐字文件树）：

```
_properties/{__init__.py, base.py, boundary.py, column_pair_trends.py,
             column_shapes.py, coverage.py, data_validity.py, structure.py, synthesis.py}
```

`QualityReport._properties` 字典逐字：

```python
self._properties = {
    'Column Shapes': ColumnShapes(),
    'Column Pair Trends': ColumnPairTrends(),
}
```

→ **修订 §2.7 / §3 判断**：`QualityReport` 两族指标是 `Column Shapes` + `Column Pair Trends`（已逐字确认），但该类**已被 sdmetrics 标记弃用**（`_warn_deprecated_report`）；`DiagnosticReport` 仍在，其 property 推测对应 `boundary / coverage / data_validity / structure / synthesis` 五个模块（**模块文件存在是逐字证据；「DiagnosticReport 用这五个」是推断，未读 diagnostic_report.py 正文**）。计划书里若要写指标层，应以 `sdmetrics.reports` 的新 API 为准，别照抄 `QualityReport`。

### 7.2 PDEBench 无 2D shallow water（证据增强，但仍非 100%）

全量文件树（`gh api repos/pdebench/PDEBench/git/trees/HEAD?recursive=1`，**完整递归，非片段**）grep `shallow|swe`：
**0 命中**。已确认存在的方程数据生成器：`AdvectionEq`、`BurgersEq`、`ReactionDiffusionEq`（含 16 组 `Rho/Nu` 参数扫描 yaml）、以及 `data_gen/configs/{diff-react, diff-sorp, ns_incomp, radial_dam_break}.yaml`。

→ **§5 第 2 项降级为「高度可能无」**：PDEBench 没有 2D shallow water；大气最接近的现成方程是 **1D advection（含 2D 配置 `config2D.yaml`）**、**2D diffusion-reaction**、**radial dam break（2D 浅水近似，实为 dam break）**。「`radial_dam_break.yaml` 用的是浅水方程求解器」**未读正文，不可断言**。
→ 这强化了「PDEBench 只借方法」的结论：大气方程仍须自建。

### 7.3 synthcity 依赖 pin：**未取到**

`https://raw.githubusercontent.com/vanderschaarlab/synthcity/main/pyproject.toml` 抓取后 grep `torch|numpy|pandas|requires-python` **无输出**（文件可能存在但未含内联 pin，或抓取失败未校验）。→ **§5 第 4 项仍未核实**，任务书给的 `torch<2.3 / numpy<2 / pandas>=2.1` **不能当作已核实事实引用**。

### 7.4 其它增量事实

- `gh api` 认证后速率：`remaining=4974/5000`（未受限）；匿名 GitHub API 在本次网络下出现过 **HTTP 504**，故全程改用 `gh api`，该差异已记录在 §6 复现命令。
- `ar/model-level-1h-0p25deg.zarr-v2` → **404**；模型层只有 **v1**（这一个细节已在 §4.2 写入，此处再确认一次）。
- WB2 全量目录共 **23 个 `.zarr` 数据集**（列目录实测，含 `LICENSE` 文件 1 个）；`_raw/wb2_listing.json` 可复算。
- PDEBench `pdebench/data_download/config/config.yaml` 是预生成数据下载入口（路径逐字存在），若不想跑生成器可从这里下。
