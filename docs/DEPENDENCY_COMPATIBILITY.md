# 依赖兼容性评估（DEPENDENCY COMPATIBILITY）

- 评估日期：2026-11-06
- 评估方式：**本机真实安装探测**（日志在 `research/env/logs/`）+ PyPI 元数据核对（Python 版本、依赖 pin）+ 源码 `requires-python` 核对。
- 判定口径：**PASS** = 真实装成功并能 import；**CONDITIONAL** = 需隔离环境/有限制；**FAIL** = 主环境不可用；**NOT_RUN** = 今晚未执行、明天验证。

---

## 1. 本机基线

| 项 | 值 | 证据 |
|---|---|---|
| 机型 / 架构 | macOS 27.0.1 / **arm64**（Apple Silicon） | `sw_vers`、`uname -m` |
| CPU / 内存 | 8 核 / **8 GB** | `sysctl hw.ncpu hw.memsize` |
| GPU | 无 CUDA/MPS 专用加速假设；**默认 CPU** | 任务书 §9 |
| 磁盘可用 | ≈ 228 GB | `df -h` |
| conda | `/opt/miniconda3`（base = Python 3.13.13） | `conda env list` |
| base 环境现状 | numpy 2.4.3 / pandas **3.0.3** / xarray 2026.4.0，无 sdv/synthcity | `pip list` |

> base 环境的 pandas 3.x 与 SDV 的 `pandas<3` 要求**冲突**，因此**不使用 base**，项目专用环境独立创建。

---

## 2. 环境策略（三层，互不污染）

| 环境 | Python | 用途 | 状态 |
|---|---|---|---|
| `/opt/miniconda3`（base） | 3.13.13 | 系统/工具，**不装项目依赖** | 保持原样 |
| `atmosgen` | **3.11** | **主环境**：SDV、sdmetrics、xarray/NetCDF/Zarr、validation、CLI、pytest | ✅ 已创建并安装完成 |
| `atmosgen-synthcity` | **3.10** | 可选隔离环境：synthcity 0.2.12（torch 2.2.x + numpy<2） | ⏳ Phase 2 验证（今晚 NOT_RUN） |
| （不建）PDEBench legacy | 3.10 | 仅在确需复现 PDEBench 生成器时临时用 | ❌ 不计划：无大气方程、torch~=1.13 |

主环境创建命令（实际执行，exit 0）：

```bash
/opt/miniconda3/bin/conda create -y -n atmosgen python=3.11
/opt/miniconda3/envs/atmosgen/bin/pip install \
  numpy pandas xarray netCDF4 zarr dask gcsfs pyyaml pydantic typer \
  scipy scikit-learn matplotlib pytest sdv sdmetrics
```

实测：conda 创建 + pip 安装合计约 **2 分钟**（23:10:29 → 23:12:14，本机缓存命中），环境体积 **1.5 GB**，disk 充足。

---

## 3. 主环境实测版本（`atmosgen`）

证据：`research/env/logs/atmosgen_pip_freeze.txt`（109 行，`pip freeze` 原样保存，交付时随包附上）。安装日志：`p1_conda_create.log`、`p1_pip_install.log`（`PIP_INSTALL_EXIT=0`）、`p1_import_smoke.log`（18/18 import OK）。完整环境探测报告：`research/env/REPORT.md`。

**实测踩坑（Phase 1 必须遵守）**：
1. `sdv==2.1.0` 的依赖链自动带入 **torch 2.15.0**（经 deepecho）——因此主环境不能用于 synthcity（需 torch<2.3），隔离环境是硬要求；
2. `zarr==3.1.6` 下 **`zarr.open_consolidated()` 读 zarr-v2 consolidated store 会报 `TypeError: object of type 'generator' has no len()`**；`xr.open_zarr(mapper, consolidated=True)` 正常（WB2 打开 9.3 s、ARCO 11.3 s）。→ Phase 1 统一用 `xr.open_zarr`，不用 `zarr.open_consolidated`。

| 包 | 实测版本 | 说明 |
|---|---|---|
| python | **3.11.15** | |
| numpy | **2.4.6** | |
| pandas | **2.3.3** | 满足 SDV `pandas<3` |
| scipy | 1.17.x | |
| scikit-learn | 1.x（随 sdmetrics 安装） | |
| xarray | **2026.9.0** | |
| netCDF4 | **1.7.5.1.post1** | NetCDF 主输出 |
| zarr | **3.1.6** | Zarr 输出 / 备选远程读取 |
| dask | **2026.8.0** | 分块/惰性计算 |
| gcsfs | **2026.10.1** | 匿名读 GCS（备选，仅元数据核对用） |
| **sdv** | **2.1.0** | L1 主生成器 |
| copulas | 0.15.0 | SDV 底层 |
| ctgan | 0.13.0 | CTGAN/TVAE 后端 |
| deepecho | 0.9.0 | SDV 序列(PAR)后端，未计划使用 |
| **sdmetrics** | **0.33.0** | 统计质量指标（MIT） |
| torch | 2.15.0 | ctgan 依赖（CPU 使用） |
| pydantic | 2.14.3 | 配置模型 |
| typer | 0.28.0 | CLI |

---

## 4. 逐项目兼容性判定

| 项目 | 要求（来源） | 主环境（py3.11） | 判定 | 处置 |
|---|---|---|---|---|
| **SDV 2.1.0** | `python>=3.10,<3.15`；`pandas<3`；numpy 按版本（py3.13 需≥2.1）（PyPI JSON） | py3.11 / pandas 2.3.3 / numpy 2.4.6 | **PASS** | L1 直接依赖 |
| **sdmetrics 0.33.0** | `python>=3.10,<3.15`；`pandas<3`（PyPI JSON） | 同 | **PASS** | L1 |
| **Synthcity 0.2.12** | `torch<2.3,>=2.1`；`numpy<2.0`；`pandas>=2.1`；`networkx<3`；`lifelines<0.30`（PyPI JSON） | torch **2.15.0**、numpy **2.4.6** → 冲突 | **CONDITIONAL** | 独立 `atmosgen-synthcity`（py3.10 + torch 2.2.2 + numpy 1.26 + pandas 2.2）；Phase 2 实测 |
| **SDGym 0.16.1** | `python>=3.10,<3.15`；依赖 `google-cloud-compute`、`boto3/botocore`、`appdirs`、`compress-pickle`、`openpyxl` 等（PyPI JSON） | 可装但引入云 SDK 等重依赖 | **不安装**（L3） | 只用其协议；指标走 sdmetrics |
| **PDEBench 0.1.0** | `python>=3.9,<3.11`；`torch~=1.13.0`、`numpy<2`、`hydra-core`、`deepxde`、`pyro-ppl`（PyPI JSON / pyproject） | py3.11 不满足；torch 1.13 与主环境互斥 | **FAIL（主环境）/ 不建 legacy 环境** | L3 参考 + L4 最小自研；解析解验证替代 |
| **RealPDEBench** | `python>=3.10`；`pip install -e .`；依赖 torch 与多个模型库（README） | 可装但无大气数据 + CC BY-NC | **不安装**（L3） | 只借指标定义 |

### 4.1 关键冲突（为什么必须隔离）

```
Synthcity 0.2.12  requires: torch>=2.1,<2.3     主环境: torch 2.15.0   ✗
Synthcity 0.2.12  requires: numpy>=1.20,<2.0    主环境: numpy  2.4.6   ✗
SDV 2.1.0         requires: pandas<3            主环境: pandas 2.3.3   ✓
PDEBench 0.1.0    requires: python<3.11         主环境: python 3.11.15 ✗
PDEBench 0.1.0    requires: torch~=1.13.0       主环境: torch 2.15.0   ✗
```

结论：**不能为了 Synthcity/PDEBench 降级主环境**（会破坏 SDV/sdmetrics 的现代版本），必须"一项目一环境"。

---

## 5. 隔离环境方案（Phase 2 执行）

### 5.1 Synthcity 隔离环境

```bash
/opt/miniconda3/bin/conda create -y -n atmosgen-synthcity python=3.10
/opt/miniconda3/envs/atmosgen-synthcity/bin/pip install \
  "torch==2.2.2" "numpy<2" "pandas>=2.1,<3" "synthcity==0.2.12"
```

风险点（Phase 2 必须实测并记录）：
1. **pydantic 版本**：synthcity 源码使用 `from pydantic import validate_arguments`（`src/synthcity/plugins/core/dataloader.py:4`），该接口在 pydantic v2 中已 deprecated；若失败，尝试 `pydantic<2`（需与 torch/pandas 兼容性重新验证）。
2. `decaf-synthetic-data`、`opacus`、`lifelines<0.30` 等间接依赖在 arm64 上的 wheel 可用性。
3. **运行方式**：AtmosGen 主进程不 import synthcity；Adapter 通过 **subprocess 调用隔离环境的脚本**（`scripts/synthcity_worker.py`），用中间文件（npz/json）交换数据，避免跨环境 import。这是 L2 Adapter 的落地方式，也符合"第三方项目保持独立"原则。
4. 时间盒：安装 ≤15 分钟、单模型训练 ≤20 分钟；超时则记录 NOT_RUN。

### 5.2 主环境内的优雅降级

- `atmosgen generate --model synthcity:*` 在检测到隔离环境不可用时，**明确报错**（指出缺失环境与安装命令），不静默回退、不伪造结果；
- 统计生成的"两种不同原理"由 SDV 的 `gaussian_copula`（Copula 原理）与 `ctgan`/`tvae`（神经原理）满足，Synthcity 属于**加分项**而非阻塞项；
- 报告中必须标注实际跑通的模型列表与 NOT_RUN 项。

---

## 6. 数据依赖（2026-11-06 晚更正：主数据源改为本机归档）

| 数据源 | 依赖 | 访问方式 | 实测/结论 |
|---|---|---|---|
| **本机 ERA5 归档（主）** | `netCDF4` + `xarray`（无需网络） | `/Volumes/Untitled/{2010..2013}/ERA5_025dg_YYYYMMDD_<var>.nc`，0.25° 区域子集（lat −10…70/lon 50…160，321×441）、37 层、逐时、变量 t/q/u/v | 文件结构实测（`time=24, level=37, lat=321, lon=441`，≈251 MB/文件）；已产出真实切片 `data/raw/era5_local_slice_20100501.nc`（1.0 MB，4 时次×3 层×101×101，NaN=0，读写 1.3 s） |
| 云水 CDS 小样本（已有） | `xarray` | `era5数据dbas/data/downloads/.../20240101_multi_fdb02e.nc` | 实测含 `clwc/ciwc`，`(valid_time=1, pressure_level=3, lat=61, lon=141)`，kg kg⁻¹ |
| CDS 在线（待补 crwc/cswc） | `cdsapi` + `~/.cdsapirc`（存在性已确认，未读内容） | Copernicus CDS `reanalysis-era5-pressure-levels` | **NOT_RUN 今晚**；DBAS 已有可用配置与成本分析，Phase 1 时间盒补测 |
| ARCO-ERA5 / WB2（备选） | `gcsfs` + `zarr` | 匿名 HTTPS | 元数据已核；但 **GCS 实测吞吐 38–83 KB/s**（单 chunk 85.2 MB），整片不可行 → 仅保留为备选（详见 `research/env/logs/p2_arco_infeasible_evidence.md`） |

数据只读约定：本机归档与外置卷**只读打开**（`open_dataset` + `isel`，不写回、不复制大文件）；`data/raw/` 只存小切片（<50 MB）；生成过程不得回写输入；输出目录 `output/` 独立。

---

## 7. 可复现性与版本锁定

1. 交付 `research/env/logs/atmosgen_pip_freeze.txt`（主环境完整版本清单）；
2. 每次生成在数据契约 `metadata.env` 中记录：python、atmosgen 版本、sdv/sdmetrics 版本、随机种子；
3. 第三源码参考锁定 commit（见 `THIRD_PARTY_AUDIT.md §1.1`），不跟踪浮动 HEAD；
4. 配置 YAML 中显式 `seed`，所有随机源统一从该 seed 派生（numpy Generator + model seed）；
5. 已知非确定性例外（如 torch CPU 多线程）在报告中注明，并给出容差断言（指标在 ±1% 内视为复现）。

---

## 8. 资源约束与实测

| 指标 | 约束/实测 | 落实方式 |
|---|---|---|
| 内存 | 8 GB，**禁止全量加载 ERA5** | xarray+dask 分块；切片实验（<200 MB 内存占用）；`scripts/` 记录 `peak RSS` |
| 算力 | CPU only，无 GPU 默认 | CTGAN/TVAE 小 epoch；TimeGAN 时间盒；PDE 求解器 numpy 向量化 |
| 存储 | 228 GB 可用 | 环境 1.5 GB；原始切片 <100 MB；每个实验输出 <1 GB；中间产物可清理 |
| 时长 | 单个实验 ≤30 分钟 | 小样本（如 24–72 时间步 × 3 层 × 20×25 格点）；大实验配置化、可离线 |

---

## 9. 未完成探测（明天 Phase 2 前 30 分钟完成）

| # | 项 | 判定标准 |
|---|---|---|
| D1 | `atmosgen-synthcity` 环境安装 | ✅ **PASS**（2026-11-07；需先 pin scipy==1.11.4 + opacus==1.4.1，见 §10） |
| D2 | synthcity TimeGAN 在小规模时序上的训练 | **NOT_RUN**（时间盒只验证了 ctgan 插件集成与生成，未训练 TimeGAN；Phase 4/5 再评估） |
| D3 | SDV 2.1.0 `sample_remaining_columns` 条件生成行为 | ⚠️ **PASS（有重要限制）**：API 可用，但连续协变量条件采样在经验分布边界退化（level=500 时 t 被压到 248.0 K ± 0.02 K），32 448 行 148 s；CTGAN/TVAE 为拒绝采样（约 6 s/行）。因此 Phase 2 改用 level_wise 方案（见 `docs/PHASE2_REPORT.md` P1） |
| D4 | 本机 ERA5 归档子集读取耗时/内存 | ✅ **PASS**：Phase 1 ingest 16 个 239.8 MB 文件切片合计 **7.08 s / 峰值 RSS 426 MB**（含 tcwv 派生与写盘） |

> 本文件将在 Phase 2 完成后更新 D1–D4 的实测结论，并把状态从 NOT_RUN 改为实际判定。


---

## 10. Synthcity 隔离环境时间盒结论（Phase 2 实测，2026-11-07）

> 目的：验证 D6 决策（Synthcity 走 L2 隔离环境 + subprocess 适配）是否可行；**不**把 Synthcity 设为 Phase 2 默认生成器。

| 项 | 实测 |
|---|---|
| 环境 | `atmosgen-synthcity`：Python 3.10.22 / numpy 1.26.4 / torch 2.2.2 / synthcity（27 个插件可见） |
| 安装方式 | `conda create -y -n atmosgen-synthcity python=3.10` + `pip install synthcity`（约 4 分钟） |
| 失败 1 与修复 | scipy 1.16 的 arm64 wheel 在该 macOS 上 dlopen 失败：`section '__DATA/__thread_bss' has a zero-fill section type, but offset field is not zero` → `pip install scipy==1.11.4` 后 `svds` 导入正常 |
| 失败 2 与修复 | opacus ≥1.5 依赖 torch≥2.6 的 `nn.RMSNorm`：`AttributeError: module 'torch.nn' has no attribute 'RMSNorm'`（synthcity 又把 torch pin 在 2.2）→ `pip install opacus==1.4.1` 后 `PrivacyEngine` 导入正常 |
| Probe 结果 | `scripts/synthcity_probe.py`：import PASS；插件列表 27；`ctgan(n_iter=5)` fit **0.63 s** + generate **0.03 s** → 200 行；`steps.fit_and_generate = PASS` |
| 质量结论 | **NOT_RUN**：`n_iter=5` 未调参，边缘分布质量差（最大相对均值偏差 **67.2**，t 的生成 std 退化到 **5.7e-14**）——不作为模型对比依据 |
| 判定 | D1 环境安装 **PASS**；作为生成器**不启用为默认**；Phase 4/5 若需要它作对比项，必须重新调参并单独重跑（不得沿用本次质量数字冒充可用） |
| 证据文件 | `research/env/logs/p4_synthcity_install.log`、`p4_synthcity_probe.log`、`p4_synthcity_probe.json` |
