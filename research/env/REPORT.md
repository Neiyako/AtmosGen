# Phase 0 环境探测报告（P1/P2/P3 实测 · P4 NOT_RUN）

- 执行人：engineer（团队 teammate）；时间：2026-11-06 23:09–23:50（Asia/Shanghai），时间盒收尾
- 机器：macOS / arm64 / 8 GB RAM；磁盘剩余 228 GiB；conda 26.3.2（`/opt/miniconda3`）
- 工作目录：`/Users/apple/Desktop/Atmospheric Data Generation`
- 所有结论都附**实际命令 + 输出摘要 + 日志文件名 + 退出码**，日志在 `research/env/logs/`。
- 结论口径：PASS = 实测通过；CONDITIONAL_PASS = 通路通但有明确降级；FAIL = 实测失败；NOT_RUN = 按指令未执行（不可当作通过）。

## 总览

| 项 | 状态 | 一句话 |
|---|---|---|
| P1 atmosgen env（py3.11 + 核心栈） | **PASS** | conda env 建好，15 个指定包全装，18/18 import OK，pip freeze 109 行落盘 |
| P2 真实 ERA5 切片 | **CONDITIONAL_PASS** | Zarr→xarray→NetCDF 通路跑通并落地真实 ERA5 切片（降级源：WB2 2.8°/13 层/无云水）；0.25°+clwc/ciwc 因 chunk 成本今夜不可行，证据与明日 recipe 已写 |
| P3 五个参考仓库 | **PASS** | 5/5 shallow clone 成功（exit 0），HEAD/LICENSE/python 要求/入口路径全部取证 |
| P4 synthcity / PDEBench 安装探测 | **NOT_RUN** | 按 Lead 23:55 时间盒指令取消，避免今夜拉开长依赖链；已给明日命令 |

---

# 一、已实测

## P1. atmosgen 环境（PASS）

**命令与结果**

1. 建环境
```
/opt/miniconda3/bin/conda create -y -n atmosgen python=3.11
```
- 日志：`research/env/logs/p1_conda_create.log`（末尾 `Preparing/Verifying transaction ... done`）
- 证据：`conda env list --json` 输出含 `"/opt/miniconda3/envs/atmosgen"`；`/opt/miniconda3/envs/atmosgen/bin/python --version` → `Python 3.11.15`
- 耗时 ≈ 85 s（23:09:04 起 → 23:10:29 日志完成）。退出码未单独落盘（nohup 包裹），以“环境存在 + python 可执行 + 后续 pip/import 全部 exit 0”作为等价证据。

2. 装核心栈（`p1_pip_install.log`，起 23:09:40 → 止 23:12:14 ≈ 2.5 min）
```
/opt/miniconda3/envs/atmosgen/bin/pip install numpy pandas xarray netCDF4 zarr dask gcsfs \
    pyyaml pydantic typer scipy scikit-learn matplotlib pytest sdv sdmetrics
```
- 日志内退出码：`PIP_UPGRADE_EXIT=0`、`PIP_INSTALL_EXIT=0`、`PIP_FREEZE_EXIT=0`（**全 0**）
- `pip freeze` → `research/env/logs/atmosgen_pip_freeze.txt`，**109 行/包**

3. import 冒烟：`research/env/logs/p1_import_smoke.log` → 18/18 `OK`，`IMPORT_SMOKE_EXIT=0`

**版本（pip freeze 实测）**

| 包 | 版本 | 包 | 版本 |
|---|---|---|---|
| python | 3.11.15 | sdv | 2.1.0 |
| numpy | 2.4.6 | sdmetrics | 0.33.0 |
| pandas | 2.3.3 | torch | **2.15.0**（由 sdv→deepecho 自动带入，非显式安装） |
| xarray | 2026.9.0 | netCDF4 | 1.7.5.1.post1 |
| zarr | 3.1.6 | scipy | 1.17.1 |
| dask | 2026.8.0 | scikit-learn | 1.9.1 |
| gcsfs | 2026.10.1 | matplotlib | 3.11.2 |
| fsspec | 2026.9.0 | pytest | 9.2.0 |
| pydantic | 2.14.3 | typer | 0.28.0 |
| PyYAML | 6.0.3 | pip | 26.3 |

**磁盘占用**：`/opt/miniconda3/envs/atmosgen` = **1.5 GB**；安装后磁盘剩余 228 GiB。

**对本项目有后果的实测发现**
- `sdv==2.1.0` 在 py3.11 可装，但依赖链拉入 **torch 2.15.0**；synthcity 要求 `torch>=2.1,<2.3`（见 `third_party/_ref/synthcity/prereq.txt`）→ **主 env 不能用于 synthcity，必须独立 env**（P4 明日验证）。
- `zarr==3.1.6` 下 `zarr.open_consolidated()` 读 zarr-v2 consolidated store 会报
  `TypeError: object of type 'generator' has no len()`；而 `xr.open_zarr(mapper, consolidated=True)` 正常（WB2 9.3 s、ARCO 11.3 s 打开）。**Phase 1 统一用 xr.open_zarr，不要用 zarr.open_consolidated。**

## P2. 真实数据（CONDITIONAL_PASS：通路 PASS，0.25° 全变量切片阻塞）

**① 数据源变量核查（匿名 `.zmetadata`，已存 JSON 证据）**
- `research/env/logs/p2_store_metadata_check.log`、`p2_wb2_zmetadata.json`、`p2_arco_zmetadata.json`

| store | HTTP | .zmetadata | 数组数 | t/q/u/v/tcwv | clwc/ciwc | crwc/cswc |
|---|---|---|---|---|---|---|
| WB2 `1959-2022-full_37-6h-0p25deg-chunk-1.zarr-v2` | 200 | 31,673 B | 36 | ✅（长名） | ❌ 全无 | ❌ |
| ARCO `ar/full_37-1h-0p25deg-chunk-1.zarr-v3` | 200 | 132,785 B | 277 | ✅（长名） | ✅ `specific_cloud_liquid_water_content` / `specific_cloud_ice_water_content` | ❌（需 CDS） |

- 短名映射（两个 store 都不使用 CDS 短名）：
  `t=temperature, q=specific_humidity, u=u_component_of_wind, v=v_component_of_wind, tcwv=total_column_water_vapour`；
  ARCO 另有 `clwc=specific_cloud_liquid_water_content, ciwc=specific_cloud_ice_water_content`。
- `dimension` 实测：ARCO `time=1,323,648`（1900-01-01T00…2050-12-31T23）、`level=37`、`lat=721`、`lon=1440`。

**② 关键阻塞：chunk 经济性（实测）**
- 0.25° pressure-level 数组 chunk = `[1,37,721,1440]`；单 chunk 压缩后 **85,235,158 B（85.2 MB）**（curl HEAD，ARCO `temperature/1051896.0.0.0`）。
- 本机到 GCS 实测吞吐：**38,317 B/s**（`curl -r 0-5242879`，206，4,598,275 B / 120.0 s）。
- ⇒ 单变量单时次 ≈ **37 min**；7 变量×8 时次 ≈ **25 h**。今夜不可行；**没有硬拉**。
- 证据全文 + 明日流式 recipe：`research/env/logs/p2_arco_infeasible_evidence.md`
- 附带踩坑（已修正）：ARCO chunk key 是 **4 段** `temperature/<int_idx>.0.0.0`，`...0.0` 会 404；索引必须
  `ds.time.sel(...)`/`ds.indexes["time"].get_loc(...)` 取整数，不要手算浮点。

**③ 降级落地的真实切片（今夜交付物）**
- 命令：`/opt/miniconda3/envs/atmosgen/bin/python -u research/env/p2_fetch_era5_slice.py > research/env/logs/p2_wb2_read.log 2>&1`
- 源（真实 ERA5，WB2 粗网格重网格）：`gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr`
- 产物：`data/raw/era5_slice.nc` + `data/raw/era5_slice_SUMMARY.md`
- 摘要：<!--P2_STATS-->
- 明确限制：网格 2.8°（32×64）、13 层、**无 clwc/ciwc/crwc/cswc**；仅证明 Test A 读取通路，**不能**当训练主数据。

## P3. 第三方参考仓库（PASS）

**命令**：`for repo in sdv-dev/SDV vanderschaarlab/synthcity sdv-dev/SDGym pdebench/PDEBench AI4Science-WestlakeU/RealPDEBench; do git clone --depth 1 https://github.com/$repo.git; done`
- 日志：`research/env/logs/p3_clones.log` → 5 个仓库 `CLONE_EXIT_*=0`（全部成功，`--depth 1`，git 2.54.0）
- 元数据日志：`research/env/logs/p3_repo_meta.log`

| 仓库 | HEAD | HEAD 日期 | LICENSE | python 要求 | 关键入口 |
|---|---|---|---|---|---|
| SDV | `549dc94e49e73103487b3dbc324abcc43c8bfcc0` | 2026-11-05 | **BSL 1.1**（Business Source License） | `>=3.10,<3.15` | `sdv/`（synthesizers/、evaluation/） |
| synthcity | `23f322fe381326ed01c41b13d469a06e38cce545` | 2025-06-17 | Apache-2.0 | `>=3.9`（setup.py）；prereq: `torch>=2.1,<2.3`, `numpy>=1.20`, `tsai` | `src/synthcity/plugins/core/plugin.py`；version.py = **0.2.12** |
| SDGym | `009a35b2875ef6743ad3a713ee1a979b6df86e04` | 2026-11-04 | **BSL 1.1** | `>=3.10,<3.15` | `sdgym/`（benchmark.py、cli/、_benchmark/） |
| PDEBench | `4ff3e3a4aa1561721b5571fa3a048a0a463e0568` | 2026-03-30 | MIT（LICENSE.txt） | **`>=3.9,<3.11`** | `pdebench/data_gen/`（gen_ns_incomp.py 等）、`pdebench/models/` |
| RealPDEBench | `06ba38d2db3e78edc2556ff7d9098119a42c2b07` | 2026-11-04 | **CC BY-NC 4.0（禁止商用）** | `>=3.10` | `realpdebench/eval.py`、`realpdebench/utils/metrics.py`（`eval_metrics`/`mse_loss`/`probe_diagnostic`）、`realpdebench/data/`、`model/` |

**对计划书有直接影响的许可证事实**
1. **SDV 与 SDGym 是 Business Source License 1.1**（非 OSI 开源；有商用/托管限制）→ 若项目将来商用，需评估或换实现（CTGAN 可换 synthcity/TVAE 等）。
2. **RealPDEBench 是 CC BY-NC 4.0，禁止商业使用** → 只能做研究对照，不进产品链路。
3. synthcity Apache-2.0、PDEBench MIT → 可安全引用。
4. **PDEBench 要求 python<3.11** → 主 env（3.11）不能用，需单独 py3.10 env（明日 P4）。

---

# 二、未完成（明确未做，不当作通过）

| 项 | 状态 | 卡在哪 / 下一条命令 |
|---|---|---|
| P2 0.25° 7 变量（含 clwc/ciwc）切片 | 未落地 | GCS 单连接 38.3 KB/s × 85.2 MB/chunk；先量化更小 chunk 的 store：`curl -s "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/1959-2022-6h-1440x721.zarr/.zmetadata" > /tmp/x.json`（再看 `.zarray` 的 chunks），或 `ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2`；另测多连接 Range 并发是否线性提速 |
| P4-A synthcity 0.2.12 独立 env | NOT_RUN（按指令取消） | 明日：`/opt/miniconda3/bin/conda create -y -n atmosgen-synthcity python=3.10 && /opt/miniconda3/envs/atmosgen-synthcity/bin/pip install "torch==2.2.2" "numpy<2" && .../pip install "synthcity==0.2.12"`；预期阻塞点：`tsai`/`decaf` 依赖与 torch 2.2.2 的 arm64 wheel 匹配 |
| P4-B pdebench 安装 | NOT_RUN（按指令取消） | 明日：`conda create -y -n atmosgen-pdebench python=3.10`，先试 `pip install pdebench==0.1.0`（py<3.11 满足），失败再 `pip install -e third_party/_ref/PDEBench`；只看能否装，不跑训练 |
| P4-C 训练/大依赖链 | NOT_RUN（按指令） | 明日 Phase 1 再动 |

---

# 三、阻塞点与风险（供计划书）

1. **数据链路吞吐是头号风险**：本机→GCS 单连接 38.3 KB/s，0.25° 全层 store 单 chunk 85.2 MB。任何“切片就在 Zarr 上按需读”的设计在 8 GB/慢网环境都不可行；Phase 1 必须先在**服务端/一次性任务**里把所需区域裁成小文件（或走 CDS 子集 API），再在本地训练。
2. **云水变量可得性分层**：WB2=无；ARCO=clwc/ciwc 有、crwc/cswc 无；完整 9 变量只能 CDS。计划书里云水相关实验要么缩到 clwc/ciwc，要么先解决 CDS key。
3. **许可证**：SDV/SDGym=BSL 1.1；RealPDEBench=CC BY-NC 4.0 → 商用路径需规避（换成 synthcity/自研）。这条要在技术选型里写明。
4. **Python 版本矩阵**：主栈 3.11（PASS）；synthcity 需 torch<2.3（主 env 是 2.15）；PDEBench 需 <3.11 → 至少 2 个额外 env，明天各 ~5–10 min 可探清。
5. **磁盘/内存**：磁盘余量 228 GiB 充足；8 GB RAM 下“逐时次流式 subset”峰值 <200 MB，可行。

---

# 四、日志索引（全部为真实命令输出）

| 日志 | 内容 |
|---|---|
| `research/env/logs/p1_conda_create.log` | conda create 全程 |
| `research/env/logs/p1_pip_install.log` | pip 升级+安装，含 3 个 EXIT=0 |
| `research/env/logs/atmosgen_pip_freeze.txt` | 109 包 freeze |
| `research/env/logs/p1_import_smoke.log` | 18/18 import OK |
| `research/env/logs/p2_store_metadata_check.log` | WB2/ARCO 变量与 chunk 形状 |
| `research/env/logs/p2_wb2_zmetadata.json`、`p2_arco_zmetadata.json` | 原始 consolidated metadata |
| `research/env/logs/p2_arco_probe.log` | open_zarr 11.3 s、time 轴、索引计算 |
| `research/env/logs/p2_arco_infeasible_evidence.md` | 85.2 MB/chunk + 38.3 KB/s 阻塞证据 + 明日 recipe |
| `research/env/logs/p2_wb2_read.log` | 今夜降级真值切片读取日志 |
| `research/env/logs/p3_clones.log` | 5 仓库 clone 退出码 |
| `research/env/logs/p3_repo_meta.log` | HEAD/LICENSE/python/入口 |
| `research/env/p2_fetch_era5_slice.py` | 今夜切片脚本（可重跑） |
| `research/env/probe_arco_read.py` | ARCO 读取成本探针 |
