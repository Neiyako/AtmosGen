# Phase 0 环境探测报告（P1 / P2 / P3 实测 · P4 NOT_RUN）

- 执行人：engineer；时间：2026-11-06 23:09–23:50（Asia/Shanghai），按 Lead 指令时间盒收尾
- 机器：macOS / arm64 / 8 GB RAM；磁盘剩余 228 GiB；conda 26.3.2（`/opt/miniconda3`）；git 2.54.0
- 工作目录：`<repo>`
- 口径：PASS=实测通过；CONDITIONAL_PASS=通路通但有明确降级；FAIL=实测失败；NOT_RUN=按指令未执行（**不可当通过**）。
- 全部结论附**实际命令 + 输出摘要 + 日志文件名 + 退出码**；日志在 `research/env/logs/`。

## 总览

| 项 | 状态 | 一句话 |
|---|---|---|
| P1 atmosgen env（py3.11 + 核心栈） | **PASS** | env 建好；15 个指定包全装；18/18 import OK；pip freeze 109 行落盘；env 1.5 GB |
| P2 真实 ERA5 切片 | **CONDITIONAL_PASS** | GCS/Zarr 方案实测失败（38.3 KB/s × 85.2 MB/chunk）→ 改用**本机真实 ERA5**，已落地 4 变量×4 时次切片 1.0 MB + clwc/ciwc 云水样本 45.9 KB |
| P3 五个参考仓库 | **PASS** | 5/5 shallow clone exit 0；HEAD/LICENSE/python 要求/入口路径全部取证；发现 BSL 1.1 与 CC BY-NC 许可证风险 |
| P4 synthcity / PDEBench 安装探测 | **NOT_RUN** | 按 Lead 时间盒指令取消（避免拉开长依赖链）；已给明日命令 |

---

# 一、已实测

## P1. atmosgen 环境（PASS）

**① conda 建环境**
```
/opt/miniconda3/bin/conda create -y -n atmosgen python=3.11
```
- 日志 `research/env/logs/p1_conda_create.log`（末尾 `Preparing transaction ... done` / `Verifying transaction ... done`）
- 证据：`/opt/miniconda3/envs/atmosgen/bin/python --version` → `Python 3.11.15`；`conda env list --json` 输出含 `/opt/miniconda3/envs/atmosgen`
- 耗时 ≈ 85 s（23:09:04 起 → 23:10:29 日志落定）。退出码未单独落盘（nohup 包裹），以「env 存在 + python 可执行 + 后续 pip/import 全部 exit 0」为等价证据。

**② 装核心栈**（日志 `p1_pip_install.log`，起 23:09:40 → 止 23:12:14 ≈ 2.5 min）
```
/opt/miniconda3/envs/atmosgen/bin/pip install numpy pandas xarray netCDF4 zarr dask gcsfs \
    pyyaml pydantic typer scipy scikit-learn matplotlib pytest sdv sdmetrics
```
- 日志内退出码：`PIP_UPGRADE_EXIT=0`、`PIP_INSTALL_EXIT=0`、`PIP_FREEZE_EXIT=0`（**全 0**）
- `pip freeze` → `research/env/logs/atmosgen_pip_freeze.txt`（**109 行/包**）
- **磁盘占用**：`/opt/miniconda3/envs/atmosgen` = **1.5 GB**；安装后磁盘剩余 228 GiB

**③ import 冒烟**：`research/env/logs/p1_import_smoke.log` → 18/18 `OK`，`IMPORT_SMOKE_EXIT=0`

**版本（pip freeze 实测）**

| 包 | 版本 | 包 | 版本 |
|---|---|---|---|
| python | 3.11.15 | sdv | 2.1.0 |
| numpy | 2.4.6 | sdmetrics | 0.33.0 |
| pandas | 2.3.3 | torch | **2.15.0**（sdv→deepecho 自动带入，非显式安装） |
| xarray | 2026.9.0 | netCDF4 | 1.7.5.1.post1 |
| zarr | 3.1.6 | scipy | 1.17.1 |
| dask | 2026.8.0 | scikit-learn | 1.9.1 |
| gcsfs | 2026.10.1 | matplotlib | 3.11.2 |
| fsspec | 2026.9.0 | pytest | 9.2.0 |
| pydantic | 2.14.3 | typer | 0.28.0 |
| PyYAML | 6.0.3 | pip | 26.3 |

**有后果的实测发现**
- `sdv==2.1.0` 拉入 **torch 2.15.0**；synthcity 要求 `torch>=2.1,<2.3`（`third_party/_ref/synthcity/prereq.txt`）→ **主 env 不能用于 synthcity**，必须独立 env。
- `zarr==3.1.6` 下 `zarr.open_consolidated()` 读 zarr-v2 consolidated store 报 `TypeError: object of type 'generator' has no len()`；`xr.open_zarr(mapper, consolidated=True)` 正常（WB2 9.3 s、ARCO 11.3 s）。Phase 1 统一用 `xr.open_zarr`。

## P2. 真实数据（CONDITIONAL_PASS）

### P2-GCS：原方案实测失败（已停止，未硬拉）
- 两个候选 store 匿名 `.zmetadata`（HTTP 200）：
  - WB2 `1959-2022-full_37-6h-0p25deg-chunk-1.zarr-v2`：36 数组，**无 clwc/ciwc/crwc/cswc**
  - ARCO `ar/full_37-1h-0p25deg-chunk-1.zarr-v3`：277 数组，**有 clwc/ciwc（长名 specific_cloud_liquid/ice_water_content）**，无 crwc/cswc
- 致命瓶颈（实测）：pressure-level 数组 chunk = `[1,37,721,1440]`，单 chunk 压缩后 **85,235,158 B（85.2 MB）**；
  本机到 GCS 实测吞吐 **38,317 B/s**（`curl --max-time 120 -r 0-5242879` → `http=206 bytes=4598275 time=120.0s`）。
  ⇒ 单变量单时次 ≈ **37 min**，7 变量×8 时次 ≈ **25 h** → 不可行，已 `pkill` 下载进程。
- 证据：`research/env/logs/p2_arco_infeasible_evidence.md`、`p2_store_metadata_check.log`、
  `p2_wb2_zmetadata.json`、`p2_arco_zmetadata.json`、`p2_arco_probe.log`
- 附带排错（明日可复用）：ARCO chunk key 是 **4 段** `temperature/<int_idx>.0.0.0`（`...0.0` → 404）；索引要用
  `ds.time.sel(...)`/`ds.indexes["time"].get_loc(...)` 取 int，不手算。

### P2-本机：改用本机真实 ERA5 并落地（PASS）
命令：`/opt/miniconda3/envs/atmosgen/bin/python -u research/env/p2b_local_era5.py > research/env/logs/p2b_local_era5.log 2>&1`
日志末尾 `[DONE] ok`；进程退出码 0。

**A. 源文件检查（只读；`/Volumes/Untitled/2010/ERA5_025dg_20100501_*.nc`）**
- 4 个文件各 **251,416,8xx B ≈ 251.4 MB**，`open_dataset` 0.3–0.6 s
- 每个文件：`dims = {time:24, level:37, latitude:321, longitude:441}`，data_vars 只有 1 个（`t`/`q`/`u`/`v`）
- `time`：2010-05-01T00:00 … 2010-05-01T23:00，**hourly，24 步**
- `level`：37 层，1…1000（**hPa，升序**：1,2,3,5,7,…,850,…,1000）
- `latitude`：**-10.0 … 70.0（降序，321 点）**；`longitude`：**50.0 … 160.0（升序，441 点）**
  → ⚠️ **本机文件实际是区域子集（东亚/中国区），不是全球 0.25°**；网格步长 0.25°
- units：t=`K`、q=`kg kg**-1`、u/v=`m s**-1`；long_name：Temperature / Specific humidity / U,V component of wind
- **四个文件时间轴一致性：实测 `True`（np.array_equal 全等）**

**B. 切片（已落盘）**
- 抽取：`time = 2010-05-01 00/06/12/18`（4 个时次）、`level = [500,700,850]` hPa、`latitude 20–45N`、`longitude 100–125E`
- 产物：**`data/raw/era5_local_slice_20100501.nc`，1,002,210 B（1.002 MB，<10 MB）**
- dims：`{time:4, level:3, latitude:101, longitude:101}`；变量名已归一为短名 `t/q/u/v`，带 `source_file` 属性
- 数值范围（子集）：t 247.98–301.15 K；q 6.25e-5–1.579e-2 kg/kg；u −12.97–40.68 m/s；v −24.46–21.15 m/s
- **NaN：4 个变量全部 0/122,412（0.00%）**
- 读取耗时：每个变量 0.1 s；NetCDF 写 0.3 s；重开验证 OK
- 摘要：`data/raw/era5_local_slice_SUMMARY.md`（含来源路径、映射、单位、dims、NaN、大小、耗时、只读声明）

**C. 云水样本（clwc/ciwc，已落盘）**
- 源：`<local>/era5数据dbas/data/downloads/reanalysis-era5-pressure-levels/2024/20240101_multi_fdb02e.nc`（45,855 B，只读，未改动）
- 副本：**`data/raw/era5_cloud_sample_20240101.nc`（45,855 B）**
- dims：`valid_time:1, pressure_level:3, latitude:61, longitude:141`；data_vars = `clwc, ciwc`
- units 均为 `kg kg**-1`；范围：clwc 0–1.662e-4、ciwc 0–1.049e-4；**NaN 0/25,803**
- 摘要：`data/raw/era5_cloud_sample_SUMMARY.md`

## P3. 第三方参考仓库（PASS）

**命令**：`git clone --depth 1 https://github.com/<org>/<repo>.git`（5 个）
- 日志 `research/env/logs/p3_clones.log` → 每个仓库 `CLONE_EXIT_<name>=0`（**5/5 成功**）
- 元数据 `research/env/logs/p3_repo_meta.log`

| 仓库 | HEAD | HEAD 日期 | LICENSE | python 要求 | 关键入口 |
|---|---|---|---|---|---|
| SDV | `549dc94e49e73103487b3dbc324abcc43c8bfcc0` | 2026-11-05 | **BSL 1.1**（Business Source License） | `>=3.10,<3.15` | `sdv/`（synthesizers/、evaluation/） |
| synthcity | `23f322fe381326ed01c41b13d469a06e38cce545` | 2025-06-17 | Apache-2.0 | `>=3.9`（setup.py）；prereq: `torch>=2.1,<2.3`, `numpy>=1.20`, `tsai` | `src/synthcity/plugins/core/plugin.py`；version.py = **0.2.12** |
| SDGym | `009a35b2875ef6743ad3a713ee1a979b6df86e04` | 2026-11-04 | **BSL 1.1** | `>=3.10,<3.15` | `sdgym/`（benchmark.py、cli/、_benchmark/） |
| PDEBench | `4ff3e3a4aa1561721b5571fa3a048a0a463e0568` | 2026-03-30 | MIT（LICENSE.txt） | **`>=3.9,<3.11`** | `pdebench/data_gen/`、`pdebench/models/` |
| RealPDEBench | `06ba38d2db3e78edc2556ff7d9098119a42c2b07` | 2026-11-04 | **CC BY-NC 4.0（禁止商用）** | `>=3.10` | `realpdebench/eval.py`、`realpdebench/utils/metrics.py`（eval_metrics/mse_loss/probe_diagnostic）、`data/`、`model/` |

**许可证结论（影响技术选型）**：SDV/SDGym = **BSL 1.1**（非 OSI 开源）；RealPDEBench = **CC BY-NC 4.0**（禁商用）；synthcity Apache-2.0、PDEBench MIT 可安全引用。

---

# 二、未完成（明确 NOT_RUN，不可当通过）

| 项 | 状态 | 原因 / 下一条命令 |
|---|---|---|
| crwc / cswc 真实数据 | **NOT_RUN** | 本机与 WB2/ARCO 均无；需 CDS。**今晚不跑 CDS**（成本+排队）。明天：确认 `~/.cdsapirc` 后用小请求测配额 |
| 0.25° / 云水多时次大样本 | **NOT_RUN** | 今晚只落 1 个 clwc/ciwc 样本（1 时次×3 层×61×141）；扩样本明天做（2024 目录其他 `*_multi_*.nc`） |
| GCS 0.25° 全变量切片 | **NOT_RUN（已证不可行）** | 38.3 KB/s × 85.2 MB/chunk；证据见 `p2_arco_infeasible_evidence.md`；若明天必须用 GCS，需换 chunk 小的 store 或并发 Range 实测 |
| P4-A synthcity 0.2.12 独立 env | **NOT_RUN（按指令取消）** | 明日：`conda create -y -n atmosgen-synthcity python=3.10`；`pip install "torch==2.2.2" "numpy<2"`；再 `pip install "synthcity==0.2.12"`；预期卡点 `tsai`/`decaf` 与 arm64 torch 2.2.2 |
| P4-B pdebench 安装 | **NOT_RUN（按指令取消）** | 明日：`conda create -y -n atmosgen-pdebench python=3.10`；`pip install pdebench==0.1.0`（py<3.11 满足）失败则 `pip install -e third_party/_ref/PDEBench`；只判可装，不跑训练 |
| P4-C 训练链 | **NOT_RUN（按指令）** | 明日 Phase 1 |

---

# 三、阻塞点与风险（供计划书）

1. **网络是头号风险**：本机→GCS 实测 38.3 KB/s；全量 Zarr chunk 85.2 MB/变量/时次 ⇒ “按需从云端切片”不可行。Phase 1 数据获取必须走「一次性批量下载/本机已有数据/CDS 服务端子集」，不能假设云端按需读。
2. **本机数据的真实覆盖**：`/Volumes/Untitled/2010` 是**区域子集**（lat −10…70、lon 50…160，321×441，0.25°），不是全球；37 层、逐时、日文件 251 MB；2010–2013 均有目录（736 文件/年）。若实验需要全球或其他区域，需另找源（这条要写进计划书的数据假设）。
3. **云水变量分层**：本机有 clwc/ciwc（2024 小样本）；WB2 全无；ARCO 有 clwc/ciwc 无 crwc/cswc；crwc/cswc 只能 CDS → 今晚标 NOT_RUN。
4. **许可证**：SDV/SDGym = BSL 1.1；RealPDEBench = CC BY-NC 4.0 → 商用路径要规避或替换。
5. **Python 版本矩阵**：主 env 3.11（PASS）；PDEBench 需 <3.11；synthcity 需 torch<2.3（主 env 2.15）→ 至少 2 个附加 env。
6. **资源**：磁盘余 228 GiB 充足；8 GB RAM 下「按天文件 → isel 子集」峰值很低（本次脚本内存占用小、切片 0.1 s/变量），完全可行。

---

# 四、交付物与日志索引

**数据产物**
| 文件 | 大小 | 内容 |
|---|---|---|
| `data/raw/era5_local_slice_20100501.nc` | 1,002,210 B | 本机真实 ERA5：t/q/u/v，4 时次×3 层×101×101，NaN 0 |
| `data/raw/era5_local_slice_SUMMARY.md` | 2,594 B | dims/coords/units/范围/NaN/耗时/只读声明 |
| `data/raw/era5_cloud_sample_20240101.nc` | 45,855 B | clwc/ciwc 真实样本（1 时次×3 层×61×141，kg/kg） |
| `data/raw/era5_cloud_sample_SUMMARY.md` | 1,468 B | 变量/单位/dims/数值范围 |

**日志**
| 日志 | 内容 |
|---|---|
| `logs/p1_conda_create.log` | conda create 全程 |
| `logs/p1_pip_install.log` | pip 升级+安装，3 个 EXIT=0 |
| `logs/atmosgen_pip_freeze.txt` | 109 包 freeze |
| `logs/p1_import_smoke.log` | 18/18 import OK |
| `logs/p2_store_metadata_check.log` | WB2/ARCO 变量清单与 chunk 形状 |
| `logs/p2_wb2_zmetadata.json` / `p2_arco_zmetadata.json` | 原始 consolidated metadata |
| `logs/p2_arco_probe.log` | open_zarr 11.3 s、time 轴、索引 |
| `logs/p2_arco_infeasible_evidence.md` | 85.2 MB/chunk + 38.3 KB/s 失败证据 + 明日 recipe |
| `logs/p2b_local_era5.log` | 本机文件检查 + 切片 + 云水样本，`[DONE] ok` |
| `logs/p3_clones.log` | 5 仓库 clone 退出码 |
| `logs/p3_repo_meta.log` | HEAD/LICENSE/python/入口 |

**脚本**：`research/env/p2b_local_era5.py`（本机切片，可重跑）、`research/env/probe_arco_read.py`（GCS 成本探针）。

---

# 五、明日第一小时建议

1. **P4 两 env 探测**（各 5–10 min，不碰训练）：`atmosgen-synthcity`（py3.10 + torch 2.2.2）与 `atmosgen-pdebench`（py3.10）。
2. **数据生产脚本化**：把 `p2b_local_era5.py` 泛化成「日文件目录 → 变量/层/区域/时次切片」，先扫 2010 全年 736 文件的时间轴与缺测，再扩 2011–2013；确认是否有全球版本的需求缺口。
3. **云水扩样**：2024 目录其他 `*_multi_*.nc` 批量小样本；**crwc/cswc** 与 CDS 配额先确认再跑。
4. 若仍要 GCS：优先量化 `ar/1959-2022-6h-1440x721.zarr` 等小 chunk store（HEAD `.zarray` 一条命令），并测 4 路并发 Range 是否线性提速。
