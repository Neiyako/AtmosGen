# Phase 1 报告 · 最小运行骨架（Test A）

- 日期：2026-11-07
- 依据：[`docs/IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §7 Phase 1、§12 开工清单；[`docs/DATA_SCHEMA.md`](DATA_SCHEMA.md)
- 结论：**Phase 1 完成（Test A PASS）**。`atmosgen` 包可安装、配置可校验、真实 ERA5 切片已规范化落地并带完整契约；`generate/evaluate/report` 按计划为显式 `NOT_RUN` 占位（非静默跳过）。

---

## 1. 一句话结果

`atmosgen ingest --config configs/era5_ingest.yaml` 在 **7.08 s / 峰值内存 426.1 MB（后台 50 ms 采样；进程 ru_maxrss 432.0 MB；多次运行 358–426 MB）** 内读取本机 ERA5 归档 16 个 239.8 MB 源文件的小切片，产出 `data/processed/era5_local.nc`（12.97 MB，16 时次 × 3 层 × 101 × 101，t/q/u/v + 派生 tcwv），契约 14 项字段齐全，**源归档 16 个文件 size+mtime 全部未变**，读回校验一致。run 目录含 `config.yaml` + `environment.json` + `test_a_report.json`。

## 2. 环境（实测）

| 项 | 实测值 |
|---|---|
| conda 环境 | `/opt/miniconda3/envs/atmosgen`，Python 3.11.15 |
| 关键依赖 | numpy 2.4.6 / pandas 2.3.3 / xarray 2026.9.0 / netCDF4 1.7.5.1 / zarr 3.1.6 / pydantic 2.14.3 / typer 0.28.0 / sdv 2.1.0 / sdmetrics 0.33.0 / psutil 7.2.2 |
| 安装方式 | `pip install -e . --no-deps`，`atmosgen --version` → `atmosgen 0.1.0` |
| 真实数据卷 | `/Volumes/Untitled/2010/` 可用（2010-05-01…04 每天 t/q/u/v 四个文件齐全） |

## 3. 交付物（Phase 1 计划项 → 实际文件）

| 计划任务 | 产物 | 验收 |
|---|---|---|
| 项目结构 + pyproject + 安装 | `pyproject.toml`、`atmosgen/__init__.py`、`atmosgen/cli.py` | `atmosgen --version` → `atmosgen 0.1.0`；`pip install -e .` 成功 |
| 配置模型 + 校验 | `atmosgen/config.py`（pydantic v2，generation/ingest 两类配置，source 字段按 kind 白名单） | `atmosgen validate-config`：合法 6 例通过、配置负例 9 类 + 数据级负例 4 个被拒（见 §5） |
| 数据契约 | `atmosgen/contract.py`（14 字段 + 内嵌 `atmosgen:contract` + `contract.json` 侧车） | 契约构造/校验/往返单测 9 个 |
| 数据 IO | `atmosgen/data/io.py`（NetCDF4/zarr、原子写、raw 写保护、只读打开） | 往返一致；写 `data/raw` 被 `RawWriteError` 拒绝 |
| 真实数据 ingest | `atmosgen/data/preprocess.py`、`scripts/ingest_local_era5.py` | **Test A PASS**（§4） |
| registry + 生成器基类 | `atmosgen/generators/{base,registry}.py` | 5 个模型注册、`list-models` 输出；基类生命周期单测 |
| 基础测试 | `tests/` 6 个文件 **55 个用例** | `pytest -q` → **55 passed in 3.95 s** |
| 运行时/资源记账 | `atmosgen/runner.py`（run 目录、种子、elapsed/峰值 RSS、环境快照） | ingest run 目录含 `config.yaml` + `environment.json` + `test_a_report.json` |

代码规模：`atmosgen/` 11 个模块 3 525 行（含文档字符串），`tests/` 1 081 行。

## 4. Test A · 真实数据读取（证据）

运行：`atmosgen ingest --config configs/era5_ingest.yaml` → `output/ingest_local_era5_20261107-112600_42/test_a_report.json`

### 4.1 数据内容（读回值，非声明值）

| 变量 | 维度 | 单位 | min | max | mean | NaN |
|---|---|---|---|---|---|---|
| t | (time, level, lat, lon) | K | 247.98 | 305.65 | 277.19 | 0 |
| q | (time, level, lat, lon) | kg kg**-1 | 3.805e-07 | 0.01730 | 0.004151 | 0 |
| u | (time, level, lat, lon) | m s**-1 | -19.54 | 42.38 | 6.450 | 0 |
| v | (time, level, lat, lon) | m s**-1 | -24.46 | 25.76 | 0.599 | 0 |
| tcwv（派生） | (time, lat, lon) | kg m**-2 | 3.979 | 65.62 | 28.08 | 0 |

- 时间：`2010-05-01T00:00` … `2010-05-04T18:00`，16 步，步长 6 h（UTC，naive）
- 层：pressure_level `[500, 700, 850]` hPa（源文件 37 层，单位 `millibars` 原样记录）
- 纬度 101 点 20–45°N，步长 0.25°，**descending**（契约 `ascending: false`）；经度 101 点 100–125°E，步长 0.25°，ascending
- 缺测：所有变量 NaN=0（契约 `missing_policy=preserve`，本次无缺测可保留）

### 4.2 只读断言

- 16 个源文件读取前后 **size + mtime 全部一致**（`readonly.all_unchanged = True`）
- 239.8 MB 源文件未做全文件 SHA-256（>64 MiB 阈值），理由与跳过量写入报告 `sha256_skipped`；`data/processed` 与 `data/raw` 产物均有 SHA-256
- 代码层面：`open_source()` 只以读模式打开（xarray Dataset 自带的 `to_netcdf` 无法移除，写保护因此由 `save_canonical()` 的路径断言承担：默认拒绝写 `data/raw/`，单测 `test_refuses_to_write_under_raw_dir`；仅 ingest 传 `allow_raw=True`）

### 4.3 派生量 tcwv 的独立校核

用**未经过 AtmosGen 代码**的独立脚本，从源归档 `q`（37 层）在同一时次/格点（2010-05-01T00, 30°N, 110°E）重算柱积分：

```
independent full-37-level column integral : 30.106841 kg m-2
processed dataset tcwv (same point/time) : 30.106841 kg m-2
relative difference                      : 2.36e-16
```

契约中 `source.derivation.tcwv` 与 `tcwv.atmosgen_derived` 均标注 derived，报告显著位置注明**不得当作 ERA5 官方 tcwv**。

### 4.4 产物

| 文件 | 大小 | SHA-256 |
|---|---|---|
| `data/raw/era5_local_slice.nc`（仅 t/q/u/v，无派生变量） | 11.96 MB | `b05c720c…` |
| `data/processed/era5_local.nc`（+ tcwv，内嵌契约） | 12.97 MB | `784cd4f3…` |
| `data/processed/era5_local.nc.contract.json`（侧车契约，含最终文件校验和） | 2.9 KB | — |

`checks`：source_files_found / time_axis_consistency / readonly_sources_unchanged / contract_valid / roundtrip_readback / no_missing_values / derived_tcwv_valid 全 **PASS**。

## 5. 测试与验收

- `pytest -q`：**55 passed，3.95 s**（契约 10、IO 9、配置 14、ingest 8、registry 5、CLI 9）
- `atmosgen validate-config` 正例：`demo.yaml`、`era5_ingest.yaml`、`era5_cloud_sample.yaml`、`physics_advection.yaml`、`stat_gaussian_copula.yaml`、`stat_ctgan.yaml` → PASS；负例：`samples.unit=field`+SDV、未知模型、physics 缺 physics 段、CFL 超限、阈值越界、local_era5 缺 root/date_range 等均被拒（退出码 2）
- `atmosgen list-models`：5 个模型注册（3 统计 / 1 物理 / 1 混合），标注阶段与依赖状态
- CLI 退出码：`--version`=0；`validate-config` 合法 0 / 非法 2；`ingest` 成功 0 / 失败 4 / 传错配置类型 2；`generate|evaluate|report`=3 且输出 `NOT_RUN`
- NetCDF 与 **Zarr 往返**均通过（zarr 3.1.6 + xarray 2026.9）
- `validate-config --no-check-data` 与数据级校验分离：数据未 ingest 时只报 warning，输入数据存在时拒绝变量/时间/层/范围不匹配

## 6. 与计划的偏差 / 需要记录的决策

| # | 事项 | 实际做法 | 理由 |
|---|---|---|---|
| P1 | 嵌入式契约的 `source.sha256` | 嵌入式契约**不含**该键；最终文件 SHA-256 写入侧车契约 + `test_a_report.json` | 文件校验和写进文件自身是自引用、无法成立；侧车 `contract.json` 是权威副本 |
| P2 | `data/raw` 写保护 | 默认拒绝；ingest 以 `allow_raw=True` 显式写入 | 满足"raw 只由 ingest 写一次 + 其它代码被拒"两条验收 |
| P3 | 示例配置变量集 | `demo.yaml` / 两个 stat 配置改为 `[t,q,u,v,tcwv]`；`hybrid.yaml` 保留 `clwc/ciwc` 并在数据级校验如实 FAIL | Phase 1 实际数据集无云水；hybrid 需合并云水样本后才能通过数据级校验，不静默放宽 |
| P4 | 源文件 SHA-256 | >64 MiB 只做 size+mtime 指纹 | 4 GB 全量摘要会主导 ingest 时间；跳过量逐文件写入报告，未标 PASS |
| P5 | `samples.unit=field` + physics | 允许 | 物理模式按时间步=场数，语义正确；仅统计表格模型与该组合互斥 |
| P6 | 两次运行的文件 SHA-256 | 不同（数值一致） | 契约 `metadata.created_at`/`elapsed_seconds` 每次不同，嵌入属性随写入变化；可复现性（数据与指标一致性）按 Test E 在 Phase 5 处理，Phase 1 不做 bitwise 承诺 |
| P7 | 峰值内存口径 | 后台线程 50 ms 采样 RSS，另报进程 `ru_maxrss` 交叉核对 | 单次 `rss` 读数是瞬时值不是峰值（独立验收 F2 指出并已修复） |

## 6.5 独立验收（对抗性复核）与修复记录

Phase 1 完成后由独立验收员（只读复核 + 现场复跑，不采信仓库文字）复核，首轮判定 **CONDITIONAL_PASS**，指出 4 个中级缺陷；本轮已全部修复并回归：

| ID | 问题 | 修复 | 证据 |
|---|---|---|---|
| F1 | 状态聚合把 `CONDITIONAL_PASS` 升级为 `PASS` | `_overall_status()` 改为：FAIL 主导，任何非 PASS 项降级为 CONDITIONAL_PASS | 单测 `test_overall_status_never_upgrades_partial_pass` |
| F2 | `peak_rss_mb` 实为瞬时 RSS | `ResourceTracker` 改为后台 50 ms 采样峰值 + 另报 `process_max_rss_mb`（ru_maxrss） | 本轮 ingest：426.1 MB / ru_maxrss 432.0 MB（比值 0.986；首轮瞬时值偏差曾达 20 倍）；多次运行 358–426 MB |
| F3 | `local_netcdf` 静默忽略 `derive.tcwv` | 配置校验直接拒绝该组合（不静默忽略） | `atmosgen validate-config /tmp/f3.yaml` → FAIL，exit 2，单测 `test_local_netcdf_derive_is_rejected_not_ignored` |
| F4 | 未捕获异常导致 CLI exit 1 | `ingest`/`validate-config` 兜底捕获，统一 exit 4 并给错误信息 | 无后缀源文件 → `EXIT=4`，`ValueError: cannot infer format…` |
| F5 | `CONDITIONAL_PASS` 被当失败（exit 4） | CLI 对 CONDITIONAL_PASS 给 exit 0 + 明示"存在 NOT_RUN" | 无 derive 配置 → `status: CONDITIONAL_PASS`、`EXIT=0` |
| F6 | "open_source 无写路径"表述不成立、单测恒真 | 表述修正为"只以读模式打开，写保护由 save_canonical 断言承担"；单测改为校验源文件指纹不变 | `test_open_source_snapshot_does_not_touch_source` |
| F7 | 同目录多数据集侧车 `contract.json` 互相覆盖 | 侧车命名：netcdf `data.nc` → `contract.json`（schema 约定），其它 → `<完整文件名>.contract.json`（含格式后缀） | `test_sidecar_does_not_collide_within_one_directory` |
| F8 | `years` / `vertical` / `read_only_assertion` 被接受但未使用 | `years` 校验归档年份与 date_range 一致性；`vertical` 校验层坐标类型（era5 与 local_netcdf 两条路径）；`read_only_assertion` 关闭时只读检查标 NOT_RUN | 单测 `test_check_vertical_type_rejects_mismatch`、`test_read_only_assertion_can_be_disabled`、`test_local_netcdf_rejects_archive_only_fields` |
| F9 | `_netcdf_encoding` 分块死代码（部分 chunk 被静默忽略） | 未指定维度回退为整维长度并 clamp，chunksizes 始终生效 | 代码路径 |
| F10 | 报告数字/表述与产物不一致（负例数、模块数、测试行数、sha 键表述、峰值内存） | 全部按实测值更新（见 §1/§3/§5/§6、P1/P7） | 本轮复跑数值 |
| F11 | `environment_snapshot()` 未调用，run 目录无环境快照 | `_run_context()` 写 `environment.json` | run 目录实测存在 `environment.json` |

> 独立验收原始证据（verifier 侧）：`/tmp/agver/verdict.yaml`、`/tmp/agver/evidence/*.log`（临时目录，非仓库产物）。

第二轮独立复核（HEAD `bac2eff`）确认 F1–F5、F9、F11 已修复，另指出 4 个轻缺陷，均已处理：

| ID | 第二轮残留问题 | 处理 | 证据 |
|---|---|---|---|
| R1 | 同 stem 不同格式（`a.nc` / `a.zarr`）侧车仍碰撞 | 侧车名改为 `<完整文件名>.contract.json`（`data.nc` 仍用 `contract.json`） | `test_sidecar_does_not_collide_within_one_directory` |
| R2 | `local_netcdf` 路径仍静默忽略 `source.years` / `source.vertical` | 配置校验拒绝 `local_netcdf` 的 `root`/`years`；`local_netcdf` ingest 也执行 `_check_vertical_type` | `test_local_netcdf_rejects_archive_only_fields`、`test_check_vertical_type_rejects_mismatch` |
| R3 | 报告 3 处表述与产物不符 | §3 资源行补 `environment.json`；负例数改为 7；F8 证据改为真实单测名；新增 3 个测试覆盖 | 本轮 `pytest 52 passed` |
| R4 | `data/io.py` docstring 仍称"无写 API" | 改为"以读模式打开；写保护由 save_canonical 路径断言承担" | 代码注释 |

第三轮独立复核（HEAD `d4e173d`）确认 R1–R4 已修复，另指出 4 个轻缺陷/口径问题，均已处理：

| ID | 第三轮残留问题 | 处理 | 证据 |
|---|---|---|---|
| N1 | `local_netcdf` 仍静默接受 `file_pattern`；`local_era5` 仍静默接受 `input` | `source` 字段按 kind 白名单：archive-only 字段（root/years/date_range/file_pattern）只允许 local_era5，`input` 只允许 local_netcdf，`request` 只允许 cds | `test_local_netcdf_rejects_archive_only_fields`、`test_local_era5_rejects_netcdf_only_fields`；手工验证两例 exit 2 |
| N2 | model-level 数据（units=`1`）被契约记为 `pressure_level` | `build_contract(..., level_type=...)` 接收配置 `source.vertical` 并写入契约；local_era5（raw/processed）与 local_netcdf 三条 build_contract 调用均传入 | 单测 `test_level_type_override_is_recorded`、`test_local_netcdf_ingest_records_level_type`；实测 `{'type':'model_level','units':'1'}` |
| N3 | `data.nc` 与 `data.zarr` 侧车都映射 `contract.json` | 侧车约定改为"仅 netcdf 的 `data.nc` 用 `contract.json`"，其余用完整文件名 | `test_sidecar_does_not_collide_within_one_directory` 扩展 |
| N4 | 报告侧车命名表述、负例计数口径不一致 | §6.5 F7 表述改为"完整文件名"；负例计数按实际测试改为 9 类 | 本轮报告数字 |

> | B1 | 第四轮复核发现的 1 行遗漏：`_ingest_local_netcdf` 的 `build_contract()` 未传 `level_type`（同时暴露 `model_level` 维度未重命名） | 补传 `level_type=source.vertical`；`model_level` 维度与 `pressure_level` 一并重命名为 `level`；新增 local_netcdf + model_level 的 ingest 契约断言 | 单测 `test_local_netcdf_ingest_records_level_type`；用第三轮 verifier 的原始复现 `/tmp/agver3/ml/config.yaml` 重跑 → 契约类型 `model_level`、`units=1`、exit 0 |

> 顺带修复：YAML 未加引号的日期（`date_range: [2010-05-01, 2010-05-02]`）此前被解析为 `date` 对象而校验失败；已加 `field_validator(mode="before")` 统一转 ISO 字符串。

## 7. 未完成 / NOT_RUN（如实标注）

- `crwc / cswc`：**NOT_RUN** —— 需 CDS 时间盒下载（计划 G2 补测），本机归档与现有 CDS 样本均无这两个变量
- `data/processed/era5_local.nc` 的云水变量（clwc/ciwc）未合并：云水样本为青藏高原 2024-01-01 单时次，与东亚 2010-05 不同域，Phase 2 起单独小数据集验证，不做静默拼接
- `generate / evaluate / report`：**NOT_RUN**（Phase 2–4 实现），CLI 明确以退出码 3 报告，不产生占位数据
- 全文件 SHA-256（239.8 MB × 16）：**NOT_RUN**（见 P4）
- Phase 2–5 全部未开始

## 8. 下一步（Phase 2 开工清单）

1. `data/tabular.py`：场 ↔ 表格（cells × time 行）双向转换 + 协变量（hour/level/lat_norm/lon_norm），用真实数据集验证可逆性
2. `generators/sdv_adapter.py`：GaussianCopula / CTGAN / TVAE 三模型 fit→sample→save，`--overwrite` 语义与 run 目录接入
3. `validation/statistical.py`（sdmetrics QualityReport + 自研 KS/矩/分位数/相关距离）与 `validation/{temporal,spatial}.py` v1
4. `atmosgen generate` / `evaluate` 端到端；产出 `output/stat_*/data.nc`、`contract.json`、`quality_report.{json,md}`
5. Synthcity 隔离环境时间盒探测（PASS 或 NOT_RUN + 日志证据）
6. E1 模型对比实验（质量分 / 训练时长 / 峰值内存）写入 `docs/EXPERIMENT_REPORT.md` 初稿

## 9. 复现命令

```bash
conda activate atmosgen
cd "Atmospheric Data Generation"
pip install -e . --no-deps

atmosgen --version
atmosgen validate-config configs/demo.yaml
atmosgen validate-config configs/era5_ingest.yaml
atmosgen ingest --config configs/era5_ingest.yaml --overwrite
atmosgen list-models
pytest -q
```

> 说明：`data/processed/*.nc`、`data/raw/*.nc`、`output/` 按 `.gitignore` 不入库；Test A 的完整数值证据在 `output/ingest_local_era5_*/test_a_report.json`（重新运行 ingest 可重建）。
