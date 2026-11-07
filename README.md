# AtmosGen — Atmospheric Synthetic Data Generation Framework

面向大气科学研究的仿真数据生成框架：输入真实大气数据（ERA5），指定生成条件与规模，输出**统计真实、时空连续、物理合理**的仿真数据，并附带四类质量验证报告。

> 本项目不重新实现生成算法，而是整合成熟开源项目（SDV / Synthcity / SDGym / PDEBench / RealPDEBench），AtmosGen 自己只负责：数据契约与 IO、适配层、任务执行、统一保存、质量验证与报告。

## 当前状态（2026-11-07）

| 阶段 | 内容 | 状态 |
|---|---|---|
| **Phase 0** | 五个第三方项目技术审查、复用矩阵、依赖兼容性、实施计划 | ✅ **已完成** |
| **Phase 1** | 最小运行骨架：契约 / IO / 配置 / CLI / 真实数据 ingest | ✅ **已完成**（2026-11-07，Test A PASS，见 [docs/PHASE1_REPORT.md](docs/PHASE1_REPORT.md)） |
| **Phase 2** | 统计生成（SDV）+ 统计/时序/空间评价 | ✅ **已完成**（2026-11-07，Test B + E1，见 [docs/PHASE2_REPORT.md](docs/PHASE2_REPORT.md)） |
| **Phase 3** | 物理仿真（2D advection–diffusion + 解析解验证） | ✅ **已完成**（2026-11-07，6 项解析验证 + Test C，见 [docs/PHASE3_REPORT.md](docs/PHASE3_REPORT.md)） |
| Phase 4 | 混合生成（物理基底 + 条件生成 + 物理筛检） | ⏳ 待实施 |
| Phase 5 | Benchmark、四类质量报告、可复现性与交付 | ⏳ 待实施 |

**Phase 1–3 已交付可运行链路**：`validate-config` / `ingest` / `list-models` / `generate` / `evaluate` / `report` 全部可用。statistics 三模型（level_wise）+ physics 2D advection–diffusion（6 项解析验证 + Test C）均已端到端产出带契约的数据与四类质量报告（statistical/temporal/spatial/physical）。实测：统计边缘分布 7/8 PASS、时空 FAIL 属预设；物理解 mass 漂移 2.9e-16、收敛阶≈2；**统计 run 饱和违例 3.3%**（Phase 4 筛检动机）。`hybrid` 仍报 `NOT_RUN`（exit 3）。验收以 `docs/PHASE{1,2,3}_REPORT.md` 数值为准。

## 先读这些文档

1. [docs/THIRD_PARTY_AUDIT.md](docs/THIRD_PARTY_AUDIT.md) — 五个项目到底能不能用、许可证、API、数据源实测
2. [docs/REUSE_MATRIX.md](docs/REUSE_MATRIX.md) — 每项能力复用谁、什么级别、不做什么
3. [docs/DEPENDENCY_COMPATIBILITY.md](docs/DEPENDENCY_COMPATIBILITY.md) — 本机 8 GB / arm64 下的真实环境结论
4. [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) — **实施计划（Phase 1–5 的任务、验收、测试 A–F）**
5. [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md) — 统一数据契约与 14 个必需元数据字段
6. [docs/PHASE1_REPORT.md](docs/PHASE1_REPORT.md) — **Phase 1 实际结果与 Test A 数值证据（含未完成项）**
7. [docs/PHASE2_REPORT.md](docs/PHASE2_REPORT.md) — **Phase 2 实际结果（Test B/E1/偏差/NOT_RUN）**
8. [docs/PHASE3_REPORT.md](docs/PHASE3_REPORT.md) — **Phase 3 实际结果（解析验证/Test C/物理验证）**
9. [docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md) — E1/E2 对比与 Synthcity 时间盒

Phase 0 原始证据在 `research/`（环境探测日志、外部资料核对、五个仓库的只读浅克隆）。

## 技术路线（一句话）

SDV 做统计生成，Synthcity 隔离环境做深度时序补充，sdmetrics（MIT）做统计/时序评价，最小自研 2D advection–diffusion 做物理仿真（PDEBench 只作参考），本机 ERA5 归档做真实数据，全部产出统一 NetCDF 契约。

## 环境

主环境（已在本机创建并实测，Python 3.11）：

```bash
conda create -y -n atmosgen python=3.11
conda activate atmosgen
pip install numpy pandas xarray netCDF4 zarr dask gcsfs pyyaml pydantic typer \
            scipy scikit-learn matplotlib pytest sdv sdmetrics
pip install -e . --no-deps      # 安装 atmosgen 包与 atmosgen 命令
pytest -q                        # 129 个用例（Phase 1–3）
```

版本锁定见 `research/env/logs/atmosgen_pip_freeze.txt`。Synthcity 独立环境已实测（Python 3.10 + torch 2.2.x + numpy<2，需 pin `scipy==1.11.4`、`opacus==1.4.1`），结论见 [docs/DEPENDENCY_COMPATIBILITY.md](docs/DEPENDENCY_COMPATIBILITY.md) §10。

## CLI 状态

```bash
atmosgen --version                                        # ✅ Phase 1
atmosgen validate-config configs/demo.yaml                # ✅ Phase 1（结构 + 数据级校验）
atmosgen ingest     --config configs/era5_ingest.yaml     # ✅ Phase 1（Test A）
atmosgen list-models                                      # ✅ Phase 1
atmosgen generate   --config configs/stat_gaussian_copula.yaml   # ✅ Phase 2（statistical）
atmosgen generate   --config configs/physics_advection.yaml     # ✅ Phase 3（physics）
atmosgen evaluate   --input output/<run>/data.nc --real data/processed/era5_local.nc --fail-on never   # ✅ Phase 2
atmosgen report     --input output/<run>/quality_report.json --format md               # ✅ Phase 2
```

退出码约定：0 成功；2 配置/用法错误；3 当前阶段未实现（NOT_RUN）；4 运行期失败或 `evaluate` 质量 FAIL（默认 `--fail-on fail`；要 CI 放行用 `--fail-on never`）。

## 真实数据

- **主数据源（本机）**：`/Volumes/Untitled/{2010..2013}/ERA5_025dg_YYYYMMDD_<var>.nc` — ERA5 0.25°、37 气压层、逐时、0.25° 亚洲区域（lat −10…70°N，lon 50…160°E），每年 5–10 月 184 天，变量 `t/q/u/v`。**只读使用**，只把小切片写入 `data/raw/`。
- **云水真实样本**：`era5数据dbas/data/downloads/.../20240101_multi_fdb02e.nc`（CDS 下载，含 `clwc/ciwc`，青藏高原 3 层 1 时次）。
- **待补**：`crwc/cswc` 与更大范围云水 → Phase 1 用 CDS API 时间盒获取，失败标 NOT_RUN。
- **tcwv**：由 q 垂直积分派生并标注 derived（或 CDS 获取真实值）。
- 远程 ARCO-ERA5 / WeatherBench2 仅作备选：实测 GCS 下载约 83 KB/s，整片传输不可行（详见 `docs/THIRD_PARTY_AUDIT.md §7`）。
- 原始数据只读，生成过程不得回写；输出与原始数据分目录存储。

## 许可证

- SDV / SDGym / Copulas：**BSL 1.1**（研究/非生产可用；禁止商业 Synthetic Data Service；4 年后转 MIT）
- sdmetrics：MIT
- Synthcity：Apache-2.0
- PDEBench：MIT（只参考，不引入依赖）
- RealPDEBench：**CC BY-NC 4.0**（只参考方法，不复制代码/数据，非商用）
- ERA5 数据（本机归档 / ARCO-ERA5）：Copernicus 产品许可（署名、无担保）
- 本项目**不 vendor 任何第三方源码**。

## 验收原则

不以目录数量、代码行数、接口数量、测试数量作为成功标准：必须**实际生成数据、读取数据、验证数据**，用 Test A–F（见实施计划 §8）的数值证据验收。未完成/未运行的项在报告中如实标注（NOT_RUN ≠ PASS）。
