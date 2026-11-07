# AtmosGen 架构说明（ARCHITECTURE）

> 结论先行：AtmosGen 只做**数据契约、IO、适配层、任务编排、统一保存与质量验证**五件事，不自研生成算法——
> 统计生成用 SDV，物理仿真用自研最小求解器（不 vendor 任何第三方源码），混合模式用「物理基底 → 条件生成 → 硬约束筛检」。
> 所有产出（真实切片 / 统计生成 / 物理仿真 / 混合结果）都携带同一份数据契约，可被任意工具直接读取。

---

## 1. 分层结构

```
┌──────────────────────────────────────────────────────────────────────────┐
│ CLI 层        atmosgen/cli.py                                            │
│               validate-config · ingest · generate · evaluate · report     │
│               · list-models        退出码 0/2/3/4（见 GENERATION_GUIDE）  │
├──────────────────────────────────────────────────────────────────────────┤
│ 编排层       config.py（pydantic v2 模型 + 跨字段校验）                  │
│               runner.py（run 目录 / 种子 / 耗时 / 峰值 RSS / 环境快照）  │
│               generation.py（run_generation：三种模式的统一编排）        │
│               evaluation.py（对齐 + 四类验证 + 报告落盘）                 │
│               hybrid.py（Mode 3 筛检与条件生成）                          │
├──────────────────────────────────────────────────────────────────────────┤
│ 数据层       data/io.py（NetCDF/Zarr 原子写、只读打开、raw 写保护）      │
│               data/tabular.py（场 ↔ 表格双向转换）                      │
│               data/selection.py（时间/层/空间子集选择）                  │
│               data/preprocess.py（真实数据 ingest：派生、契约、Test A）   │
├──────────────────────────────────────────────────────────────────────────┤
│ 生成层       generators/registry.py（模型注册表 + 惰性 factory）          │
│               generators/base.py（fit / sample / save 统一接口）           │
│               generators/sdv_adapter.py（gaussian_copula / ctgan / tvae） │
│               generators/hybrid_adapter.py（Mode 3 适配器）               │
│               generators/physics/advection_diffusion.py（自研求解器）     │
│               generators/physics/conditions.py（IC / BC 构造）           │
├──────────────────────────────────────────────────────────────────────────┤
│ 验证层       validation/statistical.py · temporal.py · spatial.py        │
│               validation/physical.py（硬约束 + 软指标）                   │
│               validation/report.py（状态聚合 + Markdown 渲染）           │
└──────────────────────────────────────────────────────────────────────────┘
```

各层职责一句话：CLI 只做参数解析与退出码；编排层只做"读配置→跑模型→落盘→记账"；数据层只关心**字节级**与**契约**；
生成层只关心"造数据"；验证层只关心"给数字和状态"，不做任何修图或美化。

## 2. 三条数据流

### 2.1 Mode 1 · 统计生成（`generation.py:154` `_run_statistical`）

| 步骤 | 代码位置 | 说明 |
|---|---|---|
| ① 选子集 | `data/selection.py:select_subset` | 按 `data.time_range / levels / spatial_extent / variables` 裁剪；越界直接报错 |
| ② 抽稀成表 | `data/tabular.py:130` `dataset_to_table` | `cell_time` 表：每 (time, level, lat, lon) 一行；`subsample` 决定网格步长 |
| ③ 拟合 | `generators/sdv_adapter.py:196` `fit` | **level_wise**：每个气压层一个 SDV 模型（见 §4.1） |
| ④ 采样 | `generators/sdv_adapter.py:288` `sample` | 每层抽够"填满网格"的行，按栅格序 (time, level, lat, lon) 回填 |
| ⑤ 重建场 | `data/tabular.py:249` `rows_to_fields` | 4D 变量逐值回填；3D `tcwv` 按层平均还原 |
| ⑥ 落盘 | `data/io.py:217` `save_canonical` | 原子写 NetCDF/Zarr + 内嵌契约 + `contract.json` 侧车 |

产物：`output/statistical_<model>_<ts>_<seed>/{data.nc, contract.json, metadata.json, environment.json, config.yaml}`。

### 2.2 Mode 2 · 物理仿真（`generation.py:241` `_run_physics`）

| 步骤 | 代码位置 | 说明 |
|---|---|---|
| ① 建网格 | `generators/physics/conditions.py:48` `build_geometry` | `gaussian_plume` 用中心+物理间距；`from_real` 用真实场经纬范围 |
| ② 造初值 | `conditions.py:148` `build_initial_condition` | 高斯羽流解析式，或真实场插值（线性，边缘最近邻兜底） |
| ③ 稳定性闸门 | `advection_diffusion.py:76` `check_stability` | 三条判据全过才允许推进（见 §4.3） |
| ④ 推进 | `advection_diffusion.py:158` `solve_advection_diffusion` | 显式 Euler + 一阶迎风 + 5 点中心扩散；等间隔快照 |
| ⑤ 记账 | `metadata.json.diagnostics` | CFL、扩散数、逐快照质量序列与漂移 |

产物：`output/physics_advection_diffusion_<ts>_<seed>/`，变量 `c`（3D 追迹浓度，schema 1.1 新增）。

### 2.3 Mode 3 · 混合生成（`generation.py:323` `_run_hybrid` → `hybrid.py`）

| 步骤 | 代码位置 | 说明 |
|---|---|---|
| ① 物理基底 | `generators/physics/advection_diffusion.py` | 用求解器把真实 `tcwv` 平流/扩散成带结构的基底 |
| ② 条件生成 | `hybrid.py:197` `conditional_generate` | `sample_remaining_columns(known=tcwv)` 生成其余变量；**断言 known 列逐值原样返回且行序不变** |
| ③ 硬约束筛检 | `hybrid.py:92` `screen_samples` | 有限/非负/范围/饱和逐行判定（掩码语义见 §4.4） |
| ④ 记录拒绝 | `hybrid.py:172` `write_rejected_jsonl` | 每个被拒样本写 `rejected.jsonl`（原因+指标），拒绝率入契约 `quality_metrics` |

> 当前状态（Phase 5）：Mode 3 已端到端跑通（`output/hybrid_hybrid_*/` 含 `data.nc` / `contract.json` / `metadata.json` / `rejected.jsonl` / `quality_report.{json,md}`）。
> 真实数据实测：32 448 个样本中 **653 个超饱和被拒绝（2.01%）**，写入 `rejected.jsonl`；拒绝单元以 NaN 遮罩并在契约 `quality_metrics.screening.mask_applied` 声明；
> 筛检通过单元重新校验 **0 违例**。E3 四组对比见 `docs/EXPERIMENT_REPORT.md`。

## 3. 数据契约机制

- 契约字段（14 项 + 2 项记账）：`contract.py:38` `REQUIRED_FIELDS` —
  `variables / coordinates / units / source / generation_method / model_name / model_version / random_seed /
  initial_conditions / boundary_conditions / physical_parameters / generation_config / quality_metrics / metadata`。
- 存储方式：NetCDF 全局属性 `atmosgen:contract`（JSON 字符串）+ 同目录 `contract.json` 侧车
  （`data/io.py:302` `sidecar_contract_path`：只有 netcdf 的 `data.nc` 用 `contract.json`，其它用 `<完整文件名>.contract.json`，避免同目录互相覆盖）。
- 校验：`contract.py:363` 逐项检查缺字段、schema 大版本、变量维度、单位、坐标类型；`read_contract()` 读不到或解析失败**直接报错**，不做"尽力而为"。
- schema 演进：`1.0`（Phase 1/2，气象变量 9 个）→ `1.1`（Phase 3，新增物理追迹变量 `c`）。
  读者只校验大版本，因此旧数据继续可读；新增为可选项属 minor 变更（`docs/DATA_SCHEMA.md` §2.1 已补记）。
- 原始数据只读：`data/io.py:93` `assert_writable_target` 拒绝写 `data/raw/`（除非 ingest 显式 `allow_raw=True`）。

## 4. 关键设计决策与代价

| # | 决策 | 依据（文件/行） | 代价与证据 |
|---|---|---|---|
| 4.1 | **level_wise 训练**：每个气压层单独训练一个 SDV 模型，而不是"一张表 + level 作协变量" | `generators/sdv_adapter.py:47` `TRAINING_SCHEME = "level_wise"`；模块 docstring 记录了实测 | 连续协变量条件采样在经验分布边界退化：`level=500` 条件下 t 被压到 **248.0 K ± 0.02 K**（真实 262.3 ± 5.0），32 448 行耗时 **148 s**；CTGAN/TVAE 的条件采样是拒绝采样（约 1 接受/100 次、约 6 s/行）。代价：层间相关不再建模，`corr_frobenius` 三模型均 FAIL（1.11–1.30），见 PHASE2_REPORT §3 |
| 4.2 | **等间隔快照调度**：`fields < steps+1` 时用 `step_every = steps // fields` 的均匀序列，而不是 `linspace` 舍入 | `advection_diffusion.py:230-237` | 舍入会产生 2400/2400/3000… s 的不等步长，任何"均匀时钟"检查（方程残差）都会失效。代价：末段内部步未持久化（24 场/96 步 → 末快照停在 step 92），diagnostics 有 warning |
| 4.3 | **二维稳定性三判据**：单轴 `C+2r ≤ 1`、扩散和 `r_x+r_y ≤ 0.5`、**联合 `(Cx+Cy)+2(r_x+r_y) ≤ 1`** | `advection_diffusion.py:76` `check_stability`；`config.py` 的 `PhysicsConfig` 同步 | 只查双轴各自 CFL ≤ 1 会漏判：dx=dy=42 km、dt=5250 s、u0=v0=8 时两轴各 1.0 全部放行，300 步 \|c\| 爆到 5.5e124（Phase 3 独立复核 F1）。代价：可用参数区间被保守收紧 |
| 4.4 | **筛检掩码语义**：逐列逐行判定，非有限值算违规（`non_finite:<var>`），不因缺失而跳过 | `hybrid.py:92` `screen_samples` | 被拒样本必须写 `rejected.jsonl`，不得静默丢弃；修正前后指标分列，禁止把"修正"当"物理保证"（plan §4.3） |
| 4.5 | **时间轴/行序不变式**：条件生成必须逐值原样返回 known 列且行序不变 | `hybrid.py:197` `conditional_generate` 内断言（偏差 > 0 即抛 `HybridError`） | 物理基底不得被统计步改写；否则"物理基底 + 条件生成"名存实亡 |
| 4.6 | **统计聚合状态**：`FAIL > NOT_RUN > CONDITIONAL_PASS > PASS`，`INFO` 不参与判定，空输入 → `NOT_RUN`，未知状态按 FAIL | `validation/report.py:23` `aggregate_status` | 避免"没跑也 PASS"；代价是统计模式整体常为 FAIL（见 VALIDATION_GUIDE §6） |
| 4.7 | **物理模块适用性**："变量缺失/不适用"类 NOT_RUN 不参与聚合但显著列出；核心变量（t/q/u/v/c）全缺失 → 整体 NOT_RUN | `validation/physical.py:1045-1066` | 让 tcwv-only 数据集不会被判成 PASS；代价是聚合规则比其它模块多一层 |

## 5. 扩展点

| 扩展类型 | 步骤 | 关键文件 |
|---|---|---|
| 新增统计模型 | ① 在 `generators/` 写一个 `BaseGenerator` 子类（实现 `fit`/`sample`）② 在 `registry.py:97` `MODELS` 注册 `ModelSpec`（含 `mode`、`phase`、`dependency_problems`、惰性 `factory`）③ 在 `config.py` 的 `MODELS_BY_MODE` 加名字 | `generators/registry.py`、`atmosgen/generators/sdv_adapter.py`（三个 SDV 模型即此模式的样板） |
| 新增验证模块 | ① 实现 `run_xxx_validation(real, synthetic, *, thresholds) -> {"metrics": {...}, "status": ..., "notes": [...]}` ② 在 `evaluation.py:80` 里挂到 `modules` ③ 在 `validation/report.py` 的小节标题表加名字 | `validation/statistical.py`（结构样板）、`validation/report.py:72` |
| 新增物理方程 | ① 在 `generators/physics/` 新写求解器模块（输入 grid/time/parameters/IC/BC，输出 `(time, latitude, longitude)` 场 + diagnostics）② IC/BC 复用 `conditions.py` ③ 注册 `ModelSpec(mode="physics")` | `generators/physics/advection_diffusion.py`、`generators/physics/conditions.py` |
| 新增配置字段 | 在 `config.py` 的对应 pydantic 模型加字段（`extra="forbid"`，拼写错会直接报错） | `atmosgen/config.py` |

## 6. 资源与性能实测

| 运行 | 拟合耗时 | 采样/求解耗时 | 峰值 RSS | 规模 |
|---|---|---|---|---|
| ingest（本机 ERA5 归档 16 文件） | — | **7.08 s** | 426 MB | 16 时次 × 3 层 × 101×101 |
| statistical / gaussian_copula | **1.95 s** | **0.235 s** | 441 MB | 32 448 行 → 26×26×16 场 |
| statistical / ctgan | 136.95 s | 0.484 s | 453 MB | 同上（epochs=100） |
| statistical / tvae | 63.08 s | 0.269 s | 567 MB | 同上（epochs=100） |
| physics / advection_diffusion | — | **0.318 s** | 283 MB | 24 场 × 64×64 |

数值来源：`output/*/metadata.json`（`fit_resources`/`resources`）与 `docs/PHASE{1,2,3}_REPORT.md`。
机器为 macOS arm64 / 8 GB，峰值 RSS 由 50 ms 后台采样得到，另以进程 `ru_maxrss` 交叉核对。

## 7. 已知限制与后续

- **Mode 3 的条件化代价**：混合模式筛检后物理违例为 0、时空指标相对统计有可测改善（如 variogram 16.6→9.7），
  但边缘统计变差（KS 0.087→0.250、column_shapes 0.953→0.882）：条件化改变剩余变量边缘分布，且物理基底的 tcwv 分布与真实不同。
  这是真实权衡（见 `PHASE4_REPORT` §4、`EXPERIMENT_REPORT` E3），未做 quantile mapping 等后处理。
- **层间/时空结构不建模**（level_wise 的直接后果）：统计模式的 temporal/spatial 指标按预设 FAIL；
  物理基底+条件生成只能部分改善（见上一条），不是"时空连续"的完整解。
- **`hybrid.screening.reject_ratio_limit` 未强制**：拒绝率超限只记录不报错（`PHASE4_REPORT` §6）。
- **时间轴索引映射**：混合输出沿用真实前 N 个时次标签，物理演化步长独立，两者无物理时钟对应（契约 notes 注明）。
- **Equation residual 未归一化**（单位 c·s⁻¹），且要求均匀时间步；阈值须在 Phase 4 定标后写死。
- **Mass budget 未做面积/气压加权**：目前是全场求和的相对漂移诊断。
- 求解器为 2D 单方程、标量常量 κ、单层语义；多层与变量 κ 尚未支持。
