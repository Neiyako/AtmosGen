# 生成指南（GENERATION_GUIDE）

> 结论先行：一条命令链即可从本机 ERA5 归档走到"带契约的合成数据 + 四类质量报告"：
> `ingest → generate → evaluate → report`。配置是唯一的入口，所有阈值、网格、样本数都在 YAML 里，代码不猜参数。

---

## 1. 环境与安装

```bash
# 1) 主环境（已在本机实测，Python 3.11）
conda create -y -n atmosgen python=3.11
conda activate atmosgen
pip install numpy pandas xarray netCDF4 zarr dask gcsfs pyyaml pydantic typer \
            scipy scikit-learn matplotlib pytest sdv sdmetrics psutil
pip install -e . --no-deps        # 安装 atmosgen 包与 atmosgen 命令
atmosgen --version               # -> atmosgen 0.1.0
pytest -q                        # 137 个用例（Phase 1–5）
```

| 项 | 说明 |
|---|---|
| 可选依赖组 | `pip install -e '.[stat]'` 装 SDV/sdmetrics（Phase 2 统计生成需要）；`.[dev]` 装 pytest |
| Synthcity 隔离环境 | 可选，仅用于对比实验：`conda create -y -n atmosgen-synthcity python=3.10`，之后必须 `pip install scipy==1.11.4 opacus==1.4.1`（新版本在本机 dlopen 失败 / 需要 torch≥2.6），细节见 `docs/DEPENDENCY_COMPATIBILITY.md` §10 |
| 机器约束 | 8 GB 内存、无独立 GPU；所有实验规模按"≤30 min / <1 GB 输出"设计（`docs/IMPLEMENTATION_PLAN.md` §10） |

## 2. 配置文件逐段说明

配置分两类：生成配置（`project/data/generation/…`）与 ingest 配置（`project/source/output`）。字段均为严格模式
（`extra="forbid"`），**拼错的键会直接报错**。

### 2.1 `project`（两类配置共有）

| 字段 | 类型 | 默认 | 约束 |
|---|---|---|---|
| `name` | str | 必填 | 非空；出现在 run 目录名里 |
| `seed` | int | 42 | 随机种子（`runner.set_global_seed` 覆盖 `random`/`numpy`/`torch`） |
| `output_dir` | Path | None | run 目录根；缺省用 `output.directory`，再缺省 `./output` |

### 2.2 `data`（生成配置；`mode=statistical`/`hybrid` 必填）

| 字段 | 类型 | 默认 | 约束 |
|---|---|---|---|
| `input` | Path | 必填 | 规范数据集（`data/processed/*.nc` 或 zarr） |
| `format` | auto\|netcdf\|zarr | auto | 按扩展名推断 |
| `variables` | list[str] | 必填 | 必须是规范变量名（`t,q,u,v,clwc,ciwc,crwc,cswc,tcwv,c`）；缺失时校验器直接报错 |
| `time_range` | [起, 止] | None | 超出数据时间轴 → 报错 |
| `levels` | list[float] | None | hPa；不在数据层坐标中 → 报错 |
| `spatial_extent` | {lat:[lo,hi], lon:[lo,hi]} | None | 超出数据经纬范围 → 报错 |
| `chunks` | {time/level/latitude/longitude: int} | None | NetCDF 写盘分块 |
| `source_variable_map` | {契约名: 源变量名} | None | 覆盖默认映射 |
| `subsample` | {lat_step, lon_step} | None | 单位**度**；1.0 表示 1° 网格（0.25° 原始网格上抽 4 倍） |

### 2.3 `generation`

| 字段 | 类型 | 默认 | 约束 |
|---|---|---|---|
| `mode` | statistical\|physics\|hybrid | 必填 | 与 `model` 所属模式必须一致 |
| `model` | str | 必填 | 必须已注册（`atmosgen list-models`）；`gaussian_copula`/`ctgan`/`tvae`/`advection_diffusion`/`hybrid` |
| `samples.unit` | cell_time\|field | cell_time | 表格模型（SDV）只能用 `cell_time`；`field` 仅用于物理/混合 |
| `samples.rows` | int | None | 统计模式**下限**：实际生成 `max(rows, 网格行数)`，契约同时记录 `rows_requested/rows_generated` |
| `samples.fields` | int | None | 物理/混合模式的场数（= 快照数） |
| `training` | {epochs, batch_size, discriminator_steps} | 100/500/1 | 仅 SDV 深度模型使用；CTGAN 需 `batch_size % 10 == 0`（不满足时自动取整并记入 metadata） |
| `covariates` | list[str] | [] | 可选协变量（`hour_sin/hour_cos/level/lat_norm/lon_norm`）；**Phase 2 的 level_wise 方案下只作记录**，见 `docs/ARCHITECTURE.md` §4.1 |

### 2.4 `physics`（`mode=physics`/`hybrid` 必填）

| 字段 | 类型 | 约束 |
|---|---|---|
| `equation` | advection_diffusion | 目前唯一支持的方程 |
| `grid` | {nx≥8, ny≥8, dx_m>0, dy_m>0} | nx/ny 为格点数，dx/dy 为物理间距（米） |
| `time` | {dt_s>0, steps≥1} | 推进步长与总步数 |
| `parameters` | {u0, v0, kappa≥0} | 常量风与扩散系数（m/s、m²/s） |
| `initial_condition.kind` | gaussian_plume\|from_real | 羽流需 `center:[lat,lon]`+`sigma_deg`；`from_real` 需 `variable` 且 `data` 段存在 |
| `boundary_condition.kind` | periodic\|open\|fixed_from_real | `periodic` 才允许质量守恒统计 |

> **稳定性在配置层就会被拒**：三条判据（单轴 `C+2r ≤ 1`、扩散和 `≤0.5`、联合 `(Cx+Cy)+2(Rx+Ry) ≤ 1`）任一不满足即报错。

### 2.5 `hybrid`（`mode=hybrid` 必填）

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `physics_component.enabled` | bool | true | 是否用物理基底 |
| `physics_component.output_variable` | str | tcwv | 作为条件生成 known 列的变量 |
| `statistical_component.model` | str | gaussian_copula | 条件生成用的统计模型 |
| `statistical_component.trained_on` | list[str] | [real] | 训练数据来源（real / physics） |
| `screening.hard_constraints` | bool | true | 是否执行硬约束筛检 |
| `screening.reject_ratio_limit` | float 0–1 | 0.2 | 拒绝率上限（超限在报告中显式告警，不静默放宽） |
| `screening.soft_metrics` | list[str] | [] | 额外软指标（如 `mass_budget`） |

### 2.6 `validation` / `output`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `validation.statistical/temporal/spatial/physical` | bool | true | 报告里是否包含该类 |
| `validation.thresholds_overrides` | {键: float} | {} | 覆盖默认阈值（键名见 `docs/VALIDATION_GUIDE.md` §2；未知键只警告） |
| `output.format` | netcdf\|zarr | netcdf | 落盘格式 |
| `output.directory` | Path | None | 输出根目录 |
| `output.compression` | {zlib, complevel 0–9} | true/4 | NetCDF 压缩 |

### 2.7 ingest 配置（`source` / `output`）

| `source` 字段 | 说明 |
|---|---|
| `kind` | `local_era5`（本机按年/日归档）/ `local_netcdf`（单文件样本）/ `cds`（预留） |
| `root` / `input` / `years` / `date_range` / `file_pattern` | `local_era5` 用；`file_pattern` 支持 `{date:%Y%m%d}` 与 `{variable}` 占位 |
| `variable_map` | 契约名 → 文件内变量名（如 `t: temperature`） |
| `subset` | {levels, latitude, longitude, time_step_hours}；`time_step_hours` 必须整除 24 |
| `derive.tcwv` | `{method: vertical_integral_q}`：由 q 垂直积分派生 tcwv（标记 derived，不得当作 ERA5 官方 tcwv） |
| `output.raw_path` / `processed_path` / `read_only_assertion` | 原始切片与规范数据路径；`read_only_assertion` 关闭时只读检查记 NOT_RUN |

字段白名单按 kind 严格区分：archive 专属字段（`root/years/date_range/file_pattern`）只允许 `local_era5`，`input` 只允许 `local_netcdf`，`request` 只允许 `cds`——写错会被拒绝而不是被忽略。

## 3. CLI 用法与退出码

```bash
atmosgen validate-config configs/demo.yaml [--check-data|--no-check-data]   # 位置参数=配置文件
atmosgen ingest      --config configs/era5_ingest.yaml [--overwrite]
atmosgen list-models
atmosgen generate    --config configs/stat_gaussian_copula.yaml [--overwrite]
atmosgen evaluate    --input output/<run>/data.nc [--real data/processed/era5_local.nc] \
                     [--out <dir>] [--config <cfg>] [--overwrite] [--fail-on fail|never]
atmosgen report      --input output/<run>/quality_report.json [--format md|json] [--out <path>]
```

| 退出码 | 含义 | 典型触发 |
|---|---|---|
| **0** | 成功 | 校验通过 / 生成完成 / 报告已写出 |
| **2** | 配置或用法错误 | 字段非法、模型名不存在、CFL 超限、`--input` 指向不存在的文件 |
| **3** | 当前阶段未实现（NOT_RUN） | 请求尚未落地的模式（例如 Phase 3 之前的 `mode=physics`） |
| **4** | 运行期失败 | 源数据缺失、契约校验失败、**或 `evaluate` 质量状态为 FAIL** |

> `evaluate` 默认 `--fail-on fail`：质量 FAIL 时退出 4（CI 可据此判定失败）；
> 需要"只出报告不管状态"时加 `--fail-on never`（退出 0，状态仍写在 stdout 与报告里）。

## 4. 三个模式的最小可复制示例

### 4.1 Mode 1 · 统计生成

配置片段（与 `configs/stat_gaussian_copula.yaml` 一致）：

```yaml
project: {name: stat_gaussian_copula, seed: 42, output_dir: ./output}
data:
  input: ./data/processed/era5_local.nc
  variables: [t, q, u, v, tcwv]
  levels: [500, 700, 850]
  chunks: {time: 6, latitude: 50, longitude: 50}
  subsample: {lat_step: 1.0, lon_step: 1.0}
generation:
  mode: statistical
  model: gaussian_copula
  samples: {unit: cell_time, rows: 20000}
  covariates: [hour_sin, hour_cos, level, lat_norm, lon_norm]
validation: {statistical: true, temporal: true, spatial: true, physical: true}
output: {format: netcdf, directory: ./output}
```

```bash
atmosgen validate-config configs/stat_gaussian_copula.yaml
atmosgen generate    --config configs/stat_gaussian_copula.yaml
atmosgen evaluate    --input output/statistical_gaussian_copula_*/data.nc \
                      --real data/processed/era5_local.nc --overwrite
```

预期产物：`output/statistical_gaussian_copula_<ts>_42/` 内 `data.nc`（16 时次 × 3 层 × 26×26 × 5 变量，32 448 行）、
`contract.json`、`metadata.json`、`environment.json`、`config.yaml`、`quality_report.{json,md}` 与 `figures/*.png`；
实测拟合 1.95 s、采样 0.235 s、峰值 RSS 441 MB（`docs/PHASE2_REPORT.md` §5）。

### 4.2 Mode 2 · 物理仿真

配置片段（与 `configs/physics_advection.yaml` 一致）：

```yaml
project: {name: physics_advection, seed: 42, output_dir: ./output}
generation:
  mode: physics
  model: advection_diffusion
  samples: {unit: field, fields: 24}
physics:
  equation: advection_diffusion
  grid: {nx: 64, ny: 64, dx_m: 42000, dy_m: 42000}
  time: {dt_s: 600, steps: 96}
  parameters: {u0: 8.0, v0: 2.0, kappa: 1.0e5}
  initial_condition: {kind: gaussian_plume, center: [32.5, 112.5], sigma_deg: 1.5, amplitude: 1.0}
  boundary_condition: {kind: periodic}
validation: {physical: true, temporal: true, spatial: true}
output: {format: netcdf, directory: ./output}
```

```bash
atmosgen generate --config configs/physics_advection.yaml
atmosgen evaluate --input output/physics_advection_diffusion_*/data.nc --overwrite --fail-on never
```

预期产物：`data.nc`（变量 `c`，24 个快照 × 64×64）、契约中 `initial_conditions/boundary_conditions/physical_parameters` 齐备；
stdout 打印 `CFL_x=0.114 CFL_y=0.0286 扩散数=0.068 质量漂移=2.862e-16`；评价结果 `physical=PASS`
（`equation_residual` 为 INFO，RMS≈5.45e-07）。解析验证请另跑 `python scripts/validate_solver_analytic.py`。

### 4.3 Mode 3 · 混合生成（**端到端待 Phase 4 收尾**）

配置片段（与 `configs/hybrid.yaml` 一致）：

```yaml
project: {name: hybrid_v1, seed: 42, output_dir: ./output}
data:
  input: ./data/processed/era5_local.nc
  variables: [t, q, u, v, tcwv]
  levels: [500, 700, 850]
  spatial_extent: {lat: [20, 45], lon: [100, 125]}
  subsample: {lat_step: 1.0, lon_step: 1.0}
generation:
  mode: hybrid
  model: hybrid
  samples: {unit: cell_time, rows: 20000}
physics:
  equation: advection_diffusion
  grid: {nx: 26, ny: 26, dx_m: 96000.0, dy_m: 111320.0}
  time: {dt_s: 600.0, steps: 15}
  parameters: {u0: 8.0, v0: 2.0, kappa: 100000.0}
  initial_condition: {kind: from_real, variable: tcwv}
  boundary_condition: {kind: periodic}
hybrid:
  physics_component: {enabled: true, output_variable: tcwv}
  statistical_component: {model: gaussian_copula, trained_on: [real, physics]}
  screening:
    hard_constraints: true
    reject_ratio_limit: 0.2
    soft_metrics: [mass_budget]
output: {format: netcdf, directory: ./output}
```

```bash
atmosgen validate-config configs/hybrid.yaml
atmosgen generate --config configs/hybrid.yaml
```

预期产物：`output/hybrid_hybrid_<ts>_42/` 内 `data.nc`（物理基底 tcwv + 条件生成的 t/q/u/v）、`rejected.jsonl`
（逐样本拒绝原因）、契约 `quality_metrics` 里的拒绝率。
**当前状态：代码已就位（`atmosgen/hybrid.py`、`generators/hybrid_adapter.py`），但工作区中的混合 run 尚未产出
`data.nc`，端到端结果标 NOT_RUN**，Phase 4 收尾时补跑并把拒绝率写入 `docs/EXPERIMENT_REPORT.md`。

### 4.4 真实数据 ingest（前置步骤）

```bash
atmosgen ingest --config configs/era5_ingest.yaml --overwrite
```

从 `/Volumes/Untitled/2010/` 只读抽取 2010-05-01..04 的 t/q/u/v（3 层、20–45°N/100–125°E、每 6 小时），
派生 tcwv，落 `data/raw/era5_local_slice.nc` + `data/processed/era5_local.nc`；实测 7.08 s / 峰值 RSS 426 MB，
源文件 size+mtime 全部未变（`docs/PHASE1_REPORT.md` §4）。

## 5. 扩展方式

### 5.1 新增一个统计模型

1. 在 `atmosgen/generators/` 新建模块，写一个 `BaseGenerator` 子类（`atmosgen/generators/base.py`）：
   实现 `fit(dataset)`、`sample() -> GenerationResult`（结果需带契约），可选 `sample_conditional()`；
2. 在 `atmosgen/generators/registry.py:97` 的 `MODELS` 注册：`ModelSpec(name, mode="statistical", description,
   phase, dependency_problems=..., implemented=True, factory=<惰性工厂>)`——工厂内部再 import，避免导入期拖慢 CLI；
3. 在 `atmosgen/config.py` 的 `MODELS_BY_MODE["statistical"]` 加入名字（否则 `validate-config` 会拒绝该模型名）；
4. 在 `configs/` 增加示例配置并跑 `atmosgen validate-config` + `atmosgen generate` 验证；
5. 建议补一条 `tests/` 用例：fit→sample→save→重读逐值相等（参考 `tests/test_sdv_adapter.py`）。

### 5.2 新增一组物理参数

不需要改代码：调 `physics.parameters`（`u0`/`v0`/`kappa`）、`physics.time`（`dt_s`/`steps`）、
`physics.grid`（`nx`/`ny`/`dx_m`/`dy_m`）与 `physics.initial_condition` 即可构成一个实验组；
`physics.time.steps` 与 `generation.samples.fields` 决定快照数与步长（步长被稳定性判据约束，改 `dt_s` 前先想清楚 CFL）。
若要**换方程**（而非换参数），则需新写 `generators/physics/<eq>.py` 求解器并注册 `ModelSpec(mode="physics")`，
详见 `docs/ARCHITECTURE.md` §5。

## 6. 常见失败与处置

| 现象 | 真实报错（实测） | 处置 |
|---|---|---|
| 统计配置用了场样本 | `ERROR generation: Value error, samples.unit=field 与表格模型（SDV）不兼容：场样本无法喂给单表模型；统计模式请使用 samples.unit=cell_time` | 改 `samples.unit: cell_time` 并给 `rows`（或留空=按网格行数） |
| 模型名写错 | `ERROR generation: Value error, generation.model='magic_net' 在 mode=statistical 下不可用；允许的模型: ['gaussian_copula', 'ctgan', 'tvae']` | 用 `atmosgen list-models` 查名字 |
| 物理参数不稳定 | `ERROR physics: Value error, physics 参数不满足显式格式稳定性要求: 2D 联合稳定性 (Cx+Cy)+2(Rx+Ry)=2.000>1（双轴 CFL≤1 不够；减小 dt_s、增大网格或降低 u0/v0/κ）` | 减小 `dt_s`、增大 `dx_m/dy_m` 或降低风速；注意双轴各 ≤1 仍不够 |
| 输入数据缺变量 | `ERROR data.variables 在输入数据中不存在: ['clwc','ciwc']；可用变量: ['q','t','tcwv','u','v']` | 先跑 ingest 生成规范数据，或调整 `variables` |
| 想写原始目录 | `RawWriteError: refusing to write … inside raw data directory …` | 原始数据只由 ingest 写一次；生成结果写 `output/` |
| 混合筛检拒绝率过高 | `rejected.jsonl` 中逐样本给出原因（`q_negative` / `t_out_of_range` / `saturation_violation` …） | 按原因分布调 `hybrid.screening.reject_ratio_limit` 或换统计模型；**不要**放宽硬约束阈值来"凑"通过（计划 X6） |
| 条件生成改写了物理基底 | `HybridError: 条件采样未原样保留 known 列 'tcwv'（最大偏差 …）` | 属不变量被破坏：检查 `sample_remaining_columns` 的 known 列类型/行序 |
| NaN 出现在生成样本 | 筛检把非有限值记为违规（`non_finite:<var>`），不会静默跳过 | 查上游模型或数据；`data.subsample` 减小规模可降低内存压力 |
| `evaluate` 退出码 4 | stdout 打印 `exit 4: 质量状态 FAIL（--fail-on never 可改为始终退出 0）` | 质量确实不达标（读报告定位指标），或加 `--fail-on never` 只取报告 |

## 7. 已知限制与后续

- Mode 3 端到端未验证（§4.3），本指南中的混合步骤以代码语义为准，实测数字待补。
- `generation.covariates` 在 Phase 2 的 level_wise 方案下只记录不使用；真正使用协变量的路径属 Phase 4 条件生成。
- 一次 `generate` 只能覆盖一个 `run_id`；批量实验请用 `scripts/run_experiments.py`（Phase 5）。
- `atmosgen ingest` 只支持本机归档与单个 NetCDF 样本；CDS 直连为预留路径（`source.kind: cds`）。
