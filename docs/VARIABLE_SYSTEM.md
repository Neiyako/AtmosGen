# AtmosGen 变量体系说明（schema 1.2）

本文档描述 1.2 版变量体系扩充：新增变量的物理定义、单位约定、依赖关系、生成方式与验证口径。变量注册的唯一来源是 `atmosgen/variables.py`；派生计算在 `atmosgen/derived.py`；物理验证在 `atmosgen/validation/physical.py`。

- 产品定位不变：不做数值模式，不新增 ML 模型/第三方依赖；扩充复用现有二维平流扩散求解器、数据契约、验证器与配置系统。

## 一、单元约定（必须全文遵守）

单位采用 CF 风格指数写法（`kg kg**-1`），坐标一律 `time / level / latitude / longitude`，level 单位 hPa（气压层）。

变量约定（任务书留白处逐一钉死）：

| 变量 | 约定 |
|---|---|
| `w` | 气压坐标垂直速度，`Pa s**-1`（ERA5 惯例），正 = 上升（气压减小）。不是 m/s。 |
| `z` | 位势（geopotential），`m**2 s**-2`（ERA5 惯例）。位势高度 = z/g，g=9.80665。 |
| `tp` | 累积总降水量：**该时间戳为止的区间内积分量**，kg/m²（= mm 液态水当量）。不是瞬时率；瞬时率 = tp/dt。 |
| `clwc/ciwc/crwc/cswc` | 质量混合比 kg kg⁻¹（非柱量），恒 ≥ 0。 |
| `tcwv/tclw/tciw/cwp/iwp/twp` | 柱积分 kg m⁻²，与单层混合比严格区分。 |
| `rh`/`r` | 同一量的两个名字（`r` 是 `rh` 的短别名，同公式同值同单位）。 |
| `mfc` vs `mfc_sl` | `mfc` 是**柱积分**水汽通量辐合（kg m⁻² s⁻¹，需要垂直层）；`mfc_sl` 是**单层**诊断（kg kg⁻¹ s⁻¹，取最低层）。两者分开注册，绝不让柱量冒充单层量。 |

## 二、变量清单（27 个，新增 17 个）

主变量（生成器直接产出 / ingest 直接映射）：`t q u v w z sp clwc ciwc crwc cswc tp c`

派生变量（由派生引擎计算）：`tcwv tclw tciw rh r qflux_u qflux_v qflux_u_col qflux_v_col mfc mfc_sl cwp iwp twp`

新增变量定义（缺省含义见 `variables.py` 的 `long_name`/`formula`/`comment`）：

| 变量 | 定义 | 单位 | dims | source | requires |
|---|---|---|---|---|---|
| `sp` | 地面气压 | Pa | (time,lat,lon) | primary | — |
| `w` | 气压坐标垂直速度（正=上升） | Pa s**-1 | (time,level,lat,lon) | primary | — |
| `z` | 位势（水静力积分） | m**2 s**-2 | (time,level,lat,lon) | primary | — |
| `tp` | 累积总降水量（区间积分） | kg m**-2 | (time,lat,lon) | primary | — |
| `rh`/`r` | 相对湿度 `100·q/qsat(t,p)`（Tetens） | % | (time,level,lat,lon) | derived | t, q |
| `tclw`/`cwp` | `(1/g)∫clwc dp`（两个名字同义） | kg m**-2 | (time,lat,lon) | derived | clwc |
| `tciw`/`iwp` | `(1/g)∫ciwc dp`（两个名字同义） | kg m**-2 | (time,lat,lon) | derived | ciwc |
| `twp` | `cwp + iwp`（总凝态水路径；**不含水汽**，水汽单独记 tcwv） | kg m**-2 | (time,lat,lon) | derived | cwp, iwp |
| `qflux_u`/`qflux_v` | `q·u` / `q·v` | kg kg**-1 m s**-1 | (time,level,lat,lon) | derived | q, u / q, v |
| `qflux_u_col`/`qflux_v_col` | `(1/g)∫qflux dp` | kg m**-1 s**-1 | (time,lat,lon) | derived | qflux_* |
| `mfc` | `-div(qflux_u_col, qflux_v_col)` | kg m**-2 s**-1 | (time,lat,lon) | derived | qflux_*_col |
| `mfc_sl` | `-div(qflux_u, qflux_v)` 于最低层 | kg kg**-1 s**-1 | (time,lat,lon) | derived | qflux_* |

## 三、依赖关系与三方生成策略

生成策略 A/B/C 依任务书分层落位：

1. **A 基础变量**：`multifield` 生成器（`generators/physics/multifield.py`，phase 7）复用现有二维平流–扩散求解器，跑两次独立模式场（旋转部分 driver A + 散度部分 driver B，周期边界、最小像距离羽流），由 `t = T_bg(p) + ΔT·shape(p)·anomA` 构造温度扰动，水汽按 RH 目标场构造，`q = min(RH_target·qsat, qsat)` —— 超饱和部分**凝结成云水**（质量守恒，不丢弃）：
   - `ice_fraction = clip((freezing_level + ice_ramp/2 − T)/ice_ramp, 0, 1)` 分配液态/冰。
   - 凝结按 `precipitation_fraction` 分离出降水：柱质量 `precip_col = frac·Σqc·Δp/g`，按 `prof(p) = exp(-(p-p_base)/150hPa)` 归一化 `Σ prof·Δp = 1` 落到云底以下层：`mixing_ratio = g·precip_col·prof`（守恒自检 1e-6）。
   - `w` 由水平散度的向上积分（w=0 于地面）给出 `ω(k) = ω(k-1) − 0.5(D(k-1)+D(k))·Δp`。
   - `z` 由水静力方程积分 `Φ(p) = R_d·T_s·ln(sp/p0) − ∫R_d·T/p dp`。
   - 平均气象量级：t∈[218,294] K，sp∈[1.00,1.03]×10⁵ Pa，|u|,|v|≤16 m/s，|w|≤2 Pa/s，z(100hPa)≈1.7×10⁵ m²/s²，cwp≤4 kg/m²，tp≤0.3 mm/10min。
2. **B 关联变量**：`atmosgen/derived.py` 由已生成的主变量派生（相对湿度、水汽通量、柱积分、MFC、总云水）。**柱积分统一 `(1/g)·∫·dp`梯形法**，与 `derive_tcwv` 同口径。缺层坐标/主变量 → `NOT_AVAILABLE`+原因，**绝不写 0 占位**。
3. **C 简化物理场**：`sp/w/z` 与 t 水静力耦合、`q ≤ qsat(t,p)`、`ω` 与水平散度一致 —— 温度-气压-湿度-位势自洽，不开发完整动力学。

## 四、配置选择与依赖检查

```yaml
generation:
  variables: [t, q, ..., mfc_sl]   # 可选；缺省 = 不改变旧行为
```

- `multifield:` 段承载耦合场参数（grid/time/levels_hpa/surface/pattern/tracer/cloud/precipitation），`generation.model==multifield` 时必需、与其他模型互斥；`validate-config` 在生成前做平流–扩散稳定性判定。
- `generation.variables` 一经给出：先做依赖检查（不可满足 → 生成前硬报错），生成后引擎补派生量并按请求集合裁剪；期间不可交付即标记 `NOT_AVAILABLE` 并在 metadata/CLI 显式给原因。旧配置（无该字段）行为完全不变。

## 五、验证口径（物理一致性）

`run_physical_validation` 新增 9 个指标：`sp_range / w_range / z_range / tp_nonneg / rh_range / column_consistency / tcwv_integral_consistency / flux_consistency / mfc_consistency`。

硬约束与容差策略：

- 范围硬约束：sp∈[5e4, 1.1e5] Pa、|w|≤10 Pa/s、tp≥0≤1e3 kg/m²、rh∈[0,200]%。
- `z_range` 是**逐层水静力参考带**（z_ref = R_d·T·ln(p_s/p)，T∈[200,320] K，±2000 m²/s² 基准面放宽容差），而非全局区间 —— 上层位势 ~1.7e5 m²/s² 属合法值，全局区间会误判。
- `rh_range`：`>200%` FAIL（物理不可能）；`(120%, 200%]` CONDITIONAL_PASS（云内过饱和物理可能，**报告不裁剪**）；≤120% PASS。
- 定义一致性（硬检查，`consistency_rtol=1e-6/atol=1e-9`）：`twp == cwp+iwp`、`tclw == cwp`、`tciw == iwp`、`qflux_* == q·*`、`mfc == -div(col flux)`（数值偏差写明，绝不为过测试裁剪极端事件）。
- `tcwv_integral_consistency`：联合生成模式（statistical/hybrid）中 tcwv 是独立柱量而非 q 的积分，偏差>容差记 CONDITIONAL_PASS，不伪装成物理因果。
- **NOT_RUN ≠ PASS**：缺输入 → NOT_RUN 并写原因；可选变量缺失 → INFO（不参与聚合）。方程残差对 multifield 数据集不适用（无示踪 c），记 INFO。

### 外部参照
- 未联网查外部参照（用户未要求，按复用项目内部方案执行）。
