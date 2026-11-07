"""Configuration models and validation (docs/IMPLEMENTATION_PLAN.md §6).

Two configuration kinds share the same YAML dialect:

- **generation configs** (``data:`` + ``generation:`` ...) — validated by
  :class:`GenerationConfigFile`;
- **ingest configs** (``source:`` + ``output:``) — validated by
  :class:`IngestConfigFile`.

Structural validation is always performed by :func:`validate_config_file`;
dataset-dependent checks (variables/time/level/extent really exist in the
input file) live in :func:`check_config_against_dataset` and are opt-in so a
structural check stays usable before data is ingested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .contract import ContractError, known_variables

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
STATISTICAL_MODELS: tuple[str, ...] = ("gaussian_copula", "ctgan", "tvae")
PHYSICS_MODELS: tuple[str, ...] = ("advection_diffusion",)
HYBRID_MODELS: tuple[str, ...] = ("hybrid",)

MODELS_BY_MODE: dict[str, tuple[str, ...]] = {
    "statistical": STATISTICAL_MODELS,
    "physics": PHYSICS_MODELS,
    "hybrid": HYBRID_MODELS,
}

ALLOWED_DIMS: tuple[str, ...] = ("time", "level", "latitude", "longitude")

#: Bounds for user-supplied validation threshold overrides (plan §5).
THRESHOLD_LIMITS: dict[str, tuple[float | None, float | None]] = {
    "column_shapes_min": (0.0, 1.0),
    "column_pair_trends_min": (0.0, 1.0),
    "mean_ratio_min": (0.0, None),
    "mean_ratio_max": (0.0, None),
    "std_ratio_min": (0.0, None),
    "std_ratio_max": (0.0, None),
    "quantile_rmse_max": (0.0, None),
    "ks_max": (0.0, 1.0),
    "corr_frobenius_max": (0.0, None),
    "acf_diff_max": (0.0, None),
    "diff_var_ratio_min": (0.0, None),
    "diff_var_ratio_max": (0.0, None),
    "spectral_slope_diff_max": (0.0, None),
    "variogram_ratio_min": (0.0, None),
    "variogram_ratio_max": (0.0, None),
    "gradient_rms_ratio_min": (0.0, None),
    "gradient_rms_ratio_max": (0.0, None),
    "t_range": (150.0, 400.0),
    "wind_max": (0.0, None),
    "tcwv_max": (0.0, None),
    "saturation_factor_max": (1.0, None),
}

_GRID_NAMES = ("nx", "ny", "dx_m", "dy_m")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sdv_available() -> tuple[bool, str]:
    """Return (available, version-or-reason) for the statistical stack."""

    try:
        import sdv  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on environment
        return False, f"{type(exc).__name__}: {exc}"
    return True, getattr(sdv, "__version__", "unknown")


class ConfigError(ValueError):
    """Raised when a configuration file cannot be parsed or validated."""

    def __init__(self, path: str | Path, problems: list[str]) -> None:
        self.path = Path(path)
        self.problems = problems
        super().__init__(
            f"invalid configuration {self.path}:\n" + "\n".join(f"  - {p}" for p in problems)
        )


def _format_validation_error(exc: ValidationError) -> list[str]:
    hints = {
        "missing": "必填字段缺失，请参考 configs/demo.yaml 或 configs/era5_ingest.yaml",
        "extra_forbidden": "未知字段，可能是拼写错误；字段清单见 docs/IMPLEMENTATION_PLAN.md §6.1",
        "literal_error": "枚举值不合法，允许值见 docs/IMPLEMENTATION_PLAN.md §6.1",
        "int_type": "类型错误：需要整数",
        "int_parsing": "类型错误：需要整数",
        "float_type": "类型错误：需要数值",
        "float_parsing": "类型错误：需要数值",
        "string_type": "类型错误：需要字符串",
        "bool_type": "类型错误：需要 true/false",
        "model_type": "类型错误：该字段应为嵌套映射（检查 YAML 缩进）",
        "list_type": "类型错误：该字段应为列表",
        "dict_type": "类型错误：该字段应为映射",
        "path_type": "类型错误：该字段应为路径字符串",
    }
    lines: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        message = error["msg"]
        hint = hints.get(error["type"], "")
        suffix = f"（{hint}）" if hint else ""
        lines.append(f"{location}: {message}{suffix}")
    return lines


def _parse_dataset_yaml(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(target, [f"配置文件不存在: {target}"]) from exc
    except yaml.YAMLError as exc:
        raise ConfigError(target, [f"YAML 解析失败: {exc}"]) from exc
    if not isinstance(raw, dict):
        raise ConfigError(target, ["顶层结构必须是映射（key: value）"])
    return raw


def _is_ingest_config(raw: Mapping[str, Any]) -> bool:
    return "source" in raw and "generation" not in raw


# --------------------------------------------------------------------------- #
# Shared nested models
# --------------------------------------------------------------------------- #
class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=False)


class ProjectConfig(_StrictModel):
    name: str = Field(min_length=1)
    seed: int = 42
    output_dir: Path | None = None


class CompressionConfig(_StrictModel):
    zlib: bool = True
    complevel: int = Field(default=4, ge=0, le=9)


class SubsetConfig(_StrictModel):
    levels: list[float] | None = None
    latitude: tuple[float, float] | None = None
    longitude: tuple[float, float] | None = None
    time_step_hours: int | None = Field(default=None, ge=1, le=24)

    @model_validator(mode="after")
    def _check(self) -> "SubsetConfig":
        for name, bounds in (("latitude", self.latitude), ("longitude", self.longitude)):
            if bounds is None:
                continue
            low, high = float(bounds[0]), float(bounds[1])
            if low > high:
                raise ValueError(f"subset.{name} 下限 {low} 大于上限 {high}")
            limit = 90.0 if name == "latitude" else 360.0
            if low < -limit or high > limit:
                raise ValueError(f"subset.{name} 超出合法范围 [-{limit}, {limit}]")
        if self.levels is not None:
            if not self.levels:
                raise ValueError("subset.levels 不能为空列表；不需要筛选请删除该字段")
            if min(self.levels) <= 0 or max(self.levels) > 1100:
                raise ValueError("subset.levels 必须在 (0, 1100] hPa 内")
        if self.time_step_hours is not None and 24 % self.time_step_hours != 0:
            raise ValueError(
                f"subset.time_step_hours={self.time_step_hours} 不能整除 24，无法对齐逐日文件"
            )
        return self


class DeriveConfig(_StrictModel):
    """Derived-variable recipes, e.g. tcwv from vertical integration of q."""

    tcwv: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check(self) -> "DeriveConfig":
        if self.tcwv is not None:
            method = self.tcwv.get("method")
            if method != "vertical_integral_q":
                raise ValueError(
                    "derive.tcwv.method 目前只支持 vertical_integral_q（tcwv = (1/g)∫q dp）"
                )
        return self


class DataConfig(_StrictModel):
    input: Path
    format: Literal["auto", "netcdf", "zarr"] = "auto"
    variables: list[str] = Field(min_length=1)
    time_range: tuple[str, str] | None = None
    levels: list[float] | None = None
    spatial_extent: dict[str, tuple[float, float]] | None = None
    chunks: dict[str, int] | None = None
    source_variable_map: dict[str, str] | None = None
    subsample: dict[str, float] | None = None

    @model_validator(mode="after")
    def _check(self) -> "DataConfig":
        unknown = [name for name in self.variables if name not in known_variables()]
        if unknown:
            raise ValueError(
                f"data.variables 含未知变量 {unknown}；规范变量为 {known_variables()}"
            )
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("data.variables 含重复项")
        if self.levels is not None and (not self.levels or min(self.levels) <= 0):
            raise ValueError("data.levels 必须为非空正数列表（hPa）")
        if self.chunks is not None:
            bad = [key for key in self.chunks if key not in ALLOWED_DIMS]
            if bad:
                raise ValueError(f"data.chunks 含未知维度 {bad}；允许 {ALLOWED_DIMS}")
            if any(value <= 0 for value in self.chunks.values()):
                raise ValueError("data.chunks 的每个维度大小必须为正整数")
        if self.subsample is not None:
            bad = [key for key in self.subsample if key not in {"lat_step", "lon_step"}]
            if bad:
                raise ValueError(f"data.subsample 含未知键 {bad}；允许 lat_step / lon_step")
            if any(value <= 0 for value in self.subsample.values()):
                raise ValueError("data.subsample 的步长必须 > 0")
        if self.spatial_extent is not None:
            bad = [key for key in self.spatial_extent if key not in {"lat", "lon"}]
            if bad:
                raise ValueError(f"data.spatial_extent 含未知键 {bad}；允许 lat / lon")
        return self


class SamplesConfig(_StrictModel):
    unit: Literal["cell_time", "field"] = "cell_time"
    rows: int | None = Field(default=None, ge=1)
    fields: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check(self) -> "SamplesConfig":
        if self.unit == "cell_time" and self.rows is None:
            raise ValueError("samples.unit=cell_time 时必须提供 samples.rows（表格行样本数）")
        if self.unit == "field" and self.fields is None:
            raise ValueError("samples.unit=field 时必须提供 samples.fields（场/时间步数）")
        return self


class TrainingConfig(_StrictModel):
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=500, ge=1)
    discriminator_steps: int = Field(default=1, ge=1)


class GenerationConfig(_StrictModel):
    mode: Literal["statistical", "physics", "hybrid"]
    model: str
    samples: SamplesConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    covariates: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "GenerationConfig":
        allowed = MODELS_BY_MODE[self.mode]
        if self.model not in allowed:
            raise ValueError(
                f"generation.model={self.model!r} 在 mode={self.mode} 下不可用；"
                f"允许的模型: {list(allowed)}"
            )
        if self.mode == "statistical" and self.samples.unit == "field":
            raise ValueError(
                "samples.unit=field 与表格模型（SDV）不兼容：场样本无法喂给单表模型；"
                "统计模式请使用 samples.unit=cell_time"
            )
        if self.mode == "statistical":
            available, detail = _sdv_available()
            if not available:
                raise ValueError(
                    "mode=statistical 需要统计生成依赖（sdv/sdmetrics），当前环境不可用: "
                    f"{detail}；请 pip install -e '.[stat]'"
                )
        return self


class GridConfig(_StrictModel):
    nx: int = Field(ge=8)
    ny: int = Field(ge=8)
    dx_m: float = Field(gt=0)
    dy_m: float = Field(gt=0)


class PhysicsTimeConfig(_StrictModel):
    dt_s: float = Field(gt=0)
    steps: int = Field(ge=1)


class PhysicsParameters(_StrictModel):
    u0: float = 0.0
    v0: float = 0.0
    kappa: float = Field(default=0.0, ge=0.0)


class InitialCondition(_StrictModel):
    model_config = ConfigDict(extra="allow")

    kind: Literal["gaussian_plume", "from_real"]
    center: tuple[float, float] | None = None
    sigma_deg: float | None = Field(default=None, gt=0)
    amplitude: float = 1.0
    variable: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "InitialCondition":
        if self.kind == "gaussian_plume":
            if self.center is None or self.sigma_deg is None:
                raise ValueError("initial_condition.kind=gaussian_plume 需要 center=[lat, lon] 与 sigma_deg")
        if self.kind == "from_real" and not self.variable:
            raise ValueError("initial_condition.kind=from_real 需要 variable（如 tcwv）")
        return self


class BoundaryCondition(_StrictModel):
    model_config = ConfigDict(extra="allow")

    kind: Literal["periodic", "open", "fixed_from_real"]


class PhysicsConfig(_StrictModel):
    equation: Literal["advection_diffusion"]
    grid: GridConfig
    time: PhysicsTimeConfig
    parameters: PhysicsParameters
    initial_condition: InitialCondition
    boundary_condition: BoundaryCondition

    @model_validator(mode="after")
    def _check_cfl(self) -> "PhysicsConfig":
        dt = self.time.dt_s
        cfl_x = abs(self.parameters.u0) * dt / self.grid.dx_m
        cfl_y = abs(self.parameters.v0) * dt / self.grid.dy_m
        diffusion = self.parameters.kappa * dt / min(self.grid.dx_m, self.grid.dy_m) ** 2
        problems = []
        if cfl_x > 1.0:
            problems.append(f"CFL_x=|u0|·dt/dx={cfl_x:.3f}>1（减小 dt_s 或增大 dx_m）")
        if cfl_y > 1.0:
            problems.append(f"CFL_y=|v0|·dt/dy={cfl_y:.3f}>1（减小 dt_s 或增大 dy_m）")
        if diffusion > 0.5:
            problems.append(
                f"扩散稳定性 κ·dt/min(dx,dy)²={diffusion:.3f}>0.5（减小 dt_s 或增大网格间距）"
            )
        if problems:
            raise ValueError("physics 参数不满足显式格式稳定性要求: " + "; ".join(problems))
        return self


class HybridPhysicsComponent(_StrictModel):
    enabled: bool = True
    output_variable: str = "tcwv"

    @model_validator(mode="after")
    def _check(self) -> "HybridPhysicsComponent":
        if self.output_variable not in known_variables():
            raise ValueError(
                f"hybrid.physics_component.output_variable={self.output_variable!r} 不是规范变量"
            )
        return self


class HybridStatisticalComponent(_StrictModel):
    model: str = "gaussian_copula"
    trained_on: list[str] = Field(default_factory=lambda: ["real"])

    @model_validator(mode="after")
    def _check(self) -> "HybridStatisticalComponent":
        if self.model not in STATISTICAL_MODELS:
            raise ValueError(
                f"hybrid.statistical_component.model 必须为 {list(STATISTICAL_MODELS)} 之一"
            )
        bad = [item for item in self.trained_on if item not in {"real", "physics"}]
        if bad:
            raise ValueError(f"trained_on 只允许 real / physics，得到 {bad}")
        return self


class HybridScreening(_StrictModel):
    hard_constraints: bool = True
    reject_ratio_limit: float = Field(default=0.2, ge=0.0, le=1.0)
    soft_metrics: list[str] = Field(default_factory=list)


class HybridConfig(_StrictModel):
    physics_component: HybridPhysicsComponent = Field(default_factory=HybridPhysicsComponent)
    statistical_component: HybridStatisticalComponent = Field(
        default_factory=HybridStatisticalComponent
    )
    screening: HybridScreening = Field(default_factory=HybridScreening)


class ValidationConfig(_StrictModel):
    statistical: bool = True
    temporal: bool = True
    spatial: bool = True
    physical: bool = True
    thresholds_overrides: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_thresholds(self) -> "ValidationConfig":
        problems: list[str] = []
        for key, value in self.thresholds_overrides.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                problems.append(f"thresholds_overrides.{key} 必须为有限数值")
                continue
            limits = THRESHOLD_LIMITS.get(key)
            if limits is None:
                continue
            low, high = limits
            if low is not None and value < low:
                problems.append(f"thresholds_overrides.{key}={value} 小于下限 {low}")
            if high is not None and value > high:
                problems.append(f"thresholds_overrides.{key}={value} 大于上限 {high}")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def unknown_thresholds(self) -> list[str]:
        """Override keys that are not in :data:`THRESHOLD_LIMITS` (warn, not error)."""

        return sorted(key for key in self.thresholds_overrides if key not in THRESHOLD_LIMITS)


class OutputConfig(_StrictModel):
    format: Literal["netcdf", "zarr"] = "netcdf"
    directory: Path | None = None
    compression: CompressionConfig = Field(default_factory=CompressionConfig)


# --------------------------------------------------------------------------- #
# Root models
# --------------------------------------------------------------------------- #
class GenerationConfigFile(_StrictModel):
    project: ProjectConfig
    data: DataConfig | None = None
    generation: GenerationConfig
    physics: PhysicsConfig | None = None
    hybrid: HybridConfig | None = None
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _check_mode_requirements(self) -> "GenerationConfigFile":
        mode = self.generation.mode
        problems: list[str] = []
        if mode in {"statistical", "hybrid"} and self.data is None:
            problems.append(f"mode={mode} 需要 data 段（训练/条件生成需要真实数据）")
        if mode in {"physics", "hybrid"} and self.physics is None:
            problems.append(f"mode={mode} 需要 physics 段（方程/网格/参数/IC/BC）")
        if mode == "physics" and self.physics is not None and self.data is not None:
            # allowed: from_real initial conditions need real fields
            pass
        if mode == "hybrid" and self.hybrid is None:
            problems.append("mode=hybrid 需要 hybrid 段（physics_component/statistical_component/screening）")
        if mode != "hybrid" and self.hybrid is not None:
            problems.append("hybrid 段只在 mode=hybrid 时允许")
        if mode == "physics" and self.physics is not None:
            if self.physics.initial_condition.kind == "from_real" and self.data is None:
                problems.append(
                    "physics.initial_condition.kind=from_real 需要 data 段提供真实场"
                )
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def output_directory(self) -> Path:
        """Resolved output directory (``output.directory`` wins over project)."""

        if self.output.directory is not None:
            return Path(self.output.directory)
        if self.project.output_dir is not None:
            return Path(self.project.output_dir)
        return Path("./output")


class SourceConfig(_StrictModel):
    kind: Literal["local_era5", "local_netcdf", "cds"]
    root: Path | None = None
    input: Path | None = None
    years: list[int] | None = None
    date_range: tuple[str, str] | None = None
    file_pattern: str | None = None
    vertical: str = "pressure_level"
    variable_map: dict[str, str] = Field(default_factory=dict)
    subset: SubsetConfig | None = None
    derive: DeriveConfig | None = None
    request: dict[str, Any] | None = None

    @field_validator("date_range", mode="before")
    @classmethod
    def _coerce_date_range(cls, value: Any) -> Any:
        """Accept unquoted YAML dates (``[2010-05-01, 2010-05-04]``) as strings."""

        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return tuple(
                item.isoformat() if hasattr(item, "isoformat") else item for item in value
            )
        return value

    @model_validator(mode="after")
    def _check(self) -> "SourceConfig":
        problems: list[str] = []
        if self.vertical not in {"pressure_level", "model_level"}:
            problems.append("source.vertical 必须为 pressure_level 或 model_level")
        # Per-kind field whitelist: a field that only one source kind supports
        # must be rejected for the other kinds instead of being silently ignored.
        archive_only = {
            "root": self.root,
            "years": self.years,
            "date_range": self.date_range,
            "file_pattern": self.file_pattern,
        }
        if self.kind == "local_netcdf":
            for field_name, value in archive_only.items():
                if value is not None:
                    problems.append(
                        f"source.{field_name} 只适用于 local_era5；local_netcdf 请删除该字段"
                    )
            if self.request is not None:
                problems.append("source.request 只适用于 cds；local_netcdf 请删除该字段")
        elif self.kind == "local_era5":
            if self.input is not None:
                problems.append("source.input 只适用于 local_netcdf；local_era5 请用 root + file_pattern")
            if self.request is not None:
                problems.append("source.request 只适用于 cds；local_era5 请删除该字段")
        elif self.kind == "cds":
            for field_name, value in archive_only.items():
                if value is not None:
                    problems.append(
                        f"source.{field_name} 只适用于 local_era5；cds 请用 request 描述下载请求"
                    )
            if self.input is not None:
                problems.append("source.input 只适用于 local_netcdf；cds 请用 request 描述下载请求")
        bad_vars = [name for name in self.variable_map if name not in known_variables()]
        if bad_vars:
            problems.append(f"source.variable_map 含未知规范变量 {bad_vars}")
        if self.kind == "local_era5":
            if self.root is None:
                problems.append("source.kind=local_era5 需要 root（本机 ERA5 归档根目录）")
            if self.date_range is None:
                problems.append("source.kind=local_era5 需要 date_range=[开始日期, 结束日期]")
            if self.file_pattern is None:
                problems.append(
                    "source.kind=local_era5 需要 file_pattern，例如 "
                    "'ERA5_025dg_{date:%Y%m%d}_{variable}.nc'"
                )
            if self.date_range is not None:
                from datetime import date

                try:
                    start = date.fromisoformat(self.date_range[0])
                    end = date.fromisoformat(self.date_range[1])
                except ValueError:
                    problems.append("source.date_range 必须是 'YYYY-MM-DD' 字符串")
                else:
                    if start > end:
                        problems.append("source.date_range 起始日期晚于结束日期")
        elif self.kind == "local_netcdf":
            if self.input is None:
                problems.append("source.kind=local_netcdf 需要 input（源 NetCDF 路径）")
            if self.derive is not None and self.derive.tcwv is not None:
                problems.append(
                    "source.kind=local_netcdf 暂不支持 derive.tcwv（Phase 1 只在 local_era5 上派生）："
                    "请直接提供 tcwv 变量，或改用 source.kind=local_era5；"
                    "不做静默忽略，避免配置合法但产物缺变量"
                )
        elif self.kind == "cds":
            if not self.request:
                problems.append("source.kind=cds 需要 request（CDS API 请求参数）")
        if problems:
            raise ValueError("; ".join(problems))
        return self


class IngestOutputConfig(_StrictModel):
    raw_path: Path | None = None
    processed_path: Path
    format: Literal["netcdf", "zarr"] = "netcdf"
    read_only_assertion: bool = True


class IngestConfigFile(_StrictModel):
    project: ProjectConfig
    source: SourceConfig
    output: IngestOutputConfig


# --------------------------------------------------------------------------- #
# Loading and structural validation
# --------------------------------------------------------------------------- #
@dataclass
class ConfigCheck:
    """Outcome of validating one configuration file."""

    path: Path
    kind: str
    name: str | None = None
    valid: bool = False
    model: BaseModel | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        state = "PASS" if self.valid else "FAIL"
        lines = [f"[{state}] {self.path}  ({self.kind}{' · ' + self.name if self.name else ''})"]
        lines.extend(f"  ERROR {err}" for err in self.errors)
        lines.extend(f"  WARN  {warn}" for warn in self.warnings)
        return "\n".join(lines)


def load_config_file(path: str | Path) -> GenerationConfigFile | IngestConfigFile:
    """Parse and structurally validate a configuration file."""

    target = Path(path)
    raw = _parse_dataset_yaml(target)
    kind = "ingest" if _is_ingest_config(raw) else "generation"
    model_cls = IngestConfigFile if kind == "ingest" else GenerationConfigFile
    try:
        return model_cls.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(target, _format_validation_error(exc)) from exc


def validate_config_file(
    path: str | Path, *, check_data: bool = False
) -> ConfigCheck:
    """Validate a configuration file; optionally check it against its input data.

    ``check_data=True`` opens ``data.input`` (generation configs only) and
    verifies that requested variables / time range / levels / extent exist.
    """

    target = Path(path)
    check = ConfigCheck(path=target, kind="unknown")
    try:
        raw = _parse_dataset_yaml(target)
    except ConfigError as exc:
        check.errors.extend(exc.problems)
        return check
    check.kind = "ingest" if _is_ingest_config(raw) else "generation"
    model_cls = IngestConfigFile if check.kind == "ingest" else GenerationConfigFile
    try:
        model = model_cls.model_validate(raw)
    except ValidationError as exc:
        check.errors.extend(_format_validation_error(exc))
        return check

    check.model = model
    check.name = model.project.name
    check.valid = True
    if isinstance(model, GenerationConfigFile):
        for key in model.validation.unknown_thresholds():
            check.warnings.append(
                f"validation.thresholds_overrides.{key} 不是已知指标名，将作为自定义阈值原样传递"
            )
        if model.data is not None:
            check.warnings.extend(
                _generation_config_warnings(model)
            )
        if check_data and model.data is not None:
            problems = check_config_against_dataset(model)
            if problems:
                check.valid = False
                check.errors.extend(problems)
    else:
        if model.source.derive is not None and model.source.derive.tcwv is not None:
            check.warnings.append(
                "tcwv 为 q 垂直积分派生量（derived），报告中不得当作 ERA5 官方 tcwv"
            )
    return check


def _generation_config_warnings(model: GenerationConfigFile) -> list[str]:
    warnings: list[str] = []
    if model.data is not None and model.data.input is not None:
        if not Path(model.data.input).exists():
            warnings.append(
                f"data.input={model.data.input} 当前不存在：结构校验通过，数据级校验已跳过"
                "（先运行 atmosgen ingest）"
            )
    if model.generation.mode == "statistical":
        available, detail = _sdv_available()
        warnings.append(f"统计生成依赖 sdv={detail}" if available else f"统计生成依赖不可用: {detail}")
    return warnings


def check_config_against_dataset(
    config: GenerationConfigFile,
) -> list[str]:
    """Verify that the requested variables/time/level/extent exist in ``data.input``."""

    problems: list[str] = []
    data = config.data
    if data is None:
        return problems
    target = Path(data.input)
    if not target.exists():
        return [f"data.input={target} 不存在，无法做数据级校验"]

    from .data.io import open_source

    try:
        ds = open_source(
            target,
            format=data.format,
            chunks=None,
            variables=None,
        )
    except Exception as exc:  # pragma: no cover - IO failure formatting
        return [f"无法读取 data.input={target}: {type(exc).__name__}: {exc}"]
    try:
        available = set(ds.variables)
        missing = [name for name in data.variables if name not in available]
        if missing:
            problems.append(
                f"data.variables 在输入数据中不存在: {missing}；可用变量: {sorted(ds.data_vars)}"
            )
        if data.time_range is not None:
            start, end = data.time_range
            try:
                import numpy as np

                requested = (np.datetime64(start), np.datetime64(end))
            except Exception:
                problems.append(f"data.time_range 无法解析: {data.time_range}")
            else:
                actual = ds["time"].values
                if actual.size:
                    if requested[0] < actual.min() or requested[1] > actual.max():
                        problems.append(
                            f"data.time_range {data.time_range} 超出输入数据时间轴 "
                            f"[{actual.min()}, {actual.max()}]"
                        )
        if data.levels is not None and "level" in ds.coords:
            actual_levels = [float(value) for value in ds["level"].values]
            missing_levels = [value for value in data.levels if value not in actual_levels]
            if missing_levels:
                problems.append(
                    f"data.levels {missing_levels} 不在输入数据层坐标中；可用层: {actual_levels}"
                )
        if data.spatial_extent is not None:
            lat = ds["latitude"].values
            lon = ds["longitude"].values
            if "lat" in data.spatial_extent:
                low, high = data.spatial_extent["lat"]
                if low < float(lat.min()) or high > float(lat.max()):
                    problems.append(
                        f"data.spatial_extent.lat {list(data.spatial_extent['lat'])} 超出输入范围 "
                        f"[{float(lat.min())}, {float(lat.max())}]"
                    )
            if "lon" in data.spatial_extent:
                low, high = data.spatial_extent["lon"]
                if low < float(lon.min()) or high > float(lon.max()):
                    problems.append(
                        f"data.spatial_extent.lon {list(data.spatial_extent['lon'])} 超出输入范围 "
                        f"[{float(lon.min())}, {float(lon.max())}]"
                    )
    finally:
        ds.close()
    return problems


def config_sha256(path: str | Path) -> str:
    """SHA-256 of the raw configuration file (provenance)."""

    target = Path(path)
    import hashlib

    return hashlib.sha256(target.read_bytes()).hexdigest()


__all__ = [
    "ALLOWED_DIMS",
    "ConfigCheck",
    "ConfigError",
    "DataConfig",
    "GenerationConfig",
    "GenerationConfigFile",
    "HybridConfig",
    "IngestConfigFile",
    "MODELS_BY_MODE",
    "PhysicsConfig",
    "SourceConfig",
    "ValidationConfig",
    "check_config_against_dataset",
    "config_sha256",
    "load_config_file",
    "validate_config_file",
]
