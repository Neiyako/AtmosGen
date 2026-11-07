"""Typer CLI for AtmosGen (plan §6.2).

Exit codes
----------
0  success
2  invalid configuration / user error
3  command not implemented in the current phase (NOT_RUN, never silently skipped)
4  runtime failure (ingest, IO, contract)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .config import (
    ConfigError,
    GenerationConfigFile,
    IngestConfigFile,
    load_config_file,
    validate_config_file,
)
from .contract import ContractError
from .data.preprocess import IngestError, ingest
from .data.selection import SubsetError
from .evaluation import EvaluationError, read_quality_report, run_evaluation
from .generation import GenerationError, NotImplementedInPhase, run_generation
from .generators.registry import registry_report

EXIT_OK = 0
EXIT_USER_ERROR = 2
EXIT_NOT_IMPLEMENTED = 3
EXIT_RUNTIME_ERROR = 4

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="AtmosGen — 大气仿真数据生成框架（数据契约 / IO / 生成适配 / 质量验证）",
)


def _echo_error(message: str) -> None:
    typer.echo(f"ERROR: {message}", err=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"atmosgen {__version__}")
        raise typer.Exit(code=EXIT_OK)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="打印版本号并退出",
    ),
) -> None:
    """AtmosGen 命令行入口。"""


@app.command("validate-config")
def command_validate_config(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="YAML 配置文件"),
    check_data: bool = typer.Option(
        True,
        "--check-data/--no-check-data",
        help="同时校验输入数据中变量/时间/层/空间范围是否满足配置（默认开启）",
    ),
) -> None:
    """校验配置文件（结构 + 跨字段 + 可选数据级检查）。"""

    try:
        check = validate_config_file(config, check_data=check_data)
    except Exception as exc:  # noqa: BLE001 - keep the documented exit-code contract
        _echo_error(f"validate-config 运行失败（未预期异常）: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=EXIT_RUNTIME_ERROR)
    typer.echo(check.summary())
    if not check.valid:
        typer.echo("result: FAIL（修复上方 ERROR 后重试；字段规范见 docs/IMPLEMENTATION_PLAN.md §6.1）")
        raise typer.Exit(code=EXIT_USER_ERROR)
    typer.echo(f"result: PASS（{len(check.warnings)} warning(s)，退出码 0）")


@app.command("ingest")
def command_ingest(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False, readable=True, help="ingest 配置文件"),
    overwrite: bool = typer.Option(False, "--overwrite", help="覆盖已存在的输出文件"),
) -> None:
    """真实数据 ingest：本机 ERA5 归档 / CDS 小样本 → 统一契约数据集（Test A）。"""

    code = run_ingest(config_path=config, overwrite=overwrite)
    raise typer.Exit(code=code)


def run_ingest(*, config_path: Path, overwrite: bool = False) -> int:
    """Implementation shared by the CLI command and ``scripts/ingest_local_era5.py``."""

    try:
        config = load_config_file(config_path)
    except ConfigError as exc:
        _echo_error(str(exc))
        return EXIT_USER_ERROR
    if not isinstance(config, IngestConfigFile):
        _echo_error(
            f"{config_path} 是 generation 配置（含 generation 段），不是 ingest 配置；"
            "ingest 需要 source 段，例如 configs/era5_ingest.yaml"
        )
        return EXIT_USER_ERROR
    try:
        report = ingest(config, config_path=config_path, overwrite=overwrite)
    except (
        IngestError,
        ContractError,
        FileExistsError,
        FileNotFoundError,
        KeyError,
        ValueError,
        OSError,
    ) as exc:
        _echo_error(f"ingest 失败: {type(exc).__name__}: {exc}")
        return EXIT_RUNTIME_ERROR
    except Exception as exc:  # noqa: BLE001 - CLI must not leak tracebacks/exit 1
        _echo_error(f"ingest 运行失败（未预期异常）: {type(exc).__name__}: {exc}")
        return EXIT_RUNTIME_ERROR
    typer.echo(report.render_text())
    if report.status == "FAIL":
        _echo_error(f"ingest 未通过（status=FAIL），详见 {report.report_json}")
        return EXIT_RUNTIME_ERROR
    if report.status == "CONDITIONAL_PASS":
        typer.echo(
            "result: CONDITIONAL_PASS（存在 NOT_RUN / 部分通过项，明细见报告；"
            f"报告: {report.report_json}）"
        )
        return EXIT_OK
    typer.echo(f"result: PASS（报告: {report.report_json}）")
    return EXIT_OK


@app.command("list-models")
def command_list_models() -> None:
    """列出已注册的生成模型（按模式分组，标注阶段与依赖状态）。"""

    rows = registry_report()
    typer.echo(f"{'name':<22}{'mode':<12}{'phase':<7}{'deps':<10}status")
    typer.echo("-" * 96)
    for row in rows:
        deps = "ok" if row["dependencies_available"] else "missing"
        typer.echo(f"{row['name']:<22}{row['mode']:<12}{row['phase']:<7}{deps:<10}{row['status']}")
    typer.echo("")
    typer.echo(
        "说明：Phase 2 statistical 三模型（SDV level_wise）与 Phase 3 physics 求解器已实现；"
        "hybrid（Phase 4）仍为 NOT_RUN。"
    )


@app.command("generate")
def command_generate(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False, readable=True, help="生成配置文件"),
    overwrite: bool = typer.Option(False, "--overwrite", help="覆盖已存在的 data.nc 输出"),
) -> None:
    """按配置生成合成数据（Mode 1 statistical；物理/混合在 Phase 3/4）。"""

    code = run_generate_command(config_path=config, overwrite=overwrite)
    raise typer.Exit(code=code)


def run_generate_command(*, config_path: Path, overwrite: bool = False) -> int:
    """Shared by the CLI and scripts/run_experiments.py."""

    try:
        parsed = load_config_file(config_path)
    except ConfigError as exc:
        _echo_error(str(exc))
        return EXIT_USER_ERROR
    if isinstance(parsed, IngestConfigFile):
        _echo_error(f"{config_path} 是 ingest 配置，请用 `atmosgen ingest --config {config_path}`")
        return EXIT_USER_ERROR
    try:
        outcome = run_generation(parsed, config_path=config_path, overwrite=overwrite)
    except NotImplementedInPhase as exc:
        typer.echo(f"NOT_RUN: {exc}")
        return EXIT_NOT_IMPLEMENTED
    except (
        GenerationError,
        SubsetError,
        ContractError,
        FileExistsError,
        FileNotFoundError,
        KeyError,
        ValueError,
        OSError,
    ) as exc:
        _echo_error(f"generate 失败: {type(exc).__name__}: {exc}")
        return EXIT_RUNTIME_ERROR
    except Exception as exc:  # noqa: BLE001 - keep the exit-code contract
        _echo_error(f"generate 运行失败（未预期异常）: {type(exc).__name__}: {exc}")
        return EXIT_RUNTIME_ERROR

    meta = outcome.metadata
    fit = meta.get("fit_resources") or {}
    sample = meta.get("resources") or {}
    samples = (outcome.contract or {}).get("generation_config", {}).get("samples", {})
    subset = meta.get("effective_grid") or meta.get("subset") or {}
    typer.echo(f"generate 完成: mode={meta.get('mode')} model={meta.get('model')} seed={meta.get('seed')}")
    typer.echo(f"  run 目录:    {outcome.run_dir}")
    typer.echo(f"  输出数据:    {outcome.data_path}")
    if subset:
        time_info = subset.get("time") or {}
        typer.echo(
            f"  训练子集:    {time_info.get('n', '?')} 时次 × "
            f"{(subset.get('level') or {}).get('n', 1)} 层 × "
            f"{(subset.get('latitude') or {}).get('n', '?')}×{(subset.get('longitude') or {}).get('n', '?')} 格点"
        )
    if samples:
        if "rows_generated" in samples:
            typer.echo(
                f"  样本:        unit={samples.get('unit')} rows_generated={samples.get('rows_generated')} "
                f"(rows_requested={samples.get('rows_requested')})"
            )
        else:
            typer.echo(
                f"  场数:        unit={samples.get('unit')} fields={samples.get('fields')}"
            )
    diagnostics = meta.get("diagnostics") or {}
    if diagnostics:
        typer.echo(
            f"  求解器:      CFL_x={diagnostics.get('cfl_x'):.3g} "
            f"CFL_y={diagnostics.get('cfl_y'):.3g} "
            f"扩散数={diagnostics.get('diffusion_number'):.3g} "
            f"质量漂移={diagnostics.get('mass_drift_max'):.3e}"
        )
    if fit:
        typer.echo(f"  训练耗时:    {fit.get('elapsed_seconds')} s（fit 峰值 RSS {fit.get('peak_rss_mb')} MB）")
    if sample:
        typer.echo(f"  采样耗时:    {sample.get('elapsed_seconds')} s（sample 峰值 RSS {sample.get('peak_rss_mb')} MB）")
    if meta.get("ignored_hyperparameters"):
        typer.echo(
            f"  忽略超参:    {meta['ignored_hyperparameters']}（当前 sdv 版本不支持，已记录不失败）"
        )
    typer.echo(
        f"下一步：atmosgen evaluate --input {outcome.data_path} --real <真实数据集>"
    )
    return EXIT_OK


@app.command("evaluate")
def command_evaluate(
    input: Path = typer.Option(..., "--input", exists=True, dir_okay=False, readable=True, help="待评价数据集（data.nc）"),
    real: Optional[Path] = typer.Option(None, "--real", help="真实参考数据集；不给则整体 NOT_RUN"),
    out: Optional[Path] = typer.Option(None, "--out", help="报告输出目录（默认与 --input 同目录）"),
    config: Optional[Path] = typer.Option(None, "--config", help="generation 配置文件（提供 thresholds_overrides）"),
    overwrite: bool = typer.Option(False, "--overwrite", help="覆盖已存在的 quality_report.json"),
    fail_on: str = typer.Option(
        "fail",
        "--fail-on",
        help="退出码策略：fail=质量 FAIL 时退出 4；never=只写报告总是退出 0",
    ),
) -> None:
    """质量评价（statistical / temporal / spatial；physical 在 Phase 3）。"""

    config_model: GenerationConfigFile | None = None
    if config is not None:
        try:
            parsed = load_config_file(config)
        except ConfigError as exc:
            _echo_error(str(exc))
            raise typer.Exit(code=EXIT_USER_ERROR)
        if isinstance(parsed, IngestConfigFile):
            _echo_error(f"--config {config} 是 ingest 配置，不是 generation 配置")
            raise typer.Exit(code=EXIT_USER_ERROR)
        config_model = parsed  # type: ignore[assignment]
    try:
        quality = run_evaluation(
            input,
            real_path=real,
            config=config_model,
            output_dir=out,
            overwrite=overwrite,
        )
    except FileExistsError as exc:
        _echo_error(f"{exc}（或传 --overwrite 重新生成）")
        raise typer.Exit(code=EXIT_USER_ERROR)
    except (EvaluationError, SubsetError, ContractError, KeyError, ValueError, OSError) as exc:
        _echo_error(f"evaluate 失败: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=EXIT_RUNTIME_ERROR)
    except Exception as exc:  # noqa: BLE001 - keep the exit-code contract
        _echo_error(f"evaluate 运行失败（未预期异常）: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=EXIT_RUNTIME_ERROR)
    typer.echo(f"overall_status: {quality['overall_status']}")
    for name in ("statistical", "temporal", "spatial", "physical"):
        block = quality.get(name) or {}
        typer.echo(f"  {name:<12} {block.get('status', 'MISSING')}")
    for path_key in ("json", "md"):
        typer.echo(f"  {path_key}: {quality['report_paths'][path_key]}")
    if quality.get("figures"):
        typer.echo(f"  figures:     {len(quality['figures'])} 个（{quality['figures'][0]} …）")
    if quality["overall_status"] != "PASS":
        typer.echo("说明：状态与阈值明细见 quality_report.json / quality_report.md")
    if fail_on not in {"fail", "never"}:
        _echo_error(f"--fail-on 必须是 fail|never，得到 {fail_on!r}")
        raise typer.Exit(code=EXIT_USER_ERROR)
    if fail_on == "fail" and quality["overall_status"] == "FAIL":
        typer.echo("exit 4: 质量状态 FAIL（--fail-on never 可改为始终退出 0）", err=True)
        raise typer.Exit(code=EXIT_RUNTIME_ERROR)


@app.command("report")
def command_report(
    input: Path = typer.Option(..., "--input", help="quality_report(.json) 文件或其所在 run 目录"),
    format: str = typer.Option("md", "--format", help="md | json"),
    out: Optional[Path] = typer.Option(None, "--out", help="Markdown 输出路径（默认打印到 stdout）"),
) -> None:
    """把 quality_report.json 渲染为 Markdown（或原样输出 JSON）。"""

    try:
        quality = read_quality_report(input)
    except EvaluationError as exc:
        _echo_error(str(exc))
        raise typer.Exit(code=EXIT_USER_ERROR)
    if format == "json":
        typer.echo(json.dumps(quality, ensure_ascii=False, indent=2, default=str))
        return
    if format != "md":
        _echo_error(f"format 必须是 md|json，得到 {format!r}")
        raise typer.Exit(code=EXIT_USER_ERROR)
    from .validation.report import render_markdown

    markdown = render_markdown(quality)
    if out is not None:
        Path(out).write_text(markdown, encoding="utf-8")
        typer.echo(f"Markdown 已写入: {out}")
        typer.echo(markdown)
    else:
        typer.echo(markdown)

if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
