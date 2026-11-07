"""Typer CLI for AtmosGen (plan §6.2).

Exit codes
----------
0  success
2  invalid configuration / user error
3  command not implemented in the current phase (NOT_RUN, never silently skipped)
4  runtime failure (ingest, IO, contract)
"""

from __future__ import annotations

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

    check = validate_config_file(config, check_data=check_data)
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
    except (IngestError, ContractError, FileExistsError, FileNotFoundError) as exc:
        _echo_error(f"ingest 失败: {exc}")
        return EXIT_RUNTIME_ERROR
    typer.echo(report.render_text())
    if report.status != "PASS":
        _echo_error(f"ingest 未通过（status={report.status}），详见 {report.report_json}")
        return EXIT_RUNTIME_ERROR
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
    typer.echo("说明：Phase 1 只注册模型与接口；适配器在 Phase 2（SDV）/ Phase 3（物理）/ Phase 4（混合）实现。")


@app.command("generate")
def command_generate(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False, readable=True, help="生成配置文件"),
    overwrite: bool = typer.Option(False, "--overwrite", help="覆盖已存在的输出目录"),
) -> None:
    """生成数据（Phase 1 尚未实现：先校验配置，再明确报 NOT_RUN）。"""

    del overwrite
    try:
        parsed = load_config_file(config)
    except ConfigError as exc:
        _echo_error(str(exc))
        raise typer.Exit(code=EXIT_USER_ERROR)
    if isinstance(parsed, IngestConfigFile):
        _echo_error(f"{config} 是 ingest 配置，请用 `atmosgen ingest --config {config}`")
        raise typer.Exit(code=EXIT_USER_ERROR)
    assert isinstance(parsed, GenerationConfigFile)
    typer.echo(
        f"配置合法：mode={parsed.generation.mode} model={parsed.generation.model} "
        f"（Phase 1 仅完成骨架）"
    )
    typer.echo(
        "NOT_RUN: generate 在 Phase 2（statistical）/ Phase 3（physics）/ Phase 4（hybrid）实现；"
        "当前不生成数据，也不写占位文件"
    )
    raise typer.Exit(code=EXIT_NOT_IMPLEMENTED)


@app.command("evaluate")
def command_evaluate(
    input: Optional[Path] = typer.Option(None, "--input", help="待评价数据集（data.nc）"),
    real: Optional[Path] = typer.Option(None, "--real", help="真实参考数据集"),
    out: Optional[Path] = typer.Option(None, "--out", help="quality_report.json 输出路径"),
) -> None:
    """质量评价（Phase 2 起实现；当前 NOT_RUN）。"""

    del input, real, out
    typer.echo("NOT_RUN: evaluate 在 Phase 2 实现（statistical/temporal/spatial/physical 四类指标）")
    raise typer.Exit(code=EXIT_NOT_IMPLEMENTED)


@app.command("report")
def command_report(
    input: Optional[Path] = typer.Option(None, "--input", help="数据集路径"),
    format: str = typer.Option("md", "--format", help="md | json"),
    out: Optional[Path] = typer.Option(None, "--out", help="报告输出路径"),
) -> None:
    """生成质量报告（Phase 2 起实现；当前 NOT_RUN）。"""

    del input, format, out
    typer.echo("NOT_RUN: report 在 Phase 2 实现（quality_report.{json,md} + figures）")
    raise typer.Exit(code=EXIT_NOT_IMPLEMENTED)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
