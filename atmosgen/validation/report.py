"""验证判定聚合与 Markdown 报告渲染（实现计划 §5.5）。

纯字符串拼接，无外部依赖。
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["aggregate_status", "render_markdown"]

# 汇总优先级：FAIL 为主导；NOT_RUN 不能被 PASS 掩盖；CONDITIONAL_PASS 优先于 PASS；
# INFO 只记录数值、不参与判定；未知状态按最保守处理（当作 FAIL，防止静默放行）。
_RANK = {
    "FAIL": 3,
    "NOT_RUN": 2,
    "CONDITIONAL_PASS": 1,
    "INFO": 0,
    "PASS": 0,
}


def aggregate_status(states: Iterable[str]) -> str:
    """聚合一组指标/模块状态。

    规则（实现计划 §5.5）：
    - 有 FAIL → ``"FAIL"``；
    - 无 FAIL 但有 NOT_RUN → ``"NOT_RUN"``；
    - 无上两者但有 CONDITIONAL_PASS → ``"CONDITIONAL_PASS"``；
    - 其余（全 PASS/INFO）→ ``"PASS"``；
    - 空输入 → ``"NOT_RUN"``（没有任何评估证据，不得默认 PASS）。

    未知状态按 FAIL 处理（保守，显式人工介入后应修正状态来源）。
    """

    worst_rank = -1
    worst_state: str | None = None
    for raw in states:
        state = "FAIL" if raw is None else str(raw)
        if state not in _RANK:
            state = "FAIL"  # 未知状态 → 保守按 FAIL
        if state == "INFO":
            continue
        rank = _RANK[state]
        if rank > worst_rank:
            worst_rank = rank
            worst_state = state
    if worst_state is None:
        return "NOT_RUN"
    return worst_state


def _fmt(value: object) -> str:
    """把 metric value 格式化成短字符串（None/nan → 'n/a'）。"""
    try:
        fv = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if fv != fv:  # NaN
        return "n/a"
    return f"{fv:.4g}"


_SECTION_TITLES = {
    "statistical": "统计分布一致性（statistical）",
    "temporal": "时间连续性（temporal）",
    "spatial": "空间连续性（spatial）",
    "physical": "物理合理性（physical）",
}


def render_markdown(report: dict) -> str:
    """把 ``{"statistical":..., "temporal":..., "spatial":..., "meta":...}``
    渲染为 Markdown。

    每个子块一张表：`metric | value | threshold | status | detail`；
    NOT_RUN 的指标保留 detail 里的原因，不在显著位置丢失。
    """

    lines: list[str] = ["# AtmosGen 质量验证报告", ""]
    meta = report.get("meta") or {}
    for key in ("title", "model_name", "model_version", "random_seed", "generated_at"):
        if key in meta:
            prefix = key.replace("_", " ").capitalize()
            lines.append(f"- **{prefix}**: {meta[key]}")

    section_names = [s for s in ("statistical", "temporal", "spatial", "physical") if s in report]
    overall = aggregate_status(
        report[s]["status"] for s in section_names if isinstance(report[s], dict)
    )
    lines.append("")
    lines.append(f"**Overall status: {overall}**")
    for s in section_names:
        block = report[s]
        if not isinstance(block, dict):
            continue
        lines.append("")
        lines.append(f"## {_SECTION_TITLES.get(s, s)}")
        lines.append(f"**Section status: {block.get('status', 'NOT_RUN')}**")
        lines.append("")
        lines.append("| metric | value | threshold | comparison | status | detail |")
        lines.append("|---|---|---|---|---|---|")
        metrics = block.get("metrics") or {}
        if not metrics:
            lines.append("| (none) | - | - | - | NOT_RUN | no metrics emitted |")
        for name, m in metrics.items():
            threshold = m.get("threshold")
            thr_txt = "-" if threshold is None else _fmt(threshold)
            detail = (m.get("detail") or "").replace("|", "/")
            lines.append(
                f"| {name} | {_fmt(m.get('value'))} | {thr_txt} "
                f"| {m.get('comparison', '-')} | {m.get('status', 'NOT_RUN')} "
                f"| {detail} |"
            )
        notes = block.get("notes") or []
        if notes:
            lines.append("")
            for note in notes:
                lines.append(f"> note: {note}")
    lines.append("")
    return "\n".join(lines)
