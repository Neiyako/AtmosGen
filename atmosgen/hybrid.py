"""Hybrid generation (Mode 3, plan §4.3): physics base + conditional SDV + screening.

Pipeline
--------
1. **Physics base** — the advection–diffusion solver produces a tracer field that
   starts from a real field (``from_real``) and carries physical structure.
2. **Conditional generation** — an SDV single-table model fitted on the real
   data generates the *remaining* variables conditioned on the physics field
   (``sample_remaining_columns``), so the physics base is preserved exactly
   (verified: known columns come back bit-identical and in order).
3. **Physical screening** — every sample is checked against the hard
   constraints of plan §5.4 (finite, non-negative, ranges, saturation).
   Rejected samples are written to ``rejected.jsonl`` with reasons and
   metrics; the rejection rate lands in the contract. Nothing is silently
   dropped, and "before/after" screening metrics are reported separately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .contract import build_contract
from .data.tabular import SUPPORTED_COVARIATES
from .runner import ResourceTracker, set_global_seed
from .validation.physical import DEFAULT_THRESHOLDS

#: Hard constraints applied by the hybrid screening step (plan §5.4).
SCREENING_RULES = (
    "finite",
    "q_nonneg",
    "cloud_nonneg",
    "t_range",
    "wind_range",
    "saturation",
    "tcwv_range",
)

#: Saturation factor (q <= factor * q_sat); plan §5.4 default 1.10.
SATURATION_FACTOR = 1.10


class HybridError(RuntimeError):
    """Raised when the hybrid pipeline cannot run."""


@dataclass
class ScreeningResult:
    """Outcome of the physical screening step."""

    accepted: np.ndarray
    reasons: list[str] = field(default_factory=list)
    per_sample: list[dict[str, Any]] = field(default_factory=list)
    rejection_rate: float = 0.0
    rejected_count: int = 0
    total: int = 0

    def summary(self, *, mask_applied: bool = False) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for reason in self.reasons:
            counts[reason] = counts.get(reason, 0) + 1
        return {
            "total_samples": self.total,
            "rejected_samples": self.rejected_count,
            "accepted_samples": int(self.accepted.sum()),
            "rejection_rate": self.rejection_rate,
            "reason_counts": counts,
            "rules": list(SCREENING_RULES),
            "mask_applied": bool(mask_applied),
        }


def q_sat_from_t_and_p(t_kelvin: np.ndarray, p_hpa: np.ndarray) -> np.ndarray:
    """Tetens (1990) saturation mixing ratio over water (kg/kg).

    ``q_sat = 0.622 * e_s / (p - 0.378 * e_s)`` with ``e_s`` in hPa.
    """

    e_s = 6.112 * np.exp(17.67 * (t_kelvin - 273.15) / (t_kelvin - 29.65))
    denominator = p_hpa - 0.378 * e_s
    with np.errstate(divide="ignore", invalid="ignore"):
        q_sat = 0.622 * e_s / denominator
    q_sat = np.where(np.isfinite(q_sat) & (q_sat > 0), q_sat, np.nan)
    return q_sat


def screen_samples(
    frame: pd.DataFrame,
    *,
    thresholds: Mapping[str, float] | None = None,
    pressure_hpa: np.ndarray | None = None,
) -> ScreeningResult:
    """Apply the hard physical constraints row-wise to a sample table.

    ``frame`` must carry the canonical columns (``t``, ``q``, ``u``, ``v``,
    ``tcwv`` and optionally the cloud-water variables). ``pressure_hpa`` gives
    the level pressure of each row (needed for the saturation check).
    """

    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update({k: float(v) for k, v in thresholds.items() if k in th})
    total = int(len(frame))
    accepted = np.ones(total, dtype=bool)
    reasons: list[str] = []
    per_sample: list[dict[str, Any]] = []

    def _violations(mask: np.ndarray, label: str) -> np.ndarray:
        nonlocal accepted
        bad = np.asarray(mask, dtype=bool)
        if not bad.any():
            return bad
        accepted &= ~bad
        for index in np.where(bad)[0]:
            reasons.append(label)
            per_sample.append({"row": int(index), "reason": label})
        return bad

    numeric = {name: np.asarray(frame[name].values, dtype=float) for name in frame.columns}

    # 1) finite
    for name, values in numeric.items():
        _violations(~np.isfinite(values), f"non_finite:{name}")

    # 2) non-negativity
    if "q" in numeric:
        _violations(numeric["q"] < 0, "q_negative")
    for name in ("clwc", "ciwc", "crwc", "cswc"):
        if name in numeric:
            _violations(numeric[name] < 0, f"{name}_negative")

    # 3) ranges
    if "t" in numeric:
        _violations(
            (numeric["t"] < th["t_min"]) | (numeric["t"] > th["t_max"]),
            "t_out_of_range",
        )
    if "u" in numeric:
        _violations(np.abs(numeric["u"]) > th["wind_max"], "u_out_of_range")
    if "v" in numeric:
        _violations(np.abs(numeric["v"]) > th["wind_max"], "v_out_of_range")
    if "tcwv" in numeric:
        _violations(
            (numeric["tcwv"] < 0) | (numeric["tcwv"] > th["tcwv_max"]),
            "tcwv_out_of_range",
        )

    # 4) saturation (needs pressure)
    if "t" in numeric and "q" in numeric and pressure_hpa is not None:
        q_sat = q_sat_from_t_and_p(numeric["t"], np.asarray(pressure_hpa, dtype=float))
        with np.errstate(invalid="ignore"):
            violating = np.isfinite(q_sat) & (numeric["q"] > SATURATION_FACTOR * q_sat)
        _violations(violating, "supersaturated")

    rejected = int((~accepted).sum())
    result = ScreeningResult(
        accepted=accepted,
        reasons=reasons,
        per_sample=per_sample,
        rejection_rate=(rejected / total) if total else 0.0,
        rejected_count=rejected,
        total=total,
    )
    return result


def write_rejected_jsonl(
    path: str | Path,
    screening: ScreeningResult,
    *,
    coordinates: Mapping[str, Any] | None = None,
) -> Path:
    """Write one JSON line per rejected sample (reasons + metrics)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[int, list[str]] = {}
    for entry in screening.per_sample:
        grouped.setdefault(int(entry["row"]), []).append(str(entry["reason"]))
    with target.open("w", encoding="utf-8") as handle:
        for row, reasons in sorted(grouped.items()):
            payload: dict[str, Any] = {"row": row, "reasons": reasons}
            if coordinates is not None:
                for key, index in coordinates.items():
                    values = coordinates.get(key)
                    if values is not None and index < len(values):
                        payload[key] = values[index]
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return target


def conditional_generate(
    synthesizer: Any,
    *,
    known_column: str,
    known_values: Sequence[float],
    batch_size: int = 4096,
) -> pd.DataFrame:
    """Sample the remaining variables conditioned on ``known_column``.

    SDV returns the known column unchanged (bit-identical, same row order);
    that invariant is asserted here because the hybrid pipeline relies on the
    physics base surviving the statistical step.
    """

    known = pd.DataFrame({known_column: np.asarray(known_values, dtype=float)})
    if len(known) == 0:
        raise HybridError("条件生成需要至少一个已知值")
    frame = synthesizer.sample_remaining_columns(known, batch_size=min(batch_size, len(known)))
    if isinstance(frame, Mapping):
        frame = frame[next(iter(frame))]
    if not isinstance(frame, pd.DataFrame):
        raise HybridError(f"条件采样返回 {type(frame).__name__}，期望 pandas.DataFrame")
    if known_column not in frame.columns:
        raise HybridError(f"条件采样结果缺少 known 列 {known_column!r}")
    difference = np.abs(
        np.asarray(frame[known_column].values, dtype=float)
        - np.asarray(known[known_column].values, dtype=float)
    )
    if difference.size and float(np.nanmax(difference)) > 0:
        raise HybridError(
            f"条件采样未原样保留 known 列 {known_column!r}"
            f"（最大偏差 {float(np.nanmax(difference)):.3g}）——混合模式不得改写物理基底"
        )
    return frame


__all__ = [
    "HybridError",
    "SATURATION_FACTOR",
    "SCREENING_RULES",
    "ScreeningResult",
    "conditional_generate",
    "q_sat_from_t_and_p",
    "screen_samples",
    "write_rejected_jsonl",
]
