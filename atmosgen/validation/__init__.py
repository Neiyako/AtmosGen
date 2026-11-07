"""AtmosGen 验证层（实现计划 §2.1 验证层 / §5 质量验证规格）。

公开 API（供 generate/evaluate CLI 直接 import，不要改名）：

- :func:`atmosgen.validation.statistical.run_statistical_validation`
- :func:`atmosgen.validation.temporal.run_temporal_validation`
- :func:`atmosgen.validation.spatial.run_spatial_validation`
- :func:`atmosgen.validation.report.aggregate_status`
- :func:`atmosgen.validation.report.render_markdown`
- :func:`atmosgen.validation.temporal.timeseries_efficacy`（可选启用，默认 NOT_RUN）

统一返回结构（所有 run_*）::

    {
      "metrics": {<metric_key>: {"value": float, "threshold": float|None,
                                 "comparison": "max"|"min"|"range"|"info",
                                 "status": "PASS"|"FAIL"|"CONDITIONAL_PASS"
                                           |"NOT_RUN"|"INFO",
                                 "detail": str}},
      "status": <aggregate over metrics via report.aggregate_status>,
      "notes": [str, ...],
    }
"""

from .physical import run_physical_validation
from .report import aggregate_status, render_markdown
from .spatial import run_spatial_validation
from .statistical import run_statistical_validation
from .temporal import run_temporal_validation, timeseries_efficacy

__all__ = [
    "aggregate_status",
    "timeseries_efficacy",
    "render_markdown",
    "run_physical_validation",
    "run_spatial_validation",
    "run_statistical_validation",
    "run_temporal_validation",
]
