"""Generator adapters (SDV / Synthcity / physics / hybrid).

Phase 1 only provides the registry and :class:`atmosgen.generators.base.BaseGenerator`;
adapters are added in Phases 2-4 as described in docs/IMPLEMENTATION_PLAN.md.
"""

from .base import BaseGenerator, GenerationResult
from .registry import MODELS, ModelSpec, get_model, list_models, model_names, registry_report

__all__ = [
    "MODELS",
    "BaseGenerator",
    "GenerationResult",
    "ModelSpec",
    "get_model",
    "list_models",
    "model_names",
    "registry_report",
]
