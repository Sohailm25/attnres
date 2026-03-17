# ABOUTME: Exposes Phase 1 validation utilities for routed residual reconstruction checks.
# ABOUTME: Keeps normalization and cache-validity logic centralized before oracle-alpha runs.

from .reconstruction import (
    CacheReconstructionMetrics,
    cache_reconstruction_metrics,
    max_abs_difference,
    rms_norm,
    routed_logits,
    shared_final_norm_logit_contributions,
)
from .model_backed import (
    LayerResidualMetrics,
    ModelBackedReconstructionMetrics,
    model_backed_reconstruction_metrics,
)

__all__ = [
    "CacheReconstructionMetrics",
    "LayerResidualMetrics",
    "ModelBackedReconstructionMetrics",
    "cache_reconstruction_metrics",
    "max_abs_difference",
    "model_backed_reconstruction_metrics",
    "rms_norm",
    "routed_logits",
    "shared_final_norm_logit_contributions",
]
