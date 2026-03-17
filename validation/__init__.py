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
from .oracle_alpha_controls import (
    BootstrapMeanInterval,
    MIBPlan,
    OracleAlphaControlPlan,
    OracleAlphaControlRegistry,
    PredictivenessPlan,
    PredictivenessSummary,
    StabilitySuitePlan,
    bootstrap_mean_confidence_interval,
    linear_alpha_predictiveness_summary,
    load_oracle_alpha_control_registry,
    mean_pairwise_js_divergence,
    mean_topk_jaccard_similarity,
)

__all__ = [
    "BootstrapMeanInterval",
    "CacheReconstructionMetrics",
    "LayerResidualMetrics",
    "MIBPlan",
    "ModelBackedReconstructionMetrics",
    "OracleAlphaControlPlan",
    "OracleAlphaControlRegistry",
    "PredictivenessPlan",
    "PredictivenessSummary",
    "StabilitySuitePlan",
    "bootstrap_mean_confidence_interval",
    "cache_reconstruction_metrics",
    "linear_alpha_predictiveness_summary",
    "load_oracle_alpha_control_registry",
    "max_abs_difference",
    "mean_pairwise_js_divergence",
    "mean_topk_jaccard_similarity",
    "model_backed_reconstruction_metrics",
    "rms_norm",
    "routed_logits",
    "shared_final_norm_logit_contributions",
]
