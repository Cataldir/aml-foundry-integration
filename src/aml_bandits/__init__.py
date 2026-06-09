"""
AML Bandits: Multi-Armed Bandits for Adaptive Experimentation on Azure ML
"""

__version__ = "1.0.0"
__author__ = "Ricardo Cataldi"

from aml_bandits.data_loader import load_bank_marketing_dataset
from aml_bandits.preprocessing import preprocess_data
from aml_bandits.bandits import (
    DeterministicBaselinePolicy,
    UCBPolicy,
    ThompsonSamplingPolicy,
)
from aml_bandits.simulator import BanditSimulator, build_bandit_environment
from aml_bandits.metrics import compute_metrics
from aml_bandits.foundry_bridge import (
    AgentResponse,
    AggregateMetrics,
    EnrichedRecommendation,
    FeatureDiscoveryResponse,
    apply_safe_ucb_alpha,
    build_aggregate_metrics,
    build_feature_discovery_payload,
    send_aggregate_to_agent,
    send_feature_profile_to_agent,
    validate_and_enrich_strategy,
    validate_and_propose_features,
)

__all__ = [
    "load_bank_marketing_dataset",
    "preprocess_data",
    "DeterministicBaselinePolicy",
    "UCBPolicy",
    "ThompsonSamplingPolicy",
    "BanditSimulator",
    "build_bandit_environment",
    "compute_metrics",
    "AgentResponse",
    "AggregateMetrics",
    "EnrichedRecommendation",
    "FeatureDiscoveryResponse",
    "apply_safe_ucb_alpha",
    "build_aggregate_metrics",
    "build_feature_discovery_payload",
    "send_aggregate_to_agent",
    "send_feature_profile_to_agent",
    "validate_and_enrich_strategy",
    "validate_and_propose_features",
]
