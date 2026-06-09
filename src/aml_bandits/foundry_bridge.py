"""
Aggregate-only Azure AI Foundry bridge for bandit strategy validation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from aml_bandits.settings import resolve_foundry_agent

try:  # pragma: no cover - optional Azure dependency
    from azure.core.exceptions import AzureError  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional Azure dependency
    class AzureError(ValueError):  # type: ignore[no-redef]
        """Fallback AzureError type used when azure-core is unavailable."""

DEFAULT_UCB_ALPHA_BOUNDS = (0.1, 2.0)
DEFAULT_UCB_ALPHA = 1.2


@dataclass(frozen=True)
class AggregateMetrics:
    """Sanitized aggregate evidence sent to the agent bridge."""

    policy_summaries: list[dict[str, Any]]
    candidate_policy: str = "UCB"
    current_ucb_alpha: float = DEFAULT_UCB_ALPHA
    ucb_alpha_bounds: tuple[float, float] = DEFAULT_UCB_ALPHA_BOUNDS
    evidence_scope: str = "aggregate_only"
    n_rounds: int | None = None


@dataclass(frozen=True)
class AgentResponse:
    """Foundry or fallback response normalized into a local contract."""

    source: str
    validation: str
    recommended_ucb_alpha: float | None
    rationale: str
    warnings: list[str] = field(default_factory=list)
    raw_response: str | None = None


@dataclass(frozen=True)
class EnrichedRecommendation:
    """Safe recommendation that can be applied to a UCB candidate."""

    accepted: bool
    recommended_ucb_alpha: float | None
    bounded_ucb_alpha: float | None
    validation: str
    rationale: str
    source: str
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FeatureDiscoveryResponse:
    """Agent response for aggregate-only feature discovery."""

    source: str
    validation: str
    accepted_features: list[str]
    rejected_features: list[dict[str, Any]]
    proposed_features: list[dict[str, Any]]
    feature_conditions: list[dict[str, Any]]
    risk_flags: list[str]
    rationale: str
    warnings: list[str] = field(default_factory=list)
    raw_response: str | None = None


def build_aggregate_metrics(
    summary: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    current_ucb_alpha: float = DEFAULT_UCB_ALPHA,
    bounds: tuple[float, float] = DEFAULT_UCB_ALPHA_BOUNDS,
) -> AggregateMetrics:
    """Build aggregate-only evidence from a summary table."""
    if isinstance(summary, pd.DataFrame):
        records = summary.to_dict(orient="records")
    else:
        records = [dict(row) for row in summary]

    sanitized = [_sanitize_summary_row(row) for row in records]
    n_rounds = _infer_n_rounds(sanitized)
    return AggregateMetrics(
        policy_summaries=sanitized,
        current_ucb_alpha=float(current_ucb_alpha),
        ucb_alpha_bounds=bounds,
        n_rounds=n_rounds,
    )


def send_aggregate_to_agent(aggregate: AggregateMetrics | Mapping[str, Any]) -> AgentResponse:
    """Send aggregate evidence to Foundry when configured, otherwise use fallback."""
    evidence = _normalize_aggregate(aggregate)
    endpoint, agent_ref = resolve_foundry_agent("strategy")

    if not endpoint or not agent_ref:
        return _local_fallback_response(
            evidence,
            ["Foundry endpoint or agent name/id is not configured."],
        )

    try:
        return _send_with_azure_ai_projects(evidence, endpoint, agent_ref)
    except (AttributeError, ImportError, TypeError, ValueError, json.JSONDecodeError, AzureError) as ex:  # pragma: no cover - depends on optional SDK and Azure auth
        return _local_fallback_response(evidence, [f"Foundry SDK path unavailable: {ex}"])


def validate_and_enrich_strategy(
    aggregate: AggregateMetrics | Mapping[str, Any],
) -> EnrichedRecommendation:
    """Validate aggregate evidence and return a bounded UCB alpha recommendation."""
    evidence = _normalize_aggregate(aggregate)
    response = send_aggregate_to_agent(evidence)
    bounds = tuple(evidence.get("ucb_alpha_bounds", DEFAULT_UCB_ALPHA_BOUNDS))
    bounded_alpha = _bound_alpha(response.recommended_ucb_alpha, bounds)

    warnings = list(response.warnings)
    accepted = bounded_alpha is not None and response.validation.lower() in {"valid", "safe"}
    if response.recommended_ucb_alpha is not None and bounded_alpha != response.recommended_ucb_alpha:
        warnings.append(
            f"Recommended alpha was clipped to safe bounds [{bounds[0]}, {bounds[1]}]."
        )

    return EnrichedRecommendation(
        accepted=accepted,
        recommended_ucb_alpha=response.recommended_ucb_alpha,
        bounded_ucb_alpha=bounded_alpha,
        validation=response.validation,
        rationale=response.rationale,
        source=response.source,
        warnings=warnings,
    )


def apply_safe_ucb_alpha(
    strategy: Any,
    recommended_alpha: float | None,
    bounds: tuple[float, float] = DEFAULT_UCB_ALPHA_BOUNDS,
) -> Any:
    """Create a new UCB-like strategy with the recommended alpha safely bounded."""
    safe_alpha = _bound_alpha(recommended_alpha, bounds)
    if safe_alpha is None:
        raise ValueError("A numeric UCB alpha recommendation is required.")

    if not hasattr(strategy, "n_arms") or not hasattr(strategy, "margins"):
        raise TypeError("Strategy must expose n_arms and margins attributes.")

    return strategy.__class__(
        n_arms=int(strategy.n_arms),
        margins_arr=np.asarray(strategy.margins, dtype=float),
        alpha=safe_alpha,
    )


def to_agent_payload(aggregate: AggregateMetrics | Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact sanitized payload that may be sent to an agent."""
    evidence = _normalize_aggregate(aggregate)
    return {
        "task": "validate_and_enrich_multi_armed_bandit_strategy",
        "privacy": "aggregate_evidence_only_no_raw_customer_rows",
        "allowed_recommendations": ["bounded_ucb_alpha"],
        "evidence": evidence,
    }


def build_feature_discovery_payload(feature_profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the aggregate-only feature evidence payload for agent review."""
    profile = dict(feature_profile)
    return {
        "task": "evaluate_feature_conditions_and_propose_features",
        "privacy": "aggregate_evidence_only_no_raw_customer_rows",
        "dataset_context": {
            "row_count": _json_safe_value(profile.get("row_count")),
            "feature_count": _json_safe_value(profile.get("feature_count")),
            "target_name": profile.get("target_name"),
            "excluded_columns": list(profile.get("excluded_columns", [])),
        },
        "feature_evidence": list(profile.get("features", [])),
        "constraints": {
            "forbidden_raw_fields": ["customer_id", "account_id", "raw_customer_rows"],
            "leakage_columns_must_remain_excluded": ["duration"],
            "recommendation_scope": [
                "accepted_features",
                "rejected_features",
                "proposed_features",
                "feature_conditions",
            ],
        },
    }


def send_feature_profile_to_agent(feature_profile: Mapping[str, Any]) -> FeatureDiscoveryResponse:
    """Send aggregate feature evidence to Foundry when configured, otherwise use fallback."""
    payload = build_feature_discovery_payload(feature_profile)
    endpoint, agent_ref = resolve_foundry_agent("feature")

    if not endpoint or not agent_ref:
        return _local_feature_discovery_response(
            payload,
            ["Foundry endpoint or feature agent name/id is not configured."],
        )

    try:
        return _send_feature_discovery_with_azure_ai_projects(payload, endpoint, agent_ref)
    except (AttributeError, ImportError, TypeError, ValueError, json.JSONDecodeError, AzureError) as ex:  # pragma: no cover - depends on optional SDK and Azure auth
        return _local_feature_discovery_response(payload, [f"Foundry SDK path unavailable: {ex}"])


def validate_and_propose_features(feature_profile: Mapping[str, Any]) -> FeatureDiscoveryResponse:
    """Validate feature evidence and propose safe derived features."""
    return send_feature_profile_to_agent(feature_profile)


def _send_with_azure_ai_projects(evidence: dict[str, Any], endpoint: str, agent_ref: str) -> AgentResponse:
    from azure.ai.projects import AIProjectClient  # type: ignore[import-not-found]
    from azure.identity import DefaultAzureCredential  # type: ignore[import-not-found]

    client = AIProjectClient(endpoint=endpoint, credential=DefaultAzureCredential())
    agent = _resolve_agent(client, agent_ref)
    prompt = _build_agent_prompt(evidence)

    thread = client.agents.threads.create()
    client.agents.messages.create(thread_id=thread.id, role="user", content=prompt)
    run = client.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
    if getattr(run, "status", None) == "failed":
        raise ValueError(f"Foundry agent run failed: {getattr(run, 'last_error', None)}")

    messages = client.agents.messages.list(thread_id=thread.id)
    content = _extract_latest_text(messages)
    parsed = _parse_agent_json(content)
    return AgentResponse(
        source="azure_ai_foundry",
        validation=str(parsed.get("validation", "valid")),
        recommended_ucb_alpha=_coerce_optional_float(parsed.get("recommended_ucb_alpha")),
        rationale=str(parsed.get("rationale", "Foundry agent returned a normalized response.")),
        warnings=list(parsed.get("warnings", [])),
        raw_response=content,
    )


def _send_feature_discovery_with_azure_ai_projects(
    payload: dict[str, Any],
    endpoint: str,
    agent_ref: str,
) -> FeatureDiscoveryResponse:
    from azure.ai.projects import AIProjectClient  # type: ignore[import-not-found]
    from azure.identity import DefaultAzureCredential  # type: ignore[import-not-found]

    client = AIProjectClient(endpoint=endpoint, credential=DefaultAzureCredential())
    agent = _resolve_agent(client, agent_ref)
    prompt = _build_feature_discovery_prompt(payload)

    thread = client.agents.threads.create()
    client.agents.messages.create(thread_id=thread.id, role="user", content=prompt)
    run = client.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
    if getattr(run, "status", None) == "failed":
        raise ValueError(f"Foundry feature agent run failed: {getattr(run, 'last_error', None)}")

    messages = client.agents.messages.list(thread_id=thread.id)
    content = _extract_latest_text(messages)
    parsed = _parse_agent_json(content)
    return _feature_response_from_mapping(parsed, source="azure_ai_foundry", raw_response=content)


def _resolve_agent(client: Any, agent_ref: str) -> Any:
    get_agent = getattr(client.agents, "get_agent", None)
    if callable(get_agent):
        try:
            return get_agent(agent_ref)
        except (AttributeError, TypeError, ValueError, AzureError):
            pass

    agents = client.agents.list_agents()
    for agent in agents:
        if getattr(agent, "name", None) == agent_ref or getattr(agent, "id", None) == agent_ref:
            return agent
    raise ValueError(f"Foundry agent not found: {agent_ref}")


def _build_agent_prompt(evidence: dict[str, Any]) -> str:
    payload = to_agent_payload(evidence)
    return (
        "Use only the aggregate evidence below. Do not request or infer raw customer rows. "
        "Validate the bandit strategy and, only if safe, recommend a bounded UCB alpha. "
        "Return strict JSON with keys validation, recommended_ucb_alpha, rationale, warnings.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def _build_feature_discovery_prompt(payload: dict[str, Any]) -> str:
    return (
        "Use only the aggregate feature evidence below. Do not request, infer, or echo raw "
        "customer rows. Evaluate feature conditions and propose only safe feature-store "
        "candidates for an Azure ML contextual bandit workload. Return strict JSON with keys "
        "validation, accepted_features, rejected_features, proposed_features, "
        "feature_conditions, risk_flags, rationale, warnings.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def _extract_latest_text(messages: Any) -> str:
    for message in messages:
        if getattr(message, "role", None) != "assistant":
            continue
        for item in getattr(message, "content", []):
            text = getattr(getattr(item, "text", None), "value", None)
            if text:
                return str(text)
    raise ValueError("Foundry agent returned no assistant text.")


def _parse_agent_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(content[start : end + 1])


def _local_fallback_response(evidence: dict[str, Any], warnings: list[str]) -> AgentResponse:
    ucb = _find_policy(evidence.get("policy_summaries", []), "UCB")
    baseline = _find_policy(evidence.get("policy_summaries", []), "Deterministic Baseline")
    current_alpha = float(evidence.get("current_ucb_alpha", DEFAULT_UCB_ALPHA))
    bounds = tuple(evidence.get("ucb_alpha_bounds", DEFAULT_UCB_ALPHA_BOUNDS))

    ucb_regret = _metric(ucb, "cumulative_regret")
    baseline_regret = _metric(baseline, "cumulative_regret")
    ucb_exploration = _metric(ucb, "exploration_share")

    if ucb_regret is not None and baseline_regret is not None and ucb_regret > baseline_regret:
        recommended = current_alpha * 0.85
        rationale = "Fallback reduced exploration because UCB regret exceeded the baseline aggregate."
    elif ucb_exploration is not None and ucb_exploration < 0.05:
        recommended = current_alpha * 1.10
        rationale = "Fallback increased exploration because aggregate UCB exploration was very low."
    else:
        recommended = current_alpha
        rationale = "Fallback kept alpha stable because aggregate metrics show no unsafe drift."

    return AgentResponse(
        source="local_fallback",
        validation="valid",
        recommended_ucb_alpha=_bound_alpha(recommended, bounds),
        rationale=rationale,
        warnings=warnings,
    )


def _local_feature_discovery_response(
    payload: Mapping[str, Any],
    warnings: list[str],
) -> FeatureDiscoveryResponse:
    feature_evidence = list(payload.get("feature_evidence", []))
    accepted_features: list[str] = []
    rejected_features: list[dict[str, Any]] = []
    risk_flags: list[str] = []

    for item in feature_evidence:
        name = str(item.get("name", ""))
        missing_rate = _coerce_optional_float(item.get("missing_rate")) or 0.0
        unique_count = int(item.get("unique_count", 0) or 0)
        kind = str(item.get("kind", "unknown"))
        leakage_risk = bool(item.get("leakage_risk", False))

        if leakage_risk or name.lower() == "duration":
            rejected_features.append(
                {"name": name, "reason": "Excluded because it is a decision-time leakage risk."}
            )
            risk_flags.append(f"leakage_risk:{name}")
            continue
        if missing_rate > 0.30:
            rejected_features.append(
                {"name": name, "reason": f"Missing rate {missing_rate:.2%} exceeds 30%."}
            )
            continue
        if kind == "categorical" and unique_count > 100:
            rejected_features.append(
                {"name": name, "reason": f"Categorical cardinality {unique_count} exceeds 100."}
            )
            continue
        accepted_features.append(name)

    evidence_names = {str(item.get("name", "")) for item in feature_evidence}
    proposed_features: list[dict[str, Any]] = []
    if "age" in evidence_names:
        proposed_features.append(
            {
                "name": "age_band",
                "description": "Decision-time-safe age bucket for offer response heterogeneity.",
                "source_columns": ["age"],
                "transformation": "bucketize age into under_30, 30_44, 45_59, 60_plus",
                "expected_signal": "captures lifecycle-level conversion differences",
                "privacy_risk": "low",
                "leakage_risk": "low",
                "implementation_hint": "pd.cut on age before encoding",
            }
        )
    if {"campaign", "previous"}.issubset(evidence_names):
        proposed_features.append(
            {
                "name": "campaign_pressure",
                "description": "Total recent contact pressure before the current offer decision.",
                "source_columns": ["campaign", "previous"],
                "transformation": "campaign + previous",
                "expected_signal": "models fatigue and outreach saturation",
                "privacy_risk": "low",
                "leakage_risk": "low",
                "implementation_hint": "sum numeric contact counters",
            }
        )
    if "previous" in evidence_names or "pdays" in evidence_names:
        proposed_features.append(
            {
                "name": "has_prior_contact",
                "description": "Binary prior-contact indicator available at decision time.",
                "source_columns": sorted(evidence_names.intersection({"previous", "pdays"})),
                "transformation": "previous > 0 or pdays not in sentinel values",
                "expected_signal": "separates new and returning outreach populations",
                "privacy_risk": "low",
                "leakage_risk": "low",
                "implementation_hint": "derive boolean before one-hot encoding",
            }
        )

    feature_conditions = [
        {
            "name": "leakage_guard",
            "passed": "duration" not in accepted_features,
            "detail": "duration must remain excluded before offer-decision time.",
        },
        {
            "name": "missingness_guard",
            "passed": not any(
                (_coerce_optional_float(item.get("missing_rate")) or 0.0) > 0.30
                for item in feature_evidence
            ),
            "detail": "features with more than 30% missingness require review.",
        },
        {
            "name": "cardinality_guard",
            "passed": not any(
                str(item.get("kind")) == "categorical" and int(item.get("unique_count", 0) or 0) > 100
                for item in feature_evidence
            ),
            "detail": "high-cardinality categorical features require encoding review.",
        },
    ]

    return FeatureDiscoveryResponse(
        source="local_fallback",
        validation="valid" if accepted_features else "needs_review",
        accepted_features=accepted_features,
        rejected_features=rejected_features,
        proposed_features=proposed_features,
        feature_conditions=feature_conditions,
        risk_flags=sorted(set(risk_flags)),
        rationale="Fallback evaluated aggregate feature evidence with leakage, missingness, and cardinality guards.",
        warnings=warnings,
    )


def _feature_response_from_mapping(
    parsed: Mapping[str, Any],
    *,
    source: str,
    raw_response: str | None,
) -> FeatureDiscoveryResponse:
    return FeatureDiscoveryResponse(
        source=source,
        validation=str(parsed.get("validation", "needs_review")),
        accepted_features=[str(item) for item in parsed.get("accepted_features", [])],
        rejected_features=[dict(item) for item in parsed.get("rejected_features", [])],
        proposed_features=[dict(item) for item in parsed.get("proposed_features", [])],
        feature_conditions=[dict(item) for item in parsed.get("feature_conditions", [])],
        risk_flags=[str(item) for item in parsed.get("risk_flags", [])],
        rationale=str(parsed.get("rationale", "Feature agent returned a normalized response.")),
        warnings=[str(item) for item in parsed.get("warnings", [])],
        raw_response=raw_response,
    )


def _normalize_aggregate(aggregate: AggregateMetrics | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(aggregate, AggregateMetrics):
        return asdict(aggregate)
    evidence = dict(aggregate)
    if "policy_summaries" not in evidence:
        evidence = asdict(build_aggregate_metrics([evidence]))
    return evidence


def _sanitize_summary_row(row: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "policy",
        "total_reward",
        "cumulative_regret",
        "conversion_rate",
        "exploration_share",
        "mean_reward",
        "std_reward",
        "min_reward",
        "max_reward",
        "n_rounds",
    }
    return {key: _json_safe_value(value) for key, value in row.items() if key in allowed}


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _infer_n_rounds(rows: Sequence[Mapping[str, Any]]) -> int | None:
    for row in rows:
        value = row.get("n_rounds")
        if value is not None:
            return int(value)
    return None


def _find_policy(rows: Sequence[Mapping[str, Any]], policy_name: str) -> Mapping[str, Any] | None:
    for row in rows:
        if str(row.get("policy", "")).lower() == policy_name.lower():
            return row
    return None


def _metric(row: Mapping[str, Any] | None, name: str) -> float | None:
    if row is None or row.get(name) is None:
        return None
    return _coerce_optional_float(row.get(name))


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _bound_alpha(value: float | None, bounds: tuple[float, float]) -> float | None:
    if value is None:
        return None
    lower, upper = bounds
    return float(min(max(value, lower), upper))
