"""
Aggregate-only Azure AI Foundry bridge for bandit strategy validation.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

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
    endpoint = os.getenv("FOUNDRY_PROJECT_ENDPOINT") or os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    agent_ref = (
        os.getenv("FOUNDRY_AGENT_ID")
        or os.getenv("AZURE_AI_AGENT_ID")
        or os.getenv("FOUNDRY_AGENT_NAME")
        or os.getenv("AZURE_AI_AGENT_NAME")
    )

    if not endpoint or not agent_ref:
        return _local_fallback_response(
            evidence,
            ["Foundry endpoint or agent name/id is not configured."],
        )

    try:
        return _send_with_azure_ai_projects(evidence, endpoint, agent_ref)
    except Exception as ex:  # pragma: no cover - depends on optional SDK and Azure auth
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
        raise RuntimeError(f"Foundry agent run failed: {getattr(run, 'last_error', None)}")

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


def _resolve_agent(client: Any, agent_ref: str) -> Any:
    get_agent = getattr(client.agents, "get_agent", None)
    if callable(get_agent):
        try:
            return get_agent(agent_ref)
        except Exception:
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


def _extract_latest_text(messages: Any) -> str:
    for message in messages:
        if getattr(message, "role", None) != "assistant":
            continue
        for item in getattr(message, "content", []):
            text = getattr(getattr(item, "text", None), "value", None)
            if text:
                return str(text)
    raise RuntimeError("Foundry agent returned no assistant text.")


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