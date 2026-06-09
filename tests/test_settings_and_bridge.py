from __future__ import annotations

import pytest

import aml_bandits.foundry_bridge as bridge
from aml_bandits.foundry_bridge import validate_and_enrich_strategy, validate_and_propose_features
from aml_bandits.settings import apply_project_environment_defaults, load_project_defaults, resolve_foundry_agent


@pytest.fixture(autouse=True)
def clear_foundry_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "FOUNDRY_PROJECT_ENDPOINT",
        "AZURE_AI_PROJECT_ENDPOINT",
        "FOUNDRY_FEATURE_AGENT_ID",
        "AZURE_AI_FEATURE_AGENT_ID",
        "FOUNDRY_FEATURE_AGENT_NAME",
        "AZURE_AI_FEATURE_AGENT_NAME",
        "FOUNDRY_STRATEGY_AGENT_ID",
        "AZURE_AI_STRATEGY_AGENT_ID",
        "FOUNDRY_STRATEGY_AGENT_NAME",
        "AZURE_AI_STRATEGY_AGENT_NAME",
    ):
        monkeypatch.delenv(key, raising=False)


def test_project_defaults_bind_to_rg_microsoft_iq() -> None:
    defaults = apply_project_environment_defaults(load_project_defaults())

    assert defaults.azure_ml.resource_group == "rg-microsoft-iq"
    assert defaults.azure_ml.workspace_name == "aml-sample"
    assert defaults.foundry.project_name == "miq-project-miqsec26"
    assert defaults.foundry.project_endpoint.endswith("/api/projects/miq-project-miqsec26")
    assert defaults.feature_store.name == "miq-7mlet-feature-store"
    assert defaults.feature_store.location == "eastus"


def test_resolve_foundry_agent_uses_finance_orchestrator_by_default() -> None:
    endpoint, feature_agent = resolve_foundry_agent("feature")
    _, strategy_agent = resolve_foundry_agent("strategy")

    assert "miq-project-miqsec26" in endpoint
    assert feature_agent == "finance-orchestrator"
    assert strategy_agent == "finance-orchestrator"


def test_resolve_foundry_agent_allows_task_specific_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOUNDRY_FEATURE_AGENT_NAME", "feature-reviewer")
    monkeypatch.setenv("FOUNDRY_STRATEGY_AGENT_NAME", "strategy-reviewer")

    assert resolve_foundry_agent("feature")[1] == "feature-reviewer"
    assert resolve_foundry_agent("strategy")[1] == "strategy-reviewer"


def test_bridge_fallback_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    def sdk_unavailable(*_: object) -> object:
        raise ImportError("SDK disabled for deterministic fallback test")

    monkeypatch.setattr(bridge, "_send_with_azure_ai_projects", sdk_unavailable)
    monkeypatch.setattr(bridge, "_send_feature_discovery_with_azure_ai_projects", sdk_unavailable)

    strategy = validate_and_enrich_strategy(
        {
            "policy_summaries": [
                {
                    "policy": "UCB",
                    "rounds": 100,
                    "total_reward": 24.0,
                    "conversion_rate": 0.24,
                    "cumulative_regret": 8.0,
                    "exploration_share": 0.12,
                },
                {
                    "policy": "Deterministic Baseline",
                    "rounds": 100,
                    "total_reward": 21.0,
                    "conversion_rate": 0.21,
                    "cumulative_regret": 11.0,
                    "exploration_share": 0.0,
                },
            ],
            "current_ucb_alpha": 1.2,
            "ucb_alpha_bounds": [0.1, 2.0],
        }
    )
    features = validate_and_propose_features(
        {
            "row_count": 12,
            "features": [
                {
                    "name": "age",
                    "kind": "numeric",
                    "missing_rate": 0.0,
                    "unique_count": 10,
                    "target_signal": 0.05,
                },
                {
                    "name": "duration",
                    "kind": "numeric",
                    "missing_rate": 0.0,
                    "unique_count": 12,
                    "target_signal": 0.4,
                },
            ],
        }
    )

    assert strategy.source == "local_fallback"
    assert strategy.validation == "valid"
    assert features.source == "local_fallback"
    assert features.accepted_features == ["age"]
    assert features.rejected_features[0]["name"] == "duration"