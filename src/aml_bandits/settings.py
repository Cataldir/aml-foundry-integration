"""Environment defaults for the rg-microsoft-iq AML and Foundry integration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


TaskName = Literal["feature", "strategy"]


@dataclass(frozen=True)
class AzureMLDefaults:
    resource_group: str
    workspace_name: str
    location: str


@dataclass(frozen=True)
class FoundryDefaults:
    resource_group: str
    account_name: str
    project_name: str
    project_endpoint: str
    location: str
    feature_agent_name: str
    strategy_agent_name: str
    available_agents: tuple[str, ...]


@dataclass(frozen=True)
class ProjectDefaults:
    azure_ml: AzureMLDefaults
    foundry: FoundryDefaults


DEFAULT_PROJECT = ProjectDefaults(
    azure_ml=AzureMLDefaults(
        resource_group="rg-microsoft-iq",
        workspace_name="aml-sample",
        location="eastus",
    ),
    foundry=FoundryDefaults(
        resource_group="rg-microsoft-iq",
        account_name="ai-miq-miqsec26",
        project_name="miq-project-miqsec26",
        project_endpoint="https://ai-miq-miqsec26.services.ai.azure.com/api/projects/miq-project-miqsec26",
        location="swedencentral",
        feature_agent_name="finance-orchestrator",
        strategy_agent_name="finance-orchestrator",
        available_agents=(
            "consultor-de-camiseta",
            "finance-orchestrator",
            "hr-orchestrator",
        ),
    ),
)


def load_project_defaults(config_path: str | Path | None = None) -> ProjectDefaults:
    """Load project defaults from config JSON, falling back to discovered rg-microsoft-iq values."""
    path = Path(config_path) if config_path else _default_config_path()
    if not path.exists():
        return DEFAULT_PROJECT

    payload = json.loads(path.read_text(encoding="utf-8"))
    azure_ml = payload.get("azure_ml", {})
    foundry = payload.get("foundry", {})
    return ProjectDefaults(
        azure_ml=AzureMLDefaults(
            resource_group=str(azure_ml.get("resource_group", DEFAULT_PROJECT.azure_ml.resource_group)),
            workspace_name=str(azure_ml.get("workspace_name", DEFAULT_PROJECT.azure_ml.workspace_name)),
            location=str(azure_ml.get("location", DEFAULT_PROJECT.azure_ml.location)),
        ),
        foundry=FoundryDefaults(
            resource_group=str(foundry.get("resource_group", DEFAULT_PROJECT.foundry.resource_group)),
            account_name=str(foundry.get("account_name", DEFAULT_PROJECT.foundry.account_name)),
            project_name=str(foundry.get("project_name", DEFAULT_PROJECT.foundry.project_name)),
            project_endpoint=str(
                foundry.get("project_endpoint", DEFAULT_PROJECT.foundry.project_endpoint)
            ),
            location=str(foundry.get("location", DEFAULT_PROJECT.foundry.location)),
            feature_agent_name=str(
                foundry.get("feature_agent_name", DEFAULT_PROJECT.foundry.feature_agent_name)
            ),
            strategy_agent_name=str(
                foundry.get("strategy_agent_name", DEFAULT_PROJECT.foundry.strategy_agent_name)
            ),
            available_agents=tuple(
                str(agent)
                for agent in foundry.get(
                    "available_agents",
                    DEFAULT_PROJECT.foundry.available_agents,
                )
            ),
        ),
    )


def apply_project_environment_defaults(defaults: ProjectDefaults | None = None) -> ProjectDefaults:
    """Populate non-secret environment defaults without overriding explicit user settings."""
    resolved = defaults or load_project_defaults()
    os.environ.setdefault("AZUREML_RESOURCE_GROUP", resolved.azure_ml.resource_group)
    os.environ.setdefault("AZUREML_WORKSPACE_NAME", resolved.azure_ml.workspace_name)
    os.environ.setdefault("AZUREML_LOCATION", resolved.azure_ml.location)
    os.environ.setdefault("FOUNDRY_PROJECT_RESOURCE_GROUP", resolved.foundry.resource_group)
    os.environ.setdefault("FOUNDRY_ACCOUNT_NAME", resolved.foundry.account_name)
    os.environ.setdefault("FOUNDRY_PROJECT_NAME", resolved.foundry.project_name)
    os.environ.setdefault("FOUNDRY_PROJECT_ENDPOINT", resolved.foundry.project_endpoint)
    os.environ.setdefault("AZURE_AI_PROJECT_ENDPOINT", resolved.foundry.project_endpoint)
    os.environ.setdefault("FOUNDRY_FEATURE_AGENT_NAME", resolved.foundry.feature_agent_name)
    os.environ.setdefault("AZURE_AI_FEATURE_AGENT_NAME", resolved.foundry.feature_agent_name)
    os.environ.setdefault("FOUNDRY_STRATEGY_AGENT_NAME", resolved.foundry.strategy_agent_name)
    os.environ.setdefault("AZURE_AI_STRATEGY_AGENT_NAME", resolved.foundry.strategy_agent_name)
    os.environ.setdefault("FOUNDRY_AVAILABLE_AGENTS", ",".join(resolved.foundry.available_agents))
    return resolved


def resolve_foundry_agent(task: TaskName, defaults: ProjectDefaults | None = None) -> tuple[str, str]:
    """Resolve the project endpoint and task-specific Foundry agent reference."""
    resolved = apply_project_environment_defaults(defaults)
    endpoint = (
        os.getenv("FOUNDRY_PROJECT_ENDPOINT")
        or os.getenv("AZURE_AI_PROJECT_ENDPOINT")
        or resolved.foundry.project_endpoint
    )
    if task == "feature":
        agent_ref = (
            os.getenv("FOUNDRY_FEATURE_AGENT_ID")
            or os.getenv("AZURE_AI_FEATURE_AGENT_ID")
            or os.getenv("FOUNDRY_FEATURE_AGENT_NAME")
            or os.getenv("AZURE_AI_FEATURE_AGENT_NAME")
            or resolved.foundry.feature_agent_name
        )
    else:
        agent_ref = (
            os.getenv("FOUNDRY_STRATEGY_AGENT_ID")
            or os.getenv("AZURE_AI_STRATEGY_AGENT_ID")
            or os.getenv("FOUNDRY_STRATEGY_AGENT_NAME")
            or os.getenv("AZURE_AI_STRATEGY_AGENT_NAME")
            or resolved.foundry.strategy_agent_name
        )
    return endpoint, agent_ref


def project_summary(defaults: ProjectDefaults | None = None) -> dict[str, Any]:
    """Return a display-safe summary of the AML and Foundry binding."""
    resolved = defaults or load_project_defaults()
    return {
        "azure_ml": {
            "resource_group": resolved.azure_ml.resource_group,
            "workspace_name": resolved.azure_ml.workspace_name,
            "location": resolved.azure_ml.location,
        },
        "foundry": {
            "resource_group": resolved.foundry.resource_group,
            "account_name": resolved.foundry.account_name,
            "project_name": resolved.foundry.project_name,
            "project_endpoint": resolved.foundry.project_endpoint,
            "location": resolved.foundry.location,
            "feature_agent_name": resolved.foundry.feature_agent_name,
            "strategy_agent_name": resolved.foundry.strategy_agent_name,
            "available_agents": list(resolved.foundry.available_agents),
        },
    }


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "foundry_project.json"