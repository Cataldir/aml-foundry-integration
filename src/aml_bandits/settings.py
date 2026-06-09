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
class FeatureStoreDefaults:
    resource_group: str
    name: str
    location: str
    offline_store: str
    storage_account: str
    materialization_identity: str


@dataclass(frozen=True)
class ProjectDefaults:
    azure_ml: AzureMLDefaults
    foundry: FoundryDefaults
    feature_store: FeatureStoreDefaults


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
    feature_store=FeatureStoreDefaults(
        resource_group="rg-microsoft-iq",
        name="miq-7mlet-feature-store",
        location="eastus",
        offline_store="/subscriptions/150e82e8-25db-4f1a-8e04-a2f6a77d26c4/resourceGroups/rg-microsoft-iq/providers/Microsoft.Storage/storageAccounts/miq7mletstorage9d037fcb2/blobServices/default/containers/miq7mletcontainec50a8368",
        storage_account="/subscriptions/150e82e8-25db-4f1a-8e04-a2f6a77d26c4/resourceGroups/rg-microsoft-iq/providers/Microsoft.Storage/storageAccounts/miq7mletstorage9d037fcb2",
        materialization_identity="/subscriptions/150e82e8-25db-4f1a-8e04-a2f6a77d26c4/resourceGroups/rg-microsoft-iq/providers/Microsoft.ManagedIdentity/userAssignedIdentities/materialization-uai-7ba3a59a854d382995b65a47a6834cff",
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
    feature_store = payload.get("feature_store", {})
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
        feature_store=FeatureStoreDefaults(
            resource_group=str(
                feature_store.get("resource_group", DEFAULT_PROJECT.feature_store.resource_group)
            ),
            name=str(feature_store.get("name", DEFAULT_PROJECT.feature_store.name)),
            location=str(feature_store.get("location", DEFAULT_PROJECT.feature_store.location)),
            offline_store=str(
                feature_store.get("offline_store", DEFAULT_PROJECT.feature_store.offline_store)
            ),
            storage_account=str(
                feature_store.get("storage_account", DEFAULT_PROJECT.feature_store.storage_account)
            ),
            materialization_identity=str(
                feature_store.get(
                    "materialization_identity",
                    DEFAULT_PROJECT.feature_store.materialization_identity,
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
    os.environ.setdefault("AZUREML_FEATURE_STORE_RESOURCE_GROUP", resolved.feature_store.resource_group)
    os.environ.setdefault("AZUREML_FEATURE_STORE_NAME", resolved.feature_store.name)
    os.environ.setdefault("AZUREML_FEATURE_STORE_LOCATION", resolved.feature_store.location)
    os.environ.setdefault("AZUREML_FEATURE_STORE_OFFLINE_STORE", resolved.feature_store.offline_store)
    os.environ.setdefault("AZUREML_FEATURE_STORE_STORAGE_ACCOUNT", resolved.feature_store.storage_account)
    os.environ.setdefault(
        "AZUREML_FEATURE_STORE_MATERIALIZATION_IDENTITY",
        resolved.feature_store.materialization_identity,
    )
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
        "feature_store": {
            "resource_group": resolved.feature_store.resource_group,
            "name": resolved.feature_store.name,
            "location": resolved.feature_store.location,
            "offline_store": resolved.feature_store.offline_store,
            "storage_account": resolved.feature_store.storage_account,
            "materialization_identity": resolved.feature_store.materialization_identity,
        },
    }


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "foundry_project.json"