# Notebook Guide

The notebook set is segmented by operational responsibility. Use `demo.ipynb` for the full datathon flow, or run the utility notebooks independently when validating a specific layer.

- `00_environment_and_foundry_agents.ipynb` validates dependencies, Azure ML defaults, the Foundry project endpoint, and the deployed agent inventory in `rg-microsoft-iq`.
- `01_data_sources_assets_imports.ipynb` materializes source data, writes the source manifest, generates a gated `az ml data import` command, registers the raw data asset, and loads data.
- `02_feature_store_and_feature_agent.ipynb` builds aggregate feature evidence, calls the deployed `finance-orchestrator` agent for feature conditions/propositions, creates MLTable artifacts, registers feature assets, and writes Feature Store specs.
- `03_bandit_strategy_agent_loop.ipynb` runs the contextual bandit workload and calls the deployed `finance-orchestrator` agent for aggregate-only strategy validation.
- `demo.ipynb` runs the complete flow end to end.

Resource-mutating operations are gated:

- Set `ENABLE_AZUREML_DATA_IMPORT=true` to execute `az ml data import`.
- Set `ENABLE_AZUREML_FEATURE_STORE_CREATE=true` to execute first-class Azure ML Feature Store commands.

Both gates default to `false` so the notebooks can be reviewed and dry-run without creating Azure resources unexpectedly.