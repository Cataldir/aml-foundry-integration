# System Architecture

## Overview

This project implements a complete machine learning pipeline for adaptive experimentation (multi-armed bandits) that can run locally or on Azure Machine Learning Foundry infrastructure.

```
┌──────────────────┐
│  Data Sources    │
│  (Kaggle/OpenML) │
└────────┬─────────┘
         │
         v
┌──────────────────────────┐
│  Data Loading Layer      │
│  (local-first, fallback) │
└────────┬─────────────────┘
         │
         v
┌──────────────────────────────┐
│  Preprocessing Layer         │
│  (leakage control, features) │
└────────┬─────────────────────┘
         │
         v
┌──────────────────────────────────┐
│  Synthetic Bandit Environment    │
│  (contexts, offers, probs)       │
└────────┬─────────────────────────┘
         │
         v
┌────────────────────────────────────────┐
│  Policy Simulation (Delayed Rewards)   │
│  - Deterministic Baseline              │
│  - UCB (Upper Confidence Bound)        │
│  - Thompson Sampling                   │
└────────┬─────────────────────────────────┘
         │
         v
┌────────────────────────────────┐
│  Evaluation & Metrics          │
│  (regret, reward, conversion)  │
└────────┬─────────────────────────┘
         │
         v
┌────────────────────────────────────┐
│  Foundry Agent Bridge (Optional)   │
│  aggregate-only validation         │
└────────┬───────────────────────────┘
         │
         v
┌────────────────────────────────┐
│  Results Visualization         │
│  (plots, tables, reports)      │
└────────────────────────────────┘
```

## Components

### Data Layer

- **Data Loader** (`data_loader.py`):
  - Tries Kaggle API first (if `kaggle` package installed)
  - Falls back to local CSV files
  - Final fallback to OpenML `bank-marketing` dataset
  - Returns provenance tracking for reproducibility

- **Preprocessing** (`preprocessing.py`):
  - Removes `duration` column (leakage risk)
  - One-hot encodes categorical features
  - Standardizes numeric features
  - Returns fitted preprocessor for reproducibility

### Bandit Policies

Three distinct policies implemented in `bandits.py`:

1. **Deterministic Baseline**:
   - Always selects the same arm (e.g., "Premium Bundle")
   - No learning or exploration
   - Serves as a lower bound for policy comparison

2. **Upper Confidence Bound (UCB)**:
   - Balances exploration and exploitation via confidence intervals
   - Optimistic: uncertain arms get exploration bonuses
   - Asymptotically optimal regret for known problem class
   - Hyperparameter: `alpha` (exploration scale)

3. **Thompson Sampling**:
   - Bayesian approach: maintains Beta posteriors per arm
   - Samples from posterior and picks highest sampled expected reward
   - Naturally balances exploration via posterior uncertainty
   - Often matches or beats UCB in practice

### Simulation Engine

`simulator.py` implements:

- **Contextual environment** with realistic arm-context-reward relationships
- **Delayed feedback** (geometric distribution, max 8 rounds)
- **Online learning**: policies update only after feedback matures
- **Metric tracking**: cumulative reward, regret, conversion rate, exploration share

### Evaluation & Metrics

`metrics.py` computes:

- **Cumulative reward**: Total value captured by policy
- **Cumulative regret**: Opportunity cost vs oracle
- **Conversion rate**: Effective offer acceptance
- **Exploration share**: Exploration vs exploitation balance
- **Windowed metrics**: Learning curves per policy

### Foundry Agent Bridge

`foundry_bridge.py` is an optional adapter/facade around Azure AI Foundry agent concepts. It receives only aggregate summary metrics, validates the strategy, and can recommend a bounded UCB alpha. When Foundry SDK packages or environment variables are unavailable, it uses a deterministic local fallback with the same safe contract.

## Execution Modes

### Local Development

```bash
python examples/run_bandits.py
```

Runs entirely on local machine; useful for development and testing.

### Azure ML Compute Instance (SSH)

Connect via Remote-SSH from VS Code (see INTEGRATION_GUIDE.md):

```bash
# From compute instance terminal
python examples/run_bandits.py
```

Code runs directly on managed Azure VM; can access workspace storage natively.

### Azure ML Job Submission

```bash
az ml job create --file examples/job_config.yaml -g <your-resource-group> -w <your-aml-workspace-name>
```

Submits job to Foundry; runs on specified compute target (instance or cluster).

### Kubernetes Cluster

Submit to AKS cluster for distributed/scaled execution:

```bash
az ml job create --file examples/job_config_kubernetes.yaml -g <your-resource-group> -w <your-aml-workspace-name>
```

## Data Flow

1. **Load**: Kaggle → Local CSV → OpenML
2. **Preprocess**: Remove leakage, encode, scale
3. **Create Environment**: Dense contexts, arm probabilities, margins
4. **Simulate**: For each round:
   - Policy selects action
   - Reward generated from true arm-context probability
   - Feedback delayed (random schedule)
   - Policy updates when feedback matures
5. **Evaluate**: Compute metrics and generate plots

## Reproducibility

- Fixed random seeds (42)
- Versioned dependencies (`pyproject.toml`)
- Data provenance tracking (returns source with dataframe)
- Saved results and plots (CSV + PNG)

## Integration with Azure ML

- Workspace: your Azure ML workspace
- Resource Group: your Azure resource group
- Compute Instance: your Azure ML compute instance
- Kubernetes Cluster: your Azure ML Kubernetes compute target
- Storage: your workspace storage account using managed identity auth
- Foundry Agent Bridge: optional aggregate-only validation path controlled by `FOUNDRY_PROJECT_ENDPOINT` and `FOUNDRY_AGENT_ID` or `FOUNDRY_AGENT_NAME`

Placeholder Azure ML compute templates are in `azure-ml/compute_instance.yml` and `azure-ml/compute_kubernetes.yml`. They intentionally use placeholders only and must be resolved outside source control before running `az ml compute create`.

See `INTEGRATION_GUIDE.md` for detailed setup instructions.
