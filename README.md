# AML Foundry Integration: Multi-Armed Bandits for Adaptive Experimentation

Complete, production-grade implementation of adaptive experimentation using multi-armed bandits (UCB, Thompson Sampling) on Azure Machine Learning Foundry environment.

**Status**: ✅ Ready for FIAP 7-MLET Datathon

## Overview

This repository implements a reproducible pipeline for adaptive offer/message selection in financial campaigns using classical multi-armed bandit algorithms on Azure ML Foundry infrastructure.

### Key Components

- **Data Pipeline**: Kaggle-first loading with local/OpenML fallback
- **Classical ML**: Preprocessing with explicit leakage handling
- **Bandits**: Deterministic baseline, UCB (Upper Confidence Bound), Thompson Sampling (Beta-Bernoulli)
- **Evaluation**: Synthetic contextual environment with delayed rewards
- **Metrics**: Cumulative reward, regret, conversion rate, exploration share
- **Azure ML Integration**: Native support for Foundry compute instances and Kubernetes clusters
- **Agentic Validation**: Optional aggregate-only Foundry bridge with local fallback for safe UCB alpha recommendations

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Cataldir/aml-foundry-integration.git
cd aml-foundry-integration

# Create virtual environment
python -m venv .venv
source .venv/Scripts/Activate.ps1  # Windows
# or source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -e .

# Optional: Install Kaggle and Foundry agent support
pip install -e ".[kaggle,foundry]"
```

### Usage

#### Option 1: Azure ML Compute Instance (Recommended)

```bash
# Connect to your compute instance, then run the pipeline
python examples/run_bandits.py
```

#### Option 2: Local or Kubernetes

```bash
# Run locally
python examples/run_bandits.py

# Run locally with aggregate-only Foundry bridge or local fallback
python examples/run_bandits.py --use-agent

# Submit to an Azure ML Kubernetes compute target with an Azure ML job config
az ml job create --file examples/job_config_kubernetes.yaml -g <your-resource-group> -w <your-aml-workspace-name>
```

#### Option 3: Interactive Notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

## Configuration

### Environment Variables

Create a `.env` file in the repository root:

```bash
# Kaggle credentials (optional)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# Azure ML
SUBSCRIPTION_ID=your-subscription-id
RESOURCE_GROUP=your-resource-group
WORKSPACE_NAME=your-aml-workspace-name
AZURE_ML_WORKSPACE=your-aml-workspace-name
AZURE_ML_COMPUTE_INSTANCE_NAME=your-compute-instance-name
AZURE_ML_COMPUTE_AKS_NAME=your-kubernetes-compute-name

# Optional Azure AI Foundry agent bridge
FOUNDRY_PROJECT_ENDPOINT=your-foundry-project-endpoint
FOUNDRY_AGENT_ID=your-foundry-agent-id
FOUNDRY_AGENT_NAME=your-foundry-agent-name-or-id

# Optional: Data paths
KAGGLE_DATASET=henriqueyamahata/bank-marketing
DATA_DIR=./data
```

For Kaggle authentication, you can also place `~/.kaggle/kaggle.json`:

```json
{
  "username": "your_username",
  "key": "your_api_key"
}
```

## Azure ML Integration

### What You Get

- ✅ **Compute Instance**: Interactive development & testing
- ✅ **Kubernetes Cluster**: Scalable production inference
- ✅ **Managed Identity**: RBAC-secured storage access
- ✅ **Network Security**: Firewall rules + whitelisted IP access
- ✅ **MLflow Tracking**: Experiment versioning and comparison
- ✅ **Agentic Bridge**: Aggregate-only strategy validation and bounded UCB alpha enrichment

### Optional Foundry Agent Bridge

Run `python examples/run_bandits.py --use-agent` to validate summary metrics through `aml_bandits.foundry_bridge`. If `FOUNDRY_PROJECT_ENDPOINT`, `FOUNDRY_AGENT_ID`/`FOUNDRY_AGENT_NAME`, or the optional Foundry SDK packages are unavailable, the code uses a deterministic local fallback. The bridge sends only aggregate policy metrics and allows only bounded UCB alpha recommendations; it never sends raw customer rows.

### Azure ML Compute Templates

Placeholder compute templates live in `azure-ml/compute_instance.yml` and `azure-ml/compute_kubernetes.yml`. Replace placeholders through your deployment process before running `az ml compute create --file ...`; do not commit concrete subscription, workspace, AKS, or identity values.

### How to Connect from VS Code

See **[INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md)** for step-by-step instructions on:

1. ✅ Connecting to Azure ML workspace
2. ✅ Setting up VS Code with remote SSH over compute instance
3. ✅ Running code directly on Foundry compute
4. ✅ Submitting jobs to Kubernetes cluster
5. ✅ Viewing logs and metrics in Azure Portal

## Project Structure

```
aml-foundry-integration/
├── README.md                    # This file
├── INTEGRATION_GUIDE.md         # Step-by-step AML + VS Code setup
├── pyproject.toml              # Project metadata & dependencies
├── .gitignore                   # Git ignore rules
├── src/
│   └── aml_bandits/
│       ├── __init__.py
│       ├── data_loader.py       # Kaggle/OpenML dataset loading
│       ├── preprocessing.py     # Classical ML preprocessing
│       ├── bandits.py           # Policy implementations
│       ├── simulator.py         # Bandit simulation engine
│       ├── metrics.py           # Evaluation metrics
│       ├── foundry_bridge.py    # Aggregate-only Foundry agent bridge
│       └── utils.py             # Utilities
├── notebooks/
│   └── demo.ipynb              # Interactive demonstration
├── examples/
│   ├── run_bandits.py          # Standalone CLI runner
│   ├── job_config.yaml         # Azure ML compute instance job template
│   └── job_config_kubernetes.yaml # Azure ML AKS-backed job template
├── azure-ml/
│   ├── compute_instance.yml    # Compute instance template
│   └── compute_kubernetes.yml  # AKS-backed Kubernetes compute template
└── docs/
    ├── architecture.md          # System design
    └── algorithms.md            # Algorithm explanations
```

## Running the Pipeline

### End-to-End Workflow

```python
from aml_bandits import (
    load_bank_marketing_dataset,
    preprocess_data,
    build_bandit_environment,
  BanditSimulator,
  UCBPolicy,
  compute_metrics,
  build_aggregate_metrics,
  validate_and_enrich_strategy,
)

# 1. Load data (Kaggle → Local → OpenML)
raw_df, target_col, provenance = load_bank_marketing_dataset()
print(f"Loaded from: {provenance}")

# 2. Preprocess with leakage control
X, y, preprocessor = preprocess_data(raw_df, target_col)

# 3. Create synthetic bandit environment
contexts, offer_catalog, all_probs, margins = build_bandit_environment(X, preprocessor)

# 4. Run a policy and evaluate it
simulator = BanditSimulator(all_probs, margins)
ucb = UCBPolicy(n_arms=len(margins), margins_arr=margins, alpha=1.2)
results = simulator.run_policy("UCB", ucb)
summary = compute_metrics(results)

# 5. Ask the agent bridge to validate aggregate evidence
evidence = build_aggregate_metrics(summary, current_ucb_alpha=1.2)
recommendation = validate_and_enrich_strategy(evidence)
print(recommendation)
```

## Algorithms

### Deterministic Baseline
- **Policy**: Always select the same fixed arm (e.g., "Premium Bundle")
- **Pros**: Simple, reproducible, low computation
- **Cons**: No exploration, ignores context, suboptimal

### Upper Confidence Bound (UCB)
- **Formula**: `UCB_t(a) = (success/(trials+2)) + sqrt(log(t)/(trials+1))`
- **Intuition**: Optimistic exploration—uncertain arms get bonuses
- **Pros**: Regret-optimal for known problem class
- **Cons**: Requires tuning alpha parameter

### Thompson Sampling (Bayesian Bandits)
- **Model**: Beta-Bernoulli posterior per arm
- **Update**: Bayesian inference on observed conversions
- **Decision**: Sample θ ~ Beta(α, β) per arm, pick arm with max expected reward
- **Pros**: Natural exploration via uncertainty, empirically competitive
- **Cons**: Slower than UCB on very large scale

## Evaluation Metrics

| Metric | Definition | Interpretation |
|--------|-----------|-----------------|
| **Cumulative Reward** | Sum of rewards over rounds | Total value captured by policy |
| **Cumulative Regret** | Sum of (optimal − chosen) reward | Opportunity cost vs oracle |
| **Conversion Rate** | Conversions / decisions | Effective offer acceptance rate |
| **Exploration Share** | Exploration actions / total actions | How much policy explores vs exploits |

## Delayed Rewards

Real-world campaigns often have delayed feedback (e.g., customer conversion after 3 days). This implementation:

- Generates random delays (geometric distribution, max 8 rounds)
- Delays policy updates until feedback matures
- Tracks metrics by decision time (not feedback time)
- Aligns with real production constraints

## Data Sources

### Recommended Datasets

| Dataset | Size | Source | Usage |
|---------|------|--------|-------|
| `henriqueyamahata/bank-marketing` | 45K | Kaggle | Primary (default) |
| `tunguz/bank-marketing-data-set` | 45K | Kaggle | Alternative 1 |
| `dharmik34/bank-term-deposit-subscription` | — | Kaggle | Alternative 2 |

All datasets are marketing/conversion-based and suitable for adaptive experimentation.

### Data Leakage Protection

The pipeline explicitly removes:
- ✅ `duration`: Call duration (known only after decision)
- ✅ Any post-contact or post-campaign columns

This ensures decisions are made on pre-contact information only.

## Testing

```bash
pytest tests/
pytest tests/ -v --cov=src/aml_bandits
```

## CI/CD & MLOps

### Azure ML Integration Points

- **Data Registry**: Versioned datasets in AML data stores
- **Model Registry**: Policy checkpoints saved and compared
- **MLflow**: Experiment tracking and hyperparameter tuning
- **Compute**: Runs on compute instances or Kubernetes
- **Inference**: Deploy policies as REST endpoints or batch jobs

### Production Checklist

- [ ] Data lineage documented in MLflow
- [ ] Model card updated with latest metrics
- [ ] Fairness and bias analysis completed
- [ ] Cold-start strategy validated
- [ ] Delayed reward handling tested
- [ ] Rollback procedure documented
- [ ] Monitoring and alerting configured

## Contributing

Contributions welcome! See `CONTRIBUTING.md` for guidelines.

## Authors

- **Ricardo Cataldi** — Azure & ML engineering
- **7-MLET Datathon Teams** — Algorithm implementations

## License

MIT License — see `LICENSE` file for details.

## References

### Papers

- Lattimore & Szepesvári (2020) — *Bandit Algorithms*
- Bubeck & Cesa-Bianchi (2012) — *Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems*
- Russo & Van Roy (2014) — *Learning to Optimize Via Posterior Sampling*

### Resources

- [Azure ML Foundry Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)
- [Multi-Armed Bandits Overview](https://en.wikipedia.org/wiki/Multi-armed_bandit)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

---

**Ready to integrate with Azure ML?** Start with [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md)
