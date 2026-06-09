# Azure ML Foundry Integration Guide for VS Code

Complete step-by-step guide to connect this project to your Azure ML workspace and run code directly from VS Code.

## Prerequisites

- Azure ML workspace created
- Azure ML compute instance available for interactive development
- Optional Azure ML Kubernetes compute target for scaled jobs
- VS Code installed locally
- Python 3.9+ installed locally

## Part 1: Install Azure ML CLI & SDK

### Step 1a: Install Azure CLI

```powershell
# Install Azure CLI
choco install azure-cli  # If using Chocolatey
# OR download from https://aka.ms/cli

# Verify installation
az --version
```

### Step 1b: Install Azure ML SDK

```powershell
# Activate your local virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install azure-ai-ml SDK
pip install azure-ai-ml
pip install azure-identity
pip install jupyter
```

### Step 1c: Authenticate with Azure

```powershell
# Log in to Azure
az login

# Set default subscription
az account set --subscription "<your-subscription-id>"

# Verify
az account show
```

## Part 2: Connect to Azure ML Workspace

### Step 2a: Create `.env` Configuration

Create a file named `.env` in the repository root:

```bash
SUBSCRIPTION_ID=<your-subscription-id>
RESOURCE_GROUP=<your-resource-group>
WORKSPACE_NAME=<your-aml-workspace-name>
WORKSPACE_LOCATION=<your-workspace-region>

# Compute targets
COMPUTE_INSTANCE_NAME=<your-compute-instance-name>
KUBERNETES_CLUSTER_NAME=<your-kubernetes-compute-name>
AZURE_ML_WORKSPACE=<your-aml-workspace-name>
AZURE_ML_COMPUTE_INSTANCE_NAME=<your-compute-instance-name>
AZURE_ML_COMPUTE_AKS_NAME=<your-kubernetes-compute-name>

# Optional aggregate-only Foundry bridge
FOUNDRY_PROJECT_ENDPOINT=<your-foundry-project-endpoint>
FOUNDRY_AGENT_ID=<your-foundry-agent-id>
FOUNDRY_AGENT_NAME=<your-foundry-agent-name-or-id>

# Storage
STORAGE_ACCOUNT_NAME=<your-storage-account-name>
STORAGE_LOCATION=<your-storage-region>

# Optional: Kaggle
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

### Step 2b: Test Connection from Local Python

```powershell
# Run this test script
python -c "
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
   subscription_id='<your-subscription-id>',
   resource_group_name='<your-resource-group>',
   workspace_name='<your-aml-workspace-name>'
)

workspace = ml_client.workspaces.get('<your-aml-workspace-name>')
print(f'Connected to workspace: {workspace.name}')
print(f'Location: {workspace.location}')
"
```

Expected output:
```
Connected to workspace: <your-aml-workspace-name>
Location: <your-workspace-region>
```

## Part 3: Option A — SSH into Compute Instance

This allows you to run VS Code **directly on the compute instance** for interactive development.

### Step 3a: Get SSH Connection Details

```powershell
# Get compute instance details
az ml compute show -n <your-compute-instance-name> -g <your-resource-group> -w <your-aml-workspace-name> --query "{name:name, sshPublicAccess:properties.sshPublicAccess, publicIpAddress:properties.publicIpAddress}"

# Example output:
# {
#   "name": "<your-compute-instance-name>",
#   "sshPublicAccess": true,
#   "publicIpAddress": "<compute-instance-public-ip>"
# }
```

### Step 3b: Configure SSH in VS Code

1. **Install Remote - SSH Extension** in VS Code:
   - Open Extensions (Ctrl+Shift+X)
   - Search for "Remote - SSH"
   - Click "Install"

2. **Add SSH Configuration**:
   - Press `Ctrl+Shift+P` → "Remote-SSH: Open SSH Configuration File"
   - Add this entry:

```
Host aml-compute
   HostName <compute-instance-public-ip>
   User <ssh-user>
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
```

3. **Generate SSH Key** (if needed):

```powershell
# Use the SSH key generated during compute instance creation
# It should be in your Azure ML workspace or generate new one:
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
```

### Step 3c: Connect from VS Code

1. Open VS Code Command Palette: `Ctrl+Shift+P`
2. Type: "Remote-SSH: Connect to Host"
3. Select: `aml-compute`
4. Wait for remote connection to establish
5. Open your project folder on the compute instance

**Now you're running VS Code on the compute instance!** Your code runs directly on Azure ML.

## Part 4: Option B — Run Jobs Remotely (Recommended for Production)

This submits jobs to Azure ML for execution while you control from local VS Code.

### Step 4a: Create Job Configuration

Create `examples/job_config.yaml`:

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/jobBaseSchema.json
type: command
code: .
command: >
   python examples/run_bandits.py
compute: azureml:<your-compute-instance-name>
description: Run multi-armed bandits experiment
display_name: bandits-experiment-001
experiment_name: bandits-datathon
```

### Step 4b: Submit Job from Local Machine

```powershell
# Submit job to Azure ML
az ml job create --file examples/job_config.yaml -g <your-resource-group> -w <your-aml-workspace-name>

# List recent jobs
az ml job list -g <your-resource-group> -w <your-aml-workspace-name> --query "[].{name:name, status:status}" -o table

# Stream logs of a specific job
az ml job stream -n <job_id> -g <your-resource-group> -w <your-aml-workspace-name>
```

## Part 5: Run Code on Kubernetes Cluster

For large-scale experiments or production inference:

### Step 5a: Deploy to Kubernetes

```powershell
# Check Kubernetes cluster status
az ml compute show -n <your-kubernetes-compute-name> -g <your-resource-group> -w <your-aml-workspace-name>

# Create job config targeting Kubernetes
# (See Step 4a, but change compute to: azureml:<your-kubernetes-compute-name>)
```

### Step 5b: Submit to Kubernetes

```powershell
az ml job create --file examples/job_config_kubernetes.yaml -g <your-resource-group> -w <your-aml-workspace-name>
```

### Step 5c: Create Placeholder Compute Targets

This repository includes placeholder-only Azure ML compute templates:

```powershell
az ml compute create --file azure-ml/compute_instance.yml -g <your-resource-group> -w <your-aml-workspace-name>
az ml compute create --file azure-ml/compute_kubernetes.yml -g <your-resource-group> -w <your-aml-workspace-name>
```

Replace placeholders outside source control before submitting. The Kubernetes template expects an existing AKS resource ID, namespace, and managed identity resource ID.

## Part 5b: Optional Foundry Agent Bridge

The runner can call an Azure AI Foundry agent using only aggregate policy metrics:

```powershell
pip install -e ".[foundry]"
python examples/run_bandits.py --use-agent
```

If the Foundry endpoint, agent ID/name, SDK, or credentials are unavailable, `aml_bandits.foundry_bridge` uses a deterministic local fallback. The bridge contract allows only a bounded UCB alpha recommendation and does not send raw customer-level rows.

## Part 6: VS Code Extensions & Configuration

### Recommended Extensions

Install these VS Code extensions for better Azure ML integration:

1. **Azure Machine Learning** (Microsoft)
   - VSCode Extension: `ms-toolsai.vscode-ai`
   - Features: Browse workspace, submit jobs, view runs

2. **Azure Account** (Microsoft)
   - VSCode Extension: `ms-vscode.azure-account`
   - Features: Azure sign-in, subscription switching

3. **Python** (Microsoft)
   - Features: Linting, formatting, testing

4. **Jupyter** (Microsoft)
   - Features: Notebook support

5. **Docker** (Microsoft)
   - Features: Containerization for custom environments

### Install Extensions

```powershell
# From command line:
code --install-extension ms-toolsai.vscode-ai
code --install-extension ms-vscode.azure-account
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-vscode.docker
```

### VS Code Settings

Create or update `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.python",
    "editor.formatOnSave": true
  }
}
```

## Part 7: Integrated Workflow

### Local Development Loop

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Pull latest code
git pull

# 3. Install dependencies
pip install -e .

# 4. Run locally to test
python examples/run_bandits.py

# 5. Once tested, submit to Azure ML
az ml job create --file examples/job_config.yaml -g <your-resource-group> -w <your-aml-workspace-name>
```

### Remote Development (SSH)

```powershell
# 1. In VS Code: Connect to aml-compute via Remote-SSH
# 2. Open integrated terminal (Ctrl+`)
# 3. Navigate to repository:
#    cd /home/azureuser/aml-foundry-integration
# 4. Pull and install:
#    git pull && pip install -e .
# 5. Run directly on compute instance:
#    python examples/run_bandits.py
```

### Kubernetes Job Submission

```powershell
# 1. Test locally first
python examples/run_bandits.py

# 2. Update job config with Kubernetes target
# 3. Submit:
az ml job create --file examples/job_config_kubernetes.yaml -g <your-resource-group> -w <your-aml-workspace-name>

# 4. Monitor:
az ml job stream -n <job_id> -g <your-resource-group> -w <your-aml-workspace-name>
```

## Part 8: Debugging & Troubleshooting

### Problem: Authentication Fails

```powershell
# Clear cached credentials
az logout
az login
az account set --subscription "<your-subscription-id>"
```

### Problem: SSH Connection Timeout

```powershell
# Verify compute instance is running
az ml compute show -n <your-compute-instance-name> -g <your-resource-group> -w <your-aml-workspace-name> --query properties.powerState

# If deallocated, start it:
az ml compute start -n <your-compute-instance-name> -g <your-resource-group> -w <your-aml-workspace-name>

# Wait 2-3 minutes for startup
```

### Problem: Module Import Errors

```powershell
# Ensure .venv is activated
.\.venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install --upgrade -e .

# Verify installation
python -c "import aml_bandits; print(aml_bandits.__file__)"
```

### Problem: Job Fails on Cluster

```powershell
# Check job logs
az ml job logs -n <job_id> -g <your-resource-group> -w <your-aml-workspace-name>

# Check cluster events
az ml compute list -g <your-resource-group> -w <your-aml-workspace-name> -o table

# Verify compute target has required dependencies installed
```

## Part 9: Advanced Configuration

### Custom Environment (Docker)

Create `docker/Dockerfile`:

```dockerfile
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

ENTRYPOINT ["python", "examples/run_bandits.py"]
```

Register environment in Azure ML:

```powershell
az ml environment create --file docker/environment.yaml -g <your-resource-group> -w <your-aml-workspace-name>
```

### Multi-Step Pipeline

Create `pipelines/bandits_pipeline.yaml` to orchestrate multiple jobs.

## Part 10: Next Steps

1. ✅ Run the demo notebook: `notebooks/demo.ipynb`
2. ✅ Submit a job to compute instance: `examples/run_bandits.py`
3. ✅ Scale to Kubernetes: Update job config and submit
4. ✅ Monitor experiments in Azure ML Studio
5. ✅ Set up MLflow tracking for hyperparameter tuning
6. ✅ Deploy policy as REST endpoint

## Quick Reference Commands

```powershell
# Show workspace info
az ml workspace show -g <your-resource-group> -w <your-aml-workspace-name>

# List compute targets
az ml compute list -g <your-resource-group> -w <your-aml-workspace-name> -o table

# List datasets
az ml data list -g <your-resource-group> -w <your-aml-workspace-name>

# Create new compute instance
az ml compute create -f compute_config.yaml -g <your-resource-group> -w <your-aml-workspace-name>

# Submit experiment job
az ml job create --file job_config.yaml -g <your-resource-group> -w <your-aml-workspace-name>

# View job status
az ml job show -n <job_id> -g <your-resource-group> -w <your-aml-workspace-name>

# Stream job logs
az ml job stream -n <job_id> -g <your-resource-group> -w <your-aml-workspace-name>

# Download job outputs
az ml job download -n <job_id> -d ./outputs -g <your-resource-group> -w <your-aml-workspace-name>
```

## Support

For issues or questions:
1. Check [Azure ML Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)
2. Review [Azure ML SDK Samples](https://github.com/Azure/azureml-examples)
3. Open an issue on this repository

---

**Next:** Start with [README.md](./README.md) for project overview, or run `python examples/run_bandits.py` to test locally!
