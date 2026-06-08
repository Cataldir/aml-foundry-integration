"""
Utility functions and helpers.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np


def load_env_config() -> dict:
    """Load configuration from .env file."""
    try:
        from dotenv import dotenv_values
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            return dotenv_values(env_path)
    except ImportError:
        pass
    return {}


def get_azure_config() -> dict:
    """Get Azure configuration from environment or .env."""
    config = load_env_config()
    return {
        "subscription_id": config.get("SUBSCRIPTION_ID") or os.getenv("SUBSCRIPTION_ID"),
        "resource_group": config.get("RESOURCE_GROUP") or os.getenv("RESOURCE_GROUP"),
        "workspace_name": config.get("WORKSPACE_NAME") or os.getenv("WORKSPACE_NAME"),
        "compute_instance": config.get("COMPUTE_INSTANCE_NAME") or os.getenv("COMPUTE_INSTANCE_NAME"),
        "kubernetes_cluster": config.get("KUBERNETES_CLUSTER_NAME") or os.getenv("KUBERNETES_CLUSTER_NAME"),
    }


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def create_output_dir(base_dir: Optional[str] = None) -> Path:
    """Create output directory for results."""
    if base_dir is None:
        base_dir = "results"
    output_dir = Path(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
