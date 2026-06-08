"""
Bandit simulation engine with delayed rewards.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid function with numerical stability."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def build_bandit_environment(
    X_data: pd.DataFrame,
    preproc: ColumnTransformer,
    n_components: int = 12,
    n_arms: int = 4,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build synthetic contextual bandit environment.

    Args:
        X_data: Feature dataframe
        preproc: Fitted column transformer
        n_components: Dimensionality of context vectors
        n_arms: Number of arms/offers
        rng: Random number generator

    Returns:
        Tuple of (contexts, offer_catalog, all_probs, margins)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Reduce feature space to dense contexts
    X_sparse = preproc.fit_transform(X_data)
    max_comp = max(2, min(n_components, X_sparse.shape[1] - 1))
    reducer = TruncatedSVD(n_components=max_comp, random_state=42)
    contexts = reducer.fit_transform(X_sparse)

    # Create offer catalog
    offer_catalog = pd.DataFrame(
        [
            {"arm": 0, "offer_name": "No Incentive", "margin": 1.0},
            {"arm": 1, "offer_name": "Fee Discount", "margin": 1.2},
            {"arm": 2, "offer_name": "Cashback", "margin": 1.4},
            {"arm": 3, "offer_name": "Premium Bundle", "margin": 1.8},
        ]
    )

    margins = offer_catalog["margin"].to_numpy()

    # Generate true arm-context conversion probabilities
    n_rounds, context_dim = contexts.shape
    true_weights = rng.normal(loc=0.0, scale=0.35, size=(n_arms, context_dim))
    true_bias = np.array([-0.6, -0.4, -0.3, -0.2])

    all_logits = contexts @ true_weights.T + true_bias
    all_probs = sigmoid(all_logits)

    return contexts, offer_catalog, all_probs, margins


class BanditSimulator:
    """Online policy simulation with delayed rewards."""

    def __init__(self, probs: np.ndarray, margins: np.ndarray, max_delay: int = 8, seed: int = 42):
        """
        Initialize simulator.

        Args:
            probs: True conversion probabilities (n_rounds, n_arms)
            margins: Arm margins/rewards
            max_delay: Maximum feedback delay in rounds
            seed: Random seed
        """
        self.probs = probs
        self.margins = margins
        self.max_delay = max_delay
        self.rng = np.random.default_rng(seed)

    def run_policy(self, policy_name: str, policy) -> pd.DataFrame:
        """
        Run policy simulation with delayed rewards.

        Args:
            policy_name: Name of policy for reporting
            policy: Policy object with select_action(t) and update(action, conv) methods

        Returns:
            DataFrame with round-level outcomes
        """
        n_rounds, _ = self.probs.shape
        pending: Dict[int, List[Tuple[int, int]]] = {}
        rows = []

        for t in range(n_rounds):
            # Process matured feedback
            matured = pending.pop(t, [])
            for act, conv in matured:
                policy.update(act, conv)

            # Select action
            action, explored = policy.select_action(t)
            p = self.probs[t, action]
            conversion = int(self.rng.random() < p)
            reward = conversion * self.margins[action]

            # Schedule feedback delivery
            delay = int(min(self.rng.geometric(0.35) - 1, self.max_delay))
            update_t = t + delay
            pending.setdefault(update_t, []).append((action, conversion))

            # Compute metrics
            expected_reward = p * self.margins[action]
            optimal_expected = float(np.max(self.probs[t] * self.margins))

            rows.append({
                "round": t,
                "action": action,
                "conversion": conversion,
                "reward": reward,
                "expected_reward": expected_reward,
                "optimal_expected_reward": optimal_expected,
                "explored": explored,
            })

        result = pd.DataFrame(rows)
        result["policy"] = policy_name
        result["instant_regret"] = result["optimal_expected_reward"] - result["expected_reward"]
        result["cum_reward"] = result["reward"].cumsum()
        result["cum_regret"] = result["instant_regret"].cumsum()
        result["cum_conversions"] = result["conversion"].cumsum()
        result["cum_conversion_rate"] = result["cum_conversions"] / (result["round"] + 1)
        result["cum_exploration_share"] = result["explored"].cumsum() / (result["round"] + 1)

        return result
