"""
Multi-armed bandit policy implementations.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class RoundOutcome:
    """Single round outcome from policy."""
    action: int
    conversion: int
    reward: float
    expected_reward: float
    optimal_expected_reward: float
    explored: int


class DeterministicBaselinePolicy:
    """Always select the same fixed arm (no exploration)."""

    def __init__(self, fixed_arm: int, n_arms: int):
        self.fixed_arm = fixed_arm
        self.n_arms = n_arms

    def select_action(self, t: int) -> Tuple[int, int]:
        """Select action and whether it was exploratory (always 0 for baseline)."""
        return self.fixed_arm, 0

    def update(self, action: int, conversion: int) -> None:
        """No learning in deterministic baseline."""
        pass


class UCBPolicy:
    """Upper Confidence Bound: optimism in the face of uncertainty."""

    def __init__(self, n_arms: int, margins_arr: np.ndarray, alpha: float = 1.0):
        """
        Initialize UCB policy.

        Args:
            n_arms: Number of arms
            margins_arr: Expected reward margin per arm
            alpha: Exploration bonus scale factor
        """
        self.n_arms = n_arms
        self.margins = margins_arr
        self.alpha = alpha
        self.successes = np.zeros(n_arms, dtype=float)
        self.failures = np.zeros(n_arms, dtype=float)

    @property
    def counts(self) -> np.ndarray:
        """Total trials per arm."""
        return self.successes + self.failures

    def select_action(self, t: int) -> Tuple[int, int]:
        """
        Select action using UCB formula.

        UCB_a(t) = (successes + 1) / (trials + 2) + alpha * sqrt(log(t+1) / (trials+1))
        """
        counts = self.counts
        total = np.maximum(1.0, counts.sum())
        mean = (self.successes + 1.0) / (counts + 2.0)
        bonus = self.alpha * np.sqrt(np.log(total + 1.0) / (counts + 1.0))
        ucb_score = (mean + bonus) * self.margins

        greedy = int(np.argmax(mean * self.margins))
        action = int(np.argmax(ucb_score))
        explored = int(action != greedy)
        return action, explored

    def update(self, action: int, conversion: int) -> None:
        """Update posterior with observed conversion."""
        if conversion:
            self.successes[action] += 1.0
        else:
            self.failures[action] += 1.0


class ThompsonSamplingPolicy:
    """Bayesian bandits: sample from Beta posterior and pick best sampled arm."""

    def __init__(self, n_arms: int, margins_arr: np.ndarray, rng: np.random.Generator = None):
        """
        Initialize Thompson Sampling policy.

        Args:
            n_arms: Number of arms
            margins_arr: Expected reward margin per arm
            rng: Random number generator (default: numpy's default)
        """
        self.n_arms = n_arms
        self.margins = margins_arr
        self.alpha = np.ones(n_arms, dtype=float)
        self.beta = np.ones(n_arms, dtype=float)
        self.rng = rng if rng is not None else np.random.default_rng(42)

    def select_action(self, t: int) -> Tuple[int, int]:
        """
        Sample θ ~ Beta(α, β) per arm and pick arm with highest expected reward.
        """
        sampled_theta = self.rng.beta(self.alpha, self.beta)
        sampled_reward = sampled_theta * self.margins

        post_mean = self.alpha / (self.alpha + self.beta)
        greedy = int(np.argmax(post_mean * self.margins))
        action = int(np.argmax(sampled_reward))
        explored = int(action != greedy)
        return action, explored

    def update(self, action: int, conversion: int) -> None:
        """Update Beta posterior with observed conversion."""
        self.alpha[action] += conversion
        self.beta[action] += 1 - conversion
