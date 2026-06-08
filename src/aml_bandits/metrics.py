"""
Evaluation metrics and summary statistics.
"""

from typing import Dict

import numpy as np
import pandas as pd


def compute_metrics(sim_results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary metrics by policy.

    Args:
        sim_results: DataFrame with columns from BanditSimulator.run_policy()

    Returns:
        Summary DataFrame with one row per policy
    """
    summary = (
        sim_results.groupby("policy", as_index=False)
        .agg(
            total_reward=("reward", "sum"),
            cumulative_regret=("instant_regret", "sum"),
            conversion_rate=("conversion", "mean"),
            exploration_share=("explored", "mean"),
            mean_reward=("reward", "mean"),
            std_reward=("reward", "std"),
            min_reward=("reward", "min"),
            max_reward=("reward", "max"),
            n_rounds=("round", "count"),
        )
        .sort_values(by="total_reward", ascending=False)
    )
    return summary


def compute_by_window(
    sim_results: pd.DataFrame, window_size: int = 100
) -> Dict[str, pd.DataFrame]:
    """
    Compute metrics in rolling windows (for learning curves).

    Args:
        sim_results: DataFrame with simulation results
        window_size: Size of rolling window in rounds

    Returns:
        Dictionary mapping policy names to windowed metric DataFrames
    """
    results_by_policy = {}
    for policy_name, group in sim_results.groupby("policy"):
        windows = []
        for i in range(0, len(group), window_size):
            window_data = group.iloc[i : i + window_size]
            windows.append({
                "window": i // window_size,
                "round_start": i,
                "round_end": i + len(window_data),
                "mean_reward": window_data["reward"].mean(),
                "std_reward": window_data["reward"].std(),
                "mean_conversion": window_data["conversion"].mean(),
                "mean_exploration": window_data["explored"].mean(),
            })
        results_by_policy[policy_name] = pd.DataFrame(windows)
    return results_by_policy


def compute_regret_by_window(
    sim_results: pd.DataFrame, window_size: int = 100
) -> Dict[str, pd.DataFrame]:
    """
    Compute windowed regret (cumulative opportunity cost).

    Args:
        sim_results: DataFrame with simulation results
        window_size: Size of rolling window

    Returns:
        Dictionary mapping policy names to windowed regret DataFrames
    """
    results_by_policy = {}
    for policy_name, group in sim_results.groupby("policy"):
        windows = []
        for i in range(0, len(group), window_size):
            window_data = group.iloc[i : i + window_size]
            windows.append({
                "window": i // window_size,
                "round_start": i,
                "round_end": i + len(window_data),
                "total_regret": window_data["instant_regret"].sum(),
                "mean_regret": window_data["instant_regret"].mean(),
            })
        results_by_policy[policy_name] = pd.DataFrame(windows)
    return results_by_policy
