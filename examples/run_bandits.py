"""
Standalone example runner for AML bandits.
"""

import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aml_bandits import (
    load_bank_marketing_dataset,
    preprocess_data,
    DeterministicBaselinePolicy,
    UCBPolicy,
    ThompsonSamplingPolicy,
    build_bandit_environment,
    BanditSimulator,
    compute_metrics,
    apply_safe_ucb_alpha,
    build_aggregate_metrics,
    validate_and_enrich_strategy,
)
from aml_bandits.utils import set_random_seed, create_output_dir


@click.command()
@click.option(
    "--use-agent/--no-agent",
    default=False,
    help="Validate aggregate metrics with the Foundry bridge or local fallback.",
)
def main(use_agent: bool):
    """Run complete multi-armed bandits pipeline."""
    set_random_seed(42)

    print("=" * 80)
    print("AML Bandits: Multi-Armed Bandits for Adaptive Experimentation")
    print("=" * 80)

    # 1. Load data
    print("\n[1/7] Loading dataset...")
    raw_df, target_col, provenance = load_bank_marketing_dataset()
    print(f"  ✓ Loaded from: {provenance}")
    print(f"  ✓ Shape: {raw_df.shape}")

    # 2. Preprocess
    print("\n[2/7] Preprocessing data (leakage control)...")
    X, y, preprocessor = preprocess_data(raw_df, target_col)
    print(f"  ✓ Features: {X.shape[1]}, Target: {len(y)}")

    # 3. Sanity check with classical ML
    print("\n[3/7] Classical ML sanity check (logistic regression)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("logreg", LogisticRegression(max_iter=1000)),
    ])
    clf.fit(X_train, y_train)
    proba_test = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba_test)
    print(f"  ✓ Validation ROC-AUC: {auc:.4f}")

    # 4. Build bandit environment
    print("\n[4/7] Building synthetic bandit environment...")
    contexts, offer_catalog, all_probs, margins = build_bandit_environment(
        X, preprocessor, n_components=12
    )
    print(f"  ✓ Contexts: {contexts.shape}")
    print(f"  ✓ Arms: {len(offer_catalog)}")
    print(f"  ✓ Margins: {margins}")

    # 5. Create policies
    print("\n[5/7] Instantiating policies...")
    n_arms = len(offer_catalog)
    rng = np.random.default_rng(42)

    baseline_policy = DeterministicBaselinePolicy(fixed_arm=3, n_arms=n_arms)
    ucb_policy = UCBPolicy(n_arms=n_arms, margins_arr=margins, alpha=1.2)
    ts_policy = ThompsonSamplingPolicy(n_arms=n_arms, margins_arr=margins, rng=rng)
    print("  ✓ Baseline, UCB, Thompson Sampling initialized")

    # 6. Run simulations
    print("\n[6/7] Running simulations...")
    simulator = BanditSimulator(all_probs, margins, max_delay=8, seed=100)

    sim_baseline = simulator.run_policy("Deterministic Baseline", baseline_policy)
    print("  ✓ Baseline simulation complete")

    ucb_policy = UCBPolicy(n_arms=n_arms, margins_arr=margins, alpha=1.2)
    sim_ucb = simulator.run_policy("UCB", ucb_policy)
    print("  ✓ UCB simulation complete")

    ts_policy = ThompsonSamplingPolicy(n_arms=n_arms, margins_arr=margins, rng=rng)
    sim_ts = simulator.run_policy("Thompson Sampling", ts_policy)
    print("  ✓ Thompson Sampling simulation complete")

    # 7. Evaluate
    print("\n[7/7] Computing metrics...")
    sim_all = pd.concat([sim_baseline, sim_ucb, sim_ts], ignore_index=True)
    summary = compute_metrics(sim_all)

    if use_agent:
        print("\n[Agent] Validating aggregate evidence only...")
        aggregate = build_aggregate_metrics(summary, current_ucb_alpha=1.2)
        recommendation = validate_and_enrich_strategy(aggregate)
        print(f"  ✓ Source: {recommendation.source}")
        print(f"  ✓ Validation: {recommendation.validation}")
        print(f"  ✓ Rationale: {recommendation.rationale}")
        for warning in recommendation.warnings:
            print(f"  ! {warning}")

        if recommendation.accepted and recommendation.bounded_ucb_alpha is not None:
            recommended_policy = apply_safe_ucb_alpha(
                UCBPolicy(n_arms=n_arms, margins_arr=margins, alpha=1.2),
                recommendation.bounded_ucb_alpha,
            )
            sim_ucb_agent = simulator.run_policy("UCB (Agent Recommended)", recommended_policy)
            sim_all = pd.concat([sim_all, sim_ucb_agent], ignore_index=True)
            summary = compute_metrics(sim_all)
            print(
                "  ✓ Added UCB (Agent Recommended) candidate "
                f"with alpha={recommendation.bounded_ucb_alpha:.3f}"
            )

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False))

    # Save results
    output_dir = create_output_dir("results")
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    sim_all.to_csv(output_dir / "full_results.csv", index=False)
    print(f"\n✓ Results saved to {output_dir}")

    # Generate plots
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for policy_name, grp in sim_all.groupby("policy"):
        axes[0, 0].plot(grp["round"], grp["cum_reward"], label=policy_name)
        axes[0, 1].plot(grp["round"], grp["cum_regret"], label=policy_name)
        axes[1, 0].plot(grp["round"], grp["cum_conversion_rate"], label=policy_name)
        axes[1, 1].plot(grp["round"], grp["cum_exploration_share"], label=policy_name)

    axes[0, 0].set_title("Cumulative Reward")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 1].set_title("Cumulative Regret")
    axes[0, 1].set_ylabel("Regret")
    axes[1, 0].set_title("Running Conversion Rate")
    axes[1, 0].set_ylabel("Conversion Rate")
    axes[1, 1].set_title("Running Exploration Share")
    axes[1, 1].set_ylabel("Exploration Share")

    for ax in axes.ravel():
        ax.set_xlabel("Round")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_dir / "policy_comparison.png", dpi=100, bbox_inches="tight")
    print(f"✓ Plot saved to {output_dir / 'policy_comparison.png'}")

    print("\n" + "=" * 80)
    print("Pipeline complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
