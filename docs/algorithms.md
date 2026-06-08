# Algorithm Explanations

## Deterministic Baseline

**Definition**: Always select the same fixed arm, regardless of context or feedback.

```
Decision: a_t = a_fixed (e.g., always arm 3)
Update: None
```

**Intuition**:
- Simple, reproducible, easy to reason about
- Ignores individual customer context
- No learning from feedback
- Serves as lower bound for adaptive policies

**Pros**:
- Predictable, auditable
- Easy to understand
- Useful for regulatory compliance

**Cons**:
- Suboptimal for heterogeneous customers
- Cannot adapt to new information
- High opportunity cost

## Upper Confidence Bound (UCB)

**Definition**: Select arm with highest "upper confidence bound" combining exploitation and exploration.

```
UCB_a(t) = μ̂_a(t) + α * sqrt(log(T) / N_a(t))
         = estimated_mean + exploration_bonus

Where:
  μ̂_a(t) = empirical mean reward of arm a up to time t
  N_a(t) = number of times arm a has been played
  α     = exploration scale parameter (tunable)
  T     = total time steps so far
```

**Intuition**:
- **Exploitation**: Choose arm with highest estimated reward
- **Exploration**: Add bonus for uncertain arms (fewer trials → higher bonus)
- Balances by being optimistic about unexplored options
- Arms with high variance get explored more (uncertainty principle)

**Algorithm**:
1. Initialize success/failure counters for all arms
2. At each time step:
   - Compute UCB score for each arm
   - Select arm with highest score
   - Observe reward and feedback
   - Update success/failure counters
3. Repeat

**Example**:
- Arm 1: 10/20 successes (50%), 100 trials
- Arm 2: 5/10 successes (50%), 10 trials
- With uncertainty bonus, Arm 2 gets higher UCB because we're less confident
- Arm 2 will be explored more until we gather enough evidence

**Pros**:
- Regret-optimal for known problem class (1/α ~ log(T) regret)
- Principled exploration via confidence intervals
- Well-understood theoretically
- No Bayesian computation needed

**Cons**:
- Requires tuning `α` parameter
- Can be overly exploratory if α too large
- Doesn't leverage prior beliefs

## Thompson Sampling (Bayesian Bandits)

**Definition**: Maintain Bayesian posteriors over arm rewards; sample plausible reward for each arm and pick best sample.

```
For each arm a:
  θ_a ~ Beta(α_a, β_a)  [sample from posterior]
  R̃_a = θ_a * margin_a  [sampled expected reward]

Decision: a_t = argmax_a R̃_a

Update (after observing reward r_a):
  If r_a > 0:
    α_a += 1  [increment successes]
  β_a += 1    [increment trials]
```

**Intuition**:
- Treat arm rewards as random variables with unknown probabilities
- Use Beta-Binomial conjugate prior (natural for Bernoulli outcomes)
- Posterior distribution captures our belief about true reward probability
- Sample from posterior: uncertain arms naturally get "lucky" samples sometimes
- This creates automatic exploration without explicit bonuses

**Algorithm**:
1. Initialize Beta(1, 1) for each arm [uniform prior = maximum uncertainty]
2. At each time step:
   - For each arm: sample θ_a ~ Beta(α_a, β_a)
   - Compute expected reward: μ_a = θ_a * margin_a
   - Select arm: a_t = argmax_a μ_a
3. Observe conversion feedback
4. Update posterior:
   - If converted: α_{a_t} += 1
   - Always: β_{a_t} += 1
5. Repeat

**Example**:
- Arm 1: 100 successes, 100 failures → Beta(101, 101) [high confidence, mean ≈ 0.5]
- Arm 2: 1 success, 1 failure → Beta(2, 2) [low confidence, wide variance]
- Sample from Arm 1: might be 0.48-0.52 (tight)
- Sample from Arm 2: might be 0.1, 0.9, or anywhere (wide)
- Even if Arm 1 has higher empirical mean, Arm 2 gets explored due to lucky samples
- Over time, confident arms rarely get "lucky" samples, so exploration naturally decreases

**Pros**:
- Natural exploration via posterior sampling (no explicit bonus tuning)
- Leverages prior beliefs (Beta(1,1) = uniform, but can use informative priors)
- Empirically competitive with or beats UCB
- Elegant Bayesian interpretation
- Extends naturally to contextual/hierarchical settings

**Cons**:
- More computation (sampling and posterior updates)
- Requires Bayesian modeling (slightly more abstract)
- Needs specification of prior distribution

## Comparison

| Aspect | Baseline | UCB | Thompson Sampling |
|--------|----------|-----|-------------------|
| **Learning** | None | Empirical | Bayesian posterior |
| **Exploration** | None | Uncertainty bonus | Posterior sampling |
| **Tuning** | None | α parameter | Prior parameters |
| **Regret** | Ω(T) | O(log T) | O(log T) |
| **Computation** | O(1) | O(K log T) | O(K) + sampling |
| **Interpretation** | Fixed rule | Optimism | Posterior beliefs |

## Delayed Rewards

Real-world campaigns often have feedback delays. This implementation:

1. **Tracks scheduled updates**: When will feedback arrive?
2. **Processes matured feedback**: At each round, apply updates from feedback that just arrived
3. **Metrics by decision time**: Regret and reward tracked by when decision was made, not when feedback arrived

This ensures realistic learning dynamics where policies operate with stale information.

## References

- Lattimore & Szepesvári (2020): *Bandit Algorithms* — Comprehensive theory
- Bubeck & Cesa-Bianchi (2012): *Regret Analysis* — Foundational regret bounds
- Russo & Van Roy (2014): *Learning to Optimize Via Posterior Sampling* — Thompson Sampling
- Agrawal & Goyal (2012): *Thompson Sampling for Contextual Bandits* — Modern variants
