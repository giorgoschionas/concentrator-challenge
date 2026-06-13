# The Concentrator

A liquidity-provision (LP) trading challenge built on **SAiFE_gym**, a vectorized, Gymnasium-compatible simulation of a Uniswap-V3-style Automated
Market Maker (AMM) with concentrated liquidity.

You write a trading strategy as a small Python `Agent` class. We run it against a
fixed, stochastic AMM market and rank it on a leaderboard by **profit and loss
(PnL)**. Beat the baselines, then beat everyone else.

## Official challenge flow

This repository is the **official challenge interface**. There is no required
website or hosted submission system for this edition of the hackathon:

1. Participants clone this repo.
2. Participants copy [`submission_template.py`](submission_template.py) and build
   one Python `Agent` file locally.
3. Participants use `experiments/hackathon_simulation.py` with public practice
   seeds to test their strategy against the official scenario and baselines.
4. Participants submit exactly one Python source file to the organizers by the
   announced channel.
5. Organizers run the same evaluation harness in `--mode official` with hidden
   seeds and publish the resulting leaderboard.

Only the submitted source file is graded. Local notebooks, model checkpoints,
training scripts, and extra data files are not accepted by the grader.

## The challenge in a nutshell

In Uniswap V3 you don't just deposit capital — you choose a **price range** (a
tick band) to concentrate your liquidity in. That single choice creates the whole
game:

- **Concentrate tightly** → earn more fees per dollar *while the price stays in
  your band* — but the moment price leaves the band you earn nothing and are left
  holding the wrong mix of assets.
- **Quote wide** → almost always in range, but your fees-per-dollar are diluted.
- **Chase the price by rebalancing** → you re-center your band to keep earning,
  but every rebalance costs **gas**.
- **Beware toxic flow** → the market contains **arbitrageurs** who trade against
  stale, mispriced positions. Sit still while the price moves and you get picked
  off (the classic *impermanent loss* / *loss-versus-rebalancing* problem).

Your agent decides, every step, **where** to place its band and **whether it is
worth paying gas to move it** — trading fee income against gas and adverse
selection. Highest mean PnL wins.

## Quick start

```bash
git clone https://github.com/giorgoschionas/concentrator-challenge.git
cd concentrator-challenge
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # Python 3.11+ (full sim + training stack)
```

1. **Copy the starter.** [`submission_template.py`](submission_template.py) is a
   complete, runnable pure-numpy agent that documents the entire contract.

   ```bash
   cp submission_template.py my_agent.py
   ```

2. **Edit `get_action`.** Implement your strategy (keep it numpy-only and
   vectorized — see the contract below).

3. **Score it locally** against the official scenario and the baselines using
   the public practice seeds:

   ```bash
   python experiments/hackathon_simulation.py --agent my_agent.py
   ```

   ```
   Rank | Agent              | Kind       |       Mean |  Std |  Median |    Min |    Max | Profitable
   --------------------------------------------------------------------------------------------------
      1 | my_agent           | submission |     +12.40 | 9.10 |  +10.05 | -14.21 | +38.6  |      78.0%   <- illustrative
      2 | HoldToken0         | baseline   |      +7.91 | 9.93 |   +6.59 | -12.29 | +36.30 |      83.3%
      3 | DeployOnceWide     | baseline   |      +5.53 | 7.78 |   +5.31 | -13.55 | +26.51 |      78.9%
   ```

   A leaderboard CSV is written to `experiments/results/hackathon_leaderboard.csv`.
   (Baseline numbers above are real; `my_agent` is an illustrative target to beat.)

## The submission contract

Your submission is **one Python file** defining a top-level class named `Agent`
(or any class whose name ends in `Agent`) with two methods:

```python
import numpy as np

class Agent:
    def __init__(self, config):
        # config: a read-only namespace of scalars describing the episode.
        # It is NOT the live environment — you cannot see the RNG, the seed,
        # or any future price. Fields:
        #   num_trajectories, n_steps, terminal_time, step_size, initial_wealth,
        #   tau, num_ticks, exponential_value, fee_tier, gas_cost,
        #   swap_fee_rate, drift, volatility
        self.tau = int(getattr(config, "tau", 10))

    def get_action(self, state):
        # Called once per step. Returns an array of shape (num_trajectories, 3),
        # one row [lower_offset, upper_offset, hold_flag] per trajectory.
        n = state["sqrt_price"].shape[0]
        lower = np.full(n, -self.tau, dtype=np.float64)
        upper = np.full(n,  self.tau, dtype=np.float64)
        hold  = np.full(n, -1.0)              # <= 0 rebalance, > 0 hold
        return np.column_stack([lower, upper, hold])
```

### The action — `[lower_offset, upper_offset, hold_flag]`

`Box(low=[-tau, -tau+1, -1.0], high=[tau-1, tau, 1.0], shape=(3,))`

| Idx | Name | Range | Meaning |
|-----|------|-------|---------|
| `[0]` | `lower_offset` | `[-tau, tau-1]` | Lower band edge, **as a tick offset from the current pool tick** |
| `[1]` | `upper_offset` | `[-tau+1, tau]` | Upper band edge, offset from the current pool tick |
| `[2]` | `hold_flag` | `[-1.0, 1.0]` | `<= 0` → **rebalance** into `[current_tick+lower, current_tick+upper]` (pays gas); `> 0` → **hold** the existing position (no gas this step) |

- Constraint: `lower_offset < upper_offset`. The engine **rounds the two offsets
  to integer ticks and clips them to the box** for you; if `lower >= upper` it
  sets `upper = lower + 1`.
- `hold_flag` is read **by sign only** — keep it within `[-1, 1]` yourself.

### What your agent sees — the `state` dict

`get_action` receives a dict of NumPy arrays, **one row per trajectory**. The
observation is deliberately trimmed to scalar-per-trajectory signals:

| Key | Shape | Description |
|-----|-------|-------------|
| `sqrt_price` | `(num_trajectories,)` | Pool **√price** — pool price `P = sqrt_price ** 2` |
| `current_tick` | `(num_trajectories,)` | Pool's current integer tick |
| `midprice` | `(num_trajectories,)` | External market price (your "fair value") |
| `time` | `(num_trajectories,)` | Elapsed simulation time |
| `lp_tick_lower` | `(num_trajectories,)` | Your position's lower tick bound (absolute) |
| `lp_tick_upper` | `(num_trajectories,)` | Your position's upper tick bound (absolute) |
| `lp_ever_deployed` | `(num_trajectories,)` | `bool` — `False` until you first deploy |
| `gas_cost` | `(num_trajectories,)` | Cost charged per rebalance |
| `portfolio_value` | `(num_trajectories,)` | Your current mark-to-market wealth |
| `active_liquidity` | `(num_trajectories,)` | Liquidity sitting at the current tick |

> **Hidden from you on purpose:** the full per-tick `liquidity_array`, per-tick
> fee accruals, your raw `lp_liquidity`, the RNG/seed, and all future prices. You
> cannot reverse-engineer the market or forge a score.

### Submission rules (checked by the grader)

- **NumPy only.** The grader rejects imports other than `numpy`; `torch`,
  `stable_baselines3`, `tensorflow`, `SAiFE_gym`, `os`, etc. are not allowed.
  Read the state via the raw string keys above.
- **No file I/O.** The grader blocks `open()`, obvious NumPy file-loading calls,
  and common dynamic-execution escape hatches. To ship a *trained* policy,
  **inline its weights as numpy literals** in your file.
- **One file, ≤ 64 KB** of source.
- **Stay vectorized.** Per-step and per-episode time limits apply — operate on
  whole arrays; never loop over trajectories in Python.

This repository-only grader is a challenge contract, not a hardened operating
system sandbox. Organizers should still review submitted files and run official
grading in a disposable local environment.

## The official scenario

Every submission runs in the **same fixed market**, defined by `ScenarioConfig`
in [`SAiFE_gym/challenge.py`](SAiFE_gym/challenge.py):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `num_trajectories` | 100 | Parallel market paths per seed |
| `n_steps` / `terminal_time` | 1000 / 1.0 | Steps per episode / episode horizon |
| `initial_wealth` | 1000.0 | Your starting capital |
| `tau` | 10 | Max band half-width, in ticks (positions span up to ±10) |
| `initial_price` | 1000.0 | Starting price |
| `drift` / `volatility` | 0.0 / 0.009 | Geometric Brownian Motion midprice (driftless, volatile) |
| `fee_tier` | 0.003 | 0.3% pool fee earned by in-range liquidity |
| `gas_cost` | 3.0 | Charged each time you rebalance (0.3% of `initial_wealth`) |
| `num_ticks` | 5000 | Size of the pool's liquidity array |
| `alpha3` | 20000 | **Toxic / arbitrage** flow intensity |
| `kernel_beta` / `kernel_k` | 0.01 / 20 | Liquidity-kernel order-flow shape |

**The market model.** The external midprice follows a driftless **Geometric
Brownian Motion**. Order flow is generated by a **liquidity-kernel arrival
model**: at each step there is at most one buy and one sell arrival (Bernoulli),
and the arrival *intensity* on each side is

```
intensity = max(alpha0, alpha1 + alpha2 · nearby_liquidity + alpha3 · mispricing_gap)
```

In the official scenario `alpha2 = 0` (the thick-market term is off), so flow is
**baseline noise (`alpha1 = 15`) plus one-sided arbitrage**: whenever the pool
price lags the external midprice, the large `alpha3` term floods the profitable
side with arbitrageurs who push the pool back toward fair value — at your expense
if your liquidity is sitting in the way. This is what makes the scenario *toxic*:
fees are there to be earned, but a stale position is actively punished.

## Scoring & the leaderboard

- Each episode produces a **final PnL per trajectory** — the sum of per-step
  rewards, where reward is the change in your `portfolio_value` (mark-to-market).
- Your rank is your **mean PnL across all trajectories, pooled over all seeds**.
  The board also reports std, median, min, max, and **% of paths profitable**.
- **Practice seeds** `[42, 314, 2718]` are public — use them to iterate. The
  **official seeds** are hidden (supplied via the `HACKATHON_SEEDS` environment
  variable at grading time), so you are ranked on **unseen randomness in the same
  scenario** — you can't overfit to specific price paths. Official mode fails if
  `HACKATHON_SEEDS` is unset.

```bash
# Practice (public seeds, the default):
python experiments/hackathon_simulation.py --agent my_agent.py

# Score a whole directory of submissions at once:
python experiments/hackathon_simulation.py --submissions-dir submissions

# Official grading (hidden seeds):
HACKATHON_SEEDS=1001,1002,1003 \
  python experiments/hackathon_simulation.py --mode official --agent my_agent.py

# Organizer flow for all submitted files:
HACKATHON_SEEDS=1001,1002,1003 \
  python experiments/hackathon_simulation.py \
    --mode official \
    --submissions-dir submissions \
    --output results/leaderboard.csv

# The CSV reports num_seeds, but never writes the hidden seed values.

# Add the always-rebalance baseline to the board:
python experiments/hackathon_simulation.py --agent my_agent.py --include-full-range
```

## Baselines to beat

Shipped in [`SAiFE_gym/agents/BaselineAgents.py`](SAiFE_gym/agents/BaselineAgents.py):

| Agent | Strategy |
|-------|----------|
| `HoldToken0Agent` | HODL the risky asset — marks to market with price, earns **no fees** |
| `DeployOnceWideAgent` | Deploy a wide band **once**, never rebalance — fees, no gas, but drifts out of range |
| `FullRangeRebalanceAgent` | Recenter `[-tau, +tau]` **every step** — max fees while in range, but pays gas every step |
| `ArrivalRebalanceAgent` | Recenter only after every *N* order arrivals — a fee/gas compromise |
| `RandomAgent` | Samples a random valid band each step — a sanity floor |

The default leaderboard pits you against `HoldToken0` and `DeployOnceWide`; add
`--include-full-range` to also race the always-rebalance agent. Note that the
naive `submission_template.py` heuristic (quote `±tau/2`, rebalance whenever price
nears a band edge) actually **loses to the simple deploy-once baselines** — its
frequent rebalancing is eaten by gas and toxic flow. That gap *is* the challenge:
your job is to earn fees without bleeding them back out in gas and adverse selection.

## Strategy hints

- **Gas is the budget.** At `gas_cost = 3.0` on `initial_wealth = 1000`, ~333
  rebalances over the episode would burn your entire stake. Rebalance *selectively*.
- **Watch the mispricing.** `midprice` vs. `sqrt_price ** 2` tells you how far the
  pool has drifted from fair value — i.e. how exposed you are to the arbitrage flow.
- **Use `hold_flag`.** Holding a still-in-range position costs nothing; only pay
  gas when re-centering actually buys you enough future fees.
- **Width is a dial.** Tighter bands earn more fees per dollar but leave range
  sooner; find the width that survives `volatility = 0.009`.
- **Train offline, ship numpy.** You may train an RL policy (see below), but the
  sandbox only runs numpy — distill the learned policy into numpy literals.

## Architecture

```
SAiFE_gym/
├── SAiFE_gym/
│   ├── challenge.py                  # OFFICIAL scenario config + submission contract
│   ├── agents/
│   │   ├── Agent.py                  # Abstract base class
│   │   ├── BaselineAgents.py         # Hold/DeployOnce/FullRange/Arrival/Random baselines
│   │   └── SbAgent.py                # Wrapper for Stable-Baselines3 models
│   ├── gym/
│   │   ├── AMMEnvironment.py         # Main Gymnasium environment
│   │   ├── ModelDynamics.py          # UniswapV3ModelDynamics (core AMM logic)
│   │   ├── StableBaselinesAMMEnvironment.py  # SB3-compatible flat-obs wrapper
│   │   ├── index_names.py            # State dictionary key constants
│   │   └── helpers/AMM_utils.py      # Tick/price conversion, position value
│   ├── rewards/
│   │   └── RewardFunctions.py        # PnL, ExponentialUtility, RunningInventoryPenalty
│   └── stochastic_processes/
│       ├── midprice_models.py        # Brownian, GBM, Ornstein–Uhlenbeck
│       └── arrival_models.py         # Poisson, PoissonLinear, LiquidityKernel, PoissonNonLinear
├── experiments/
│   ├── hackathon_simulation.py       # The challenge evaluation harness + leaderboard
│   ├── agent_comparison.py           # Baseline comparison plots
│   └── helpers.py                    # Env factory, SB3 wrap, PPO trainer, rollout helper
├── tests/                            # pytest regression suite
└── submission_template.py            # Pure-numpy Agent starter — copy this
```

Run the test suite with `pytest`. SAiFE_gym is **fully vectorized**: every core
component batches over `num_trajectories`, so prefer NumPy array ops (`np.where`,
`np.clip`, masks) over Python loops everywhere.

## Advanced: training an RL policy

The environment is a standard Gymnasium env, so you can train with
Stable-Baselines3. Remember the sandbox runs **numpy only** — you must export the
trained policy's weights and reimplement the forward pass in numpy in your
submission file.

```python
from experiments.helpers import get_amm_env, get_ppo_learner_and_callback

env = get_amm_env(num_trajectories=50, tau=10, volatility=0.009)
model, callback = get_ppo_learner_and_callback(env, normalise_obs=True)
model.learn(total_timesteps=2_000_000, callback=callback)
```

To experiment against the *exact* official market instead of the generic helper,
build it directly:

```python
from SAiFE_gym.challenge import ScenarioConfig, create_environment

env = create_environment(ScenarioConfig(), seed=42)
obs, _ = env.reset()
```

## Dependencies

The challenge harness and training stack need `gymnasium`, `numpy`, `pandas`,
`matplotlib`, `stable-baselines3`, and `torch` (see `requirements.txt`). **Your
submission itself only ever gets `numpy`.**

## License

Licensed under either of

- MIT license ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
