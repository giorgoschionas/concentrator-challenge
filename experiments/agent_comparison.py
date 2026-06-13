"""
Compare baseline liquidity-provision agents.

The script evaluates deterministic baseline agents over shared market
trajectories, prints a compact PnL summary, and saves two figures to
experiments/figures/<RUN_TAG>/.

Toggle baseline agents on/off via ENABLE_AGENTS.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np

from SAiFE_gym.agents.BaselineAgents import (
    DeployOnceWideAgent,
    FullRangeRebalanceAgent,
    HoldToken0Agent,
)
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.index_names import (
    ASSET_PRICE_KEY,
    LP_TICK_LOWER_KEY,
    LP_TICK_UPPER_KEY,
    POOL_SQRT_PRICE_KEY,
    TIME_KEY,
)
from SAiFE_gym.rewards.RewardFunctions import PnL
from SAiFE_gym.stochastic_processes.arrival_models import LiquidityKernelArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import GeometricBrownianMotionMidpriceModel, OrnsteinUhlenbeckMidpriceModel


# ============================================================================
# Configuration
# ============================================================================

SEED = 6
TERMINAL_TIME = 1.0
N_STEPS = 1000
NUM_TRAJECTORIES_EVAL = 100
INITIAL_WEALTH = 1000
TAU = int(os.environ.get("TAU", 10))
LIQUIDITY_SCALE = 1e5

INITIAL_PRICE = 1000
INITIAL_POOL_PRICE = None
VOLATILITY = float(os.environ.get("VOLATILITY", 0.009))
FEE_TIER = 0.003
EXP_VALUE = 1.0001
GAS_COST = float(os.environ.get("GAS_COST", 3))

ALPHA0 = np.array([1.0, 1.0])
ALPHA1 = np.array([15.0, 15.0])
ALPHA2 = np.full(2, float(os.environ.get("ALPHA2", 0.0)))
ALPHA3 = np.full(2, float(os.environ.get("ALPHA3", 20000)))

KERNEL_BETA = float(os.environ.get("KERNEL_BETA", 0.01))
KERNEL_K = int(os.environ.get("KERNEL_K", 20))

# DeployOnceWide quote bounds.
DEPLOY_ONCE_WIDE_LOWER = -TAU
DEPLOY_ONCE_WIDE_UPPER = TAU

RUN_TAG = os.environ.get("RUN_TAG", "local")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures", RUN_TAG)
job_id = os.environ.get("SLURM_JOB_ID", "local")

# Agent toggles.
ENABLE_AGENTS = {
    "HoldToken0": True,
    "FullRangeRebalance": False,
    "DeployOnceWide": True,
}

AGENT_NAMES = [name for name, on in ENABLE_AGENTS.items() if on]
AGENT_COLORS = {
    "HoldToken0": "#7f7f7f",
    "FullRangeRebalance": "#1f77b4",
    "DeployOnceWide": "#9467bd",
}


# ============================================================================
# Environment Factory
# ============================================================================


def create_environment(num_trajectories: int, seed: int = None):
    step_size = TERMINAL_TIME / N_STEPS
    alpha = np.array([ALPHA0, ALPHA1, ALPHA2, ALPHA3])

    midprice_model = GeometricBrownianMotionMidpriceModel(
        drift=0.0,
        volatility=VOLATILITY,
        num_trajectories=num_trajectories,
        seed=seed,
        initial_price=INITIAL_PRICE,
        terminal_time=TERMINAL_TIME,
        step_size=step_size,
    )

    arrival_model = LiquidityKernelArrivalModel(
        alpha=alpha,
        beta=KERNEL_BETA,
        K=KERNEL_K,
        liquidity_scale=LIQUIDITY_SCALE,
        step_size=step_size,
        num_trajectories=num_trajectories,
        seed=seed + 1 if seed else None,
    )
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        num_trajectories=num_trajectories,
        fee_tier=FEE_TIER,
        tau=TAU,
        num_ticks=5000,
        exponential_value=EXP_VALUE,
        gas_cost=GAS_COST,
        seed=seed + 2 if seed else None,
    )
    return AMMEnvironment(
        terminal_time=TERMINAL_TIME,
        n_steps=N_STEPS,
        initial_wealth=INITIAL_WEALTH,
        reward_function=PnL(),
        model_dynamics=model_dynamics,
        num_trajectories=num_trajectories,
        initial_pool_price=INITIAL_POOL_PRICE,
        seed=seed,
    )


# ============================================================================
# Agent Helpers
# ============================================================================

def build_agent(name: str, env):
    if name == "HoldToken0":
        return HoldToken0Agent(env)
    if name == "FullRangeRebalance":
        return FullRangeRebalanceAgent(env)
    if name == "DeployOnceWide":
        return DeployOnceWideAgent(
            env,
            lower_offset=DEPLOY_ONCE_WIDE_LOWER,
            upper_offset=DEPLOY_ONCE_WIDE_UPPER,
        )
    raise ValueError(f"Unknown agent: {name}")


# ============================================================================
# Evaluation Helpers
# ============================================================================

def evaluate_on_trajectories(env, get_action_fn):
    """Run one full episode and return per-trajectory cumulative PnL."""
    state, _ = env.reset()
    rewards_list = []
    terminated = np.zeros(env.num_trajectories, dtype=bool)

    while not np.any(terminated):
        action = get_action_fn(state)
        state, reward, terminated, _, _ = env.step(action)
        rewards_list.append(reward)

    return np.sum(np.array(rewards_list), axis=0)


def collect_single_trajectory(env, get_action_fn):
    """Run one episode with one trajectory and return per-step plot data."""
    assert env.num_trajectories == 1
    data = {k: [] for k in [
        "time",
        "pool_price",
        "midprice",
        "position_lower_price",
        "position_upper_price",
        "cumulative_pnl",
    ]}
    state, _ = env.reset()
    cumulative_pnl = 0.0

    for _ in range(env.n_steps):
        data["time"].append(state[TIME_KEY][0])
        data["pool_price"].append(state[POOL_SQRT_PRICE_KEY][0] ** 2)
        data["midprice"].append(state[ASSET_PRICE_KEY][0])

        action = get_action_fn(state)
        state, reward, terminated, _, _ = env.step(action)

        data["position_lower_price"].append(EXP_VALUE ** state[LP_TICK_LOWER_KEY][0])
        data["position_upper_price"].append(EXP_VALUE ** state[LP_TICK_UPPER_KEY][0])
        cumulative_pnl += reward[0]
        data["cumulative_pnl"].append(cumulative_pnl)

        if terminated[0]:
            break

    return {k: np.array(v) for k, v in data.items()}


# ============================================================================
# Plotting
# ============================================================================

def plot_pnl_distribution(pnl_results):
    """Histogram of cumulative PnL."""
    agents = [name for name in AGENT_NAMES if name in pnl_results]
    if not agents:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    all_pnl = np.concatenate([pnl_results[name] for name in agents])
    lo, hi = np.percentile(all_pnl, [1, 99])
    bins = np.linspace(lo, hi, 50)
    for name in agents:
        ax.hist(
            pnl_results[name],
            bins=bins,
            alpha=0.35,
            color=AGENT_COLORS[name],
            density=True,
            label=f"{name} (mean={np.mean(pnl_results[name]):+.1f})",
        )
        ax.axvline(
            np.mean(pnl_results[name]),
            color=AGENT_COLORS[name],
            linestyle="--",
            linewidth=1.5,
        )
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Cumulative PnL", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=14, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"pnl_distribution_{job_id}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_single_trajectory(single_data):
    """Plot one trajectory's prices, LP ranges, and cumulative PnL."""
    agents = [name for name in AGENT_NAMES if name in single_data]
    if not agents:
        return

    fig, (price_ax, pnl_ax) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    first = single_data[agents[0]]
    price_ax.plot(first["time"], first["pool_price"], "k-", linewidth=1.2, label="Pool Price")
    price_ax.plot(first["time"], first["midprice"], color="gray", linewidth=1.0, label="Midprice")

    for name in agents:
        data = single_data[name]
        color = AGENT_COLORS[name]
        price_ax.fill_between(
            data["time"],
            data["position_lower_price"],
            data["position_upper_price"],
            alpha=0.2,
            color=color,
            label=f"{name} Range",
        )
        pnl_ax.plot(
            data["time"],
            data["cumulative_pnl"],
            color=color,
            linewidth=1.8,
            label=name,
        )

    price_ax.set_ylabel("Price", fontsize=14)
    price_ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    price_ax.legend(fontsize=10, loc="upper left")
    price_ax.grid(True, alpha=0.3)

    pnl_ax.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    pnl_ax.set_xlabel("Time", fontsize=14)
    pnl_ax.set_ylabel("Cumulative PnL", fontsize=14)
    pnl_ax.legend(fontsize=10, loc="upper left")
    pnl_ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"single_trajectory_{job_id}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    np.random.seed(SEED)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("Run configuration")
    print("=" * 60)
    print(f"RUN_TAG={RUN_TAG}")
    print("Reward: PnL")

    print("\n" + "=" * 60)
    print(f"Phase 1: Evaluating baseline agents on {NUM_TRAJECTORIES_EVAL} trajectories")
    print("=" * 60)

    eval_seed = SEED + 999
    pnl_results = {}

    for name in AGENT_NAMES:
        env = create_environment(NUM_TRAJECTORIES_EVAL, eval_seed)
        agent = build_agent(name, env)
        pnl_results[name] = evaluate_on_trajectories(env, agent.get_action)

    active = [name for name in AGENT_NAMES if name in pnl_results]
    header = (
        f"{'Agent':<18} | {'Mean PnL':>10} | {'Std':>10} | {'Median':>10} | "
        f"{'Profitable':>12}"
    )
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))
    for name in active:
        pnl = pnl_results[name]
        pct = 100 * np.mean(pnl > 0)
        print(
            f"{name:<18} | {np.mean(pnl):>+10.1f} | {np.std(pnl):>10.1f} "
            f"| {np.median(pnl):>+10.1f} | {np.sum(pnl > 0):>4d}/{len(pnl)} ({pct:.0f}%) "
        )
    print("-" * len(header))

    print("\n" + "=" * 60)
    print("Phase 2: Collecting one single-trajectory example per agent")
    print("=" * 60)

    single_seed = SEED + 7777
    single_data = {}

    for name in AGENT_NAMES:
        env = create_environment(1, single_seed)
        agent = build_agent(name, env)
        single_data[name] = collect_single_trajectory(env, agent.get_action)

    for name in AGENT_NAMES:
        if name in single_data:
            final_pnl = single_data[name]["cumulative_pnl"][-1]
            print(f"  {name:18s} final PnL = {final_pnl:+.1f}")

    print("\n" + "=" * 60)
    print("Phase 3: Generating plots")
    print("=" * 60)

    if pnl_results:
        plot_pnl_distribution(pnl_results)
    if single_data:
        plot_single_trajectory(single_data)

    print("\nDone. All figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
