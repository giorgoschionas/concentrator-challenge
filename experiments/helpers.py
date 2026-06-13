"""
Minimal training helpers for SAiFE_gym RL experiments.

AMM liquidity-provision setting:
  - Dict-based observations → StableBaselinesAMMEnvironment flattens them
  - Reward functions: PnL, ExponentialUtility, RunningInventoryPenalty

Provides an environment factory (get_amm_env), SB3 wrapping (wrap_env), a PPO
trainer (get_ppo_learner_and_callback), and a rollout helper (run_episode).
Bring your own Agent for submissions — only simple baselines (RandomAgent,
HoldToken0Agent, FullRangeRebalanceAgent) ship in SAiFE_gym.agents.BaselineAgents.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.StableBaselinesAMMEnvironment import StableBaselinesAMMEnvironment
from SAiFE_gym.rewards.RewardFunctions import PnL, RewardFunction
from SAiFE_gym.stochastic_processes.arrival_models import PoissonLinearArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel


# ---------------------------------------------------------------------------
# Default experiment parameters
# ---------------------------------------------------------------------------

TERMINAL_TIME = 1.0
N_STEPS = 2000
INITIAL_PRICE = 100.0
VOLATILITY = 2.0
FEE_TIER = 0.003
NUM_TICKS = 5000
LIQUIDITY_SCALE = 1e6
INITIAL_WEALTH = 1e6
SEED = 42

# Arrival model coefficients (α shape: (4, 2) for [sell, buy])
#   α₀ = minimum floor, α₁ = baseline, α₂ = liquidity coeff, α₃ = arbitrage (toxicity)
ALPHA0 = np.array([10.0,  10.0])
ALPHA1 = np.array([100.0, 100.0])
ALPHA2 = np.array([0.0,   0.0])   # no liquidity-dependent component by default


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def get_amm_env(
    num_trajectories: int = 1,
    terminal_time: float = TERMINAL_TIME,
    n_steps: int = N_STEPS,
    tau: int = 5,
    volatility: float = VOLATILITY,
    arrival_rate: float = 100.0,
    alpha3: float = 0.0,
    gas_cost: float = 0.0,
    swap_fee_rate: float = 0.0,
    reward_function: RewardFunction = None,
    seed: int = SEED,
) -> AMMEnvironment:
    """Build an AMMEnvironment for LP training / evaluation.

    Args:
        num_trajectories: Parallel trajectories (acts as vectorised batch size).
        terminal_time:    Episode length.
        n_steps:          Number of discrete steps per episode.
        tau:              LP position half-width in ticks (action range ±tau).
        volatility:       Brownian motion volatility of the external mid-price.
        arrival_rate:     Baseline Poisson order arrival rate (α₁ for both sides).
        alpha3:           Arbitrage / toxicity coefficient (α₃). Higher values
                          mean informed traders exploit mispricing more aggressively.
        gas_cost:         Fixed cost per rebalance in token1 units (e.g. 10.0).
                          Applied only when the LP already holds a position.
        swap_fee_rate:    Fee rate on the imbalanced swap amount when rebalancing
                          (e.g. 0.001 = 0.1%). Cost = rate * W * |α_new - α_current|.
        reward_function:  Defaults to PnL.
        seed:             Random seed.

    Returns:
        Configured AMMEnvironment ready for reset / step.
    """
    step_size = terminal_time / n_steps

    alpha = np.array([
        ALPHA0,
        np.array([arrival_rate, arrival_rate]),
        ALPHA2,
        np.array([alpha3, alpha3]),
    ])

    midprice_model = BrownianMotionMidpriceModel(
        drift=0.0,
        volatility=volatility,
        initial_price=INITIAL_PRICE,
        terminal_time=terminal_time,
        step_size=step_size,
        num_trajectories=num_trajectories,
        seed=seed,
    )
    arrival_model = PoissonLinearArrivalModel(
        alpha=alpha,
        liquidity_scale=LIQUIDITY_SCALE,
        step_size=step_size,
        num_trajectories=num_trajectories,
        seed=seed + 1,
    )
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        num_trajectories=num_trajectories,
        fee_tier=FEE_TIER,
        tau=tau,
        num_ticks=NUM_TICKS,
        exponential_value=1.0001,
        gas_cost=gas_cost,
        swap_fee_rate=swap_fee_rate,
        seed=seed + 2,
    )
    return AMMEnvironment(
        terminal_time=terminal_time,
        n_steps=n_steps,
        model_dynamics=model_dynamics,
        reward_function=reward_function or PnL(),
        initial_wealth=INITIAL_WEALTH,
        num_trajectories=num_trajectories,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# SB3 wrapping
# ---------------------------------------------------------------------------

def wrap_env(env: AMMEnvironment, normalise_obs: bool = True) -> VecMonitor:
    """Wrap AMMEnvironment for SB3 training.

    Pipeline: AMMEnvironment → StableBaselinesAMMEnvironment (Dict→flat obs)
              → VecMonitor (episode stats) → VecNormalize (running mean/std).

    VecNormalize is strongly recommended when features span very different
    scales (e.g. time ∈ [0,1] vs lp_liquidity ∈ [0, 2×10⁸]).  It maintains
    a running mean and std for each feature and clips at ±10σ, preventing
    large activations from dominating the first layer's gradients.

    Args:
        normalise_obs: If True (default), wrap with VecNormalize.
                       Disable only when loading a pre-trained model that
                       already carries its own normalisation stats.
    """
    vec = VecMonitor(StableBaselinesAMMEnvironment(env))
    if normalise_obs:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return vec


# ---------------------------------------------------------------------------
# PPO setup
# ---------------------------------------------------------------------------

def get_ppo_learner_and_callback(
    env: AMMEnvironment,
    tensorboard_base_logdir: str = None,
    best_model_path: str = "./best_models",
    tau: int = None,
    alpha3: float = None,
    normalise_obs: bool = True,
    learning_rate: float = 3e-4,
    eval_log_path: str = None,
):
    """Build a PPO model and EvalCallback for the given environment.

    Hyperparameter guidance
    -----------------------
    n_steps:      One full episode per rollout (env.n_steps).  With gamma=1
                  the agent needs complete trajectories to estimate returns.

    batch_size:   Rollout buffer has n_steps * num_trajectories transitions.
                  We split it into 16 mini-batches per epoch so each update
                  uses a representative slice of the collected data.
                  Rule of thumb: aim for 8–16 mini-batches per epoch.

    n_epochs:     10 reuses per rollout is standard.  Reduce to 4–6 if you
                  observe policy loss exploding (use clip_range 0.2 as guard).

    learning_rate: Default 3e-4 works well when obs are normalised (VecNormalize).
                   Without normalisation use 1e-4 to avoid overshooting with
                   large-magnitude raw features (current_tick ~46k, lp_liquidity ~2e8).

    total_timesteps (caller's choice):
                  Scales roughly as O(obs_dim * log(obs_dim)) with state
                  dimension.  Practical guideline for this environment:
                    obs_dim=2  (mbt reduced)  →   500k –   2M
                    obs_dim=9  (SAiFE)        →    2M  –   5M   (with VecNormalize)
                    obs_dim=9  (no normalise) →    5M  –  20M

    Returns:
        (model, callback) — call model.learn(total_timesteps=...) to train.
    """
    tau = tau if tau is not None else env.model_dynamics.tau
    alpha3 = alpha3 if alpha3 is not None else 0.0
    experiment_str = get_experiment_string(env, tau=tau, alpha3=alpha3)

    rollout_size = env.n_steps * env.num_trajectories
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    ppo_params = dict(
        policy="MlpPolicy",
        env=wrap_env(env, normalise_obs=normalise_obs),
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(tensorboard_base_logdir, experiment_str) if tensorboard_base_logdir else None,
        learning_rate=learning_rate,
        n_epochs=10,
        batch_size=max(64, rollout_size // 16),   # 16 mini-batches per epoch
        normalize_advantage=True,
        n_steps=env.n_steps,
        gae_lambda=0.95,
        gamma=1.0,
    )
    eval_vec = wrap_env(env, normalise_obs=normalise_obs)
    if isinstance(eval_vec, VecNormalize):
        eval_vec.training = False  # stats synced from training env; don't update here
    callback_params = dict(
        eval_env=eval_vec,
        n_eval_episodes=10,
        best_model_save_path=os.path.join(best_model_path, experiment_str),
        log_path=eval_log_path,
        deterministic=True,
        eval_freq=rollout_size * 10,
    )
    model = PPO(**ppo_params)
    callback = EvalCallback(**callback_params)
    return model, callback


def get_experiment_string(
    env: AMMEnvironment,
    tau: int = None,
    alpha3: float = None,
    gas_cost: float = None,
    swap_fee_rate: float = None,
) -> str:
    tau = tau if tau is not None else env.model_dynamics.tau
    alpha3 = alpha3 if alpha3 is not None else 0.0
    reward_name = type(env.reward_function).__name__
    s = (
        f"n_traj_{env.num_trajectories}"
        f"__tau_{tau}"
        f"__alpha3_{alpha3}"
        f"__vol_{env.model_dynamics.midprice_model.volatility}"
        f"__reward_{reward_name}"
    )
    if gas_cost is not None and gas_cost > 0:
        s += f"__gas_{gas_cost}"
    if swap_fee_rate is not None and swap_fee_rate > 0:
        s += f"__swapfee_{swap_fee_rate}"
    return s


# ---------------------------------------------------------------------------
# Episode evaluation
# ---------------------------------------------------------------------------

def run_episode(
    env: AMMEnvironment,
    agent,
    obs_transform=None,
    initial_wealth: float = INITIAL_WEALTH,
) -> np.ndarray:
    """Run one full episode and return final wealth per trajectory.

    Args:
        env:            Raw AMMEnvironment (not wrapped).
        agent:          Agent with a get_action(obs) method.
        obs_transform:  Optional callable applied to dict obs before get_action.
                        Pass StableBaselinesAMMEnvironment._flatten_obs for SbAgent.
        initial_wealth: Starting wealth used to compute final wealth from cumulative PnL.

    Returns:
        Final wealth per trajectory, shape (num_trajectories,).
    """
    obs, _ = env.reset()
    cumulative_reward = np.zeros(env.num_trajectories)

    for _ in range(env.n_steps):
        agent_obs = obs_transform(obs) if obs_transform else obs
        action = agent.get_action(agent_obs)
        obs, rewards, terminated, truncated, _ = env.step(action)
        cumulative_reward += rewards

    return initial_wealth + cumulative_reward
