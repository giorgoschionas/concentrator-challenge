"""Official hackathon challenge scenario and participant-facing contract."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.index_names import (
    ASSET_PRICE_KEY,
    GAS_COST_KEY,
    LP_EVER_DEPLOYED_KEY,
    LP_TICK_LOWER_KEY,
    LP_TICK_UPPER_KEY,
    POOL_CURRENT_TICK_KEY,
    POOL_SQRT_PRICE_KEY,
    PORTFOLIO_VALUE_KEY,
    TIME_KEY,
)
from SAiFE_gym.rewards.RewardFunctions import PnL
from SAiFE_gym.stochastic_processes.arrival_models import LiquidityKernelArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import GeometricBrownianMotionMidpriceModel


ACTIVE_LIQUIDITY_KEY = "active_liquidity"

OFFICIAL_OBSERVATION_KEYS = (
    POOL_SQRT_PRICE_KEY,
    POOL_CURRENT_TICK_KEY,
    ASSET_PRICE_KEY,
    TIME_KEY,
    LP_TICK_LOWER_KEY,
    LP_TICK_UPPER_KEY,
    LP_EVER_DEPLOYED_KEY,
    GAS_COST_KEY,
    PORTFOLIO_VALUE_KEY,
)


@dataclass(frozen=True)
class ScenarioConfig:
    terminal_time: float = 1.0
    n_steps: int = 1000
    num_trajectories: int = 100
    initial_wealth: float = 1000.0
    tau: int = 10
    num_ticks: int = 5000
    initial_price: float = 1000.0
    initial_pool_price: float | None = None
    drift: float = 0.0
    volatility: float = 0.009
    fee_tier: float = 0.003
    exponential_value: float = 1.0001
    gas_cost: float = 3.0
    swap_fee_rate: float = 0.0
    liquidity_scale: float = 1e5
    alpha0: tuple[float, float] = (1.0, 1.0)
    alpha1: tuple[float, float] = (15.0, 15.0)
    alpha2: tuple[float, float] = (0.0, 0.0)
    alpha3: tuple[float, float] = (20000.0, 20000.0)
    kernel_beta: float = 0.01
    kernel_k: int = 20

    @property
    def step_size(self) -> float:
        return self.terminal_time / self.n_steps

    def submission_namespace(self) -> SimpleNamespace:
        """Return the scalar config namespace passed to submissions."""
        return SimpleNamespace(
            num_trajectories=self.num_trajectories,
            n_steps=self.n_steps,
            terminal_time=self.terminal_time,
            step_size=self.step_size,
            initial_wealth=self.initial_wealth,
            tau=self.tau,
            num_ticks=self.num_ticks,
            exponential_value=self.exponential_value,
            fee_tier=self.fee_tier,
            gas_cost=self.gas_cost,
            swap_fee_rate=self.swap_fee_rate,
            drift=self.drift,
            volatility=self.volatility,
        )


class _ChallengeAMMEnvironment(AMMEnvironment):
    """Official PnL environment with a minimal previous-state snapshot."""

    def step(self, action: np.ndarray):
        current_state = {
            PORTFOLIO_VALUE_KEY: self.model_dynamics.state[PORTFOLIO_VALUE_KEY].copy()
        }
        next_state = self._update_state(action)
        terminated = self._get_terminated()
        truncated = np.zeros(self.num_trajectories, dtype=bool)
        rewards = self.reward_function.calculate(current_state, action, next_state, terminated[0])
        info = self._calculate_infos(current_state, action, rewards)
        return next_state, rewards, terminated, truncated, info


def submission_config(config: ScenarioConfig) -> SimpleNamespace:
    """Compatibility helper for code that wants the participant config namespace."""
    return config.submission_namespace()


def create_environment(config: ScenarioConfig, seed: int) -> AMMEnvironment:
    """Build the fixed official challenge environment."""
    alpha = np.array([config.alpha0, config.alpha1, config.alpha2, config.alpha3])
    midprice_model = GeometricBrownianMotionMidpriceModel(
        drift=config.drift,
        volatility=config.volatility,
        num_trajectories=config.num_trajectories,
        seed=seed,
        initial_price=config.initial_price,
        terminal_time=config.terminal_time,
        step_size=config.step_size,
    )
    arrival_model = LiquidityKernelArrivalModel(
        alpha=alpha,
        beta=config.kernel_beta,
        K=config.kernel_k,
        liquidity_scale=config.liquidity_scale,
        step_size=config.step_size,
        num_trajectories=config.num_trajectories,
        seed=seed + 1,
    )
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        num_trajectories=config.num_trajectories,
        fee_tier=config.fee_tier,
        tau=config.tau,
        num_ticks=config.num_ticks,
        exponential_value=config.exponential_value,
        gas_cost=config.gas_cost,
        swap_fee_rate=config.swap_fee_rate,
        seed=seed + 2,
    )
    return _ChallengeAMMEnvironment(
        terminal_time=config.terminal_time,
        n_steps=config.n_steps,
        initial_wealth=config.initial_wealth,
        reward_function=PnL(),
        model_dynamics=model_dynamics,
        num_trajectories=config.num_trajectories,
        initial_pool_price=config.initial_pool_price,
        seed=seed,
    )


def official_observation(state: dict[str, np.ndarray], env: AMMEnvironment) -> dict[str, np.ndarray]:
    """Return the participant-visible observation with copied scalar arrays only."""
    observation = {key: np.asarray(state[key]).copy() for key in OFFICIAL_OBSERVATION_KEYS}
    _, active_liquidity = env.model_dynamics._get_current_tick_liquidity()
    observation[ACTIVE_LIQUIDITY_KEY] = np.asarray(active_liquidity).copy()
    return observation


def validate_action(action: Any, num_trajectories: int) -> tuple[np.ndarray | None, str]:
    """Materialize and sanity-check an Agent action without raising."""
    try:
        array = np.asarray(action, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        return None, f"action must be numeric: {exc}"

    expected_shape = (num_trajectories, 3)
    if array.shape != expected_shape:
        return None, f"expected action shape {expected_shape}, got {array.shape}"
    if not np.all(np.isfinite(array)):
        return None, "action contains non-finite values"

    return array, "ok"
