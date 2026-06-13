import gymnasium
import numpy as np
from SAiFE_gym.agents.Agent import Agent
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.index_names import LP_EVER_DEPLOYED_KEY, TIME_KEY


class RandomAgent(Agent):
    """
    Randomly samples LP position bounds uniformly from [-tau, tau].

    Uses order statistics: samples two points, sorts them to ensure lower < upper.
    This guarantees valid actions where lower_offset < upper_offset.
    """
    def __init__(self, env: gymnasium.Env, seed: int = None):
        self.tau = env.model_dynamics.tau
        self.num_trajectories = env.num_trajectories
        self.rng = np.random.default_rng(seed)

    def get_action(self, state: dict) -> np.ndarray:
        # Sample two points uniformly from [-tau, tau] for each trajectory
        # Shape: (num_trajectories, 2)
        samples = self.rng.uniform(-self.tau, self.tau, size=(self.num_trajectories, 2))

        # Sort along axis=1 so that [:, 0] < [:, 1]
        actions = np.sort(samples, axis=1)

        # Clip to valid action space bounds: lower ∈ [-tau, tau-1], upper ∈ [-tau+1, tau]
        actions[:, 0] = np.clip(actions[:, 0], -self.tau, self.tau - 1)
        actions[:, 1] = np.clip(actions[:, 1], -self.tau + 1, self.tau)

        # Ensure minimum width of 1 tick (lower < upper)
        too_close = actions[:, 1] <= actions[:, 0]
        actions[too_close, 1] = actions[too_close, 0] + 1

        # Append hold_flag = -1.0 (always rebalance)
        hold_col = np.full((self.num_trajectories, 1), -1.0, dtype=np.float32)
        return np.concatenate([actions.astype(np.float32), hold_col], axis=1)


class DeployOnceWideAgent(Agent):
    """
    Deploys liquidity once in a wide fixed tick range and holds for the episode.

    On the first step (LP not yet deployed), emits hold_flag = -1 to trigger
    deployment at [lower_offset, upper_offset]. On all subsequent steps, emits
    hold_flag = +1 to hold the existing position without rebalancing.

    Defaults to the symmetric widest action range [-tau, +tau]. Pass
    `lower_offset` / `upper_offset` to quote asymmetrically, e.g. [-3, +7].
    """
    def __init__(self, env: AMMEnvironment,
                 lower_offset: int = None, upper_offset: int = None):
        self.env = env
        tau = env.model_dynamics.tau
        self.lower_offset = -tau if lower_offset is None else int(lower_offset)
        self.upper_offset = tau if upper_offset is None else int(upper_offset)
        assert -tau <= self.lower_offset < self.upper_offset <= tau, (
            f"need -tau <= lower_offset < upper_offset <= tau, got "
            f"[{self.lower_offset}, {self.upper_offset}] with tau={tau}"
        )

    def get_action(self, state: dict) -> np.ndarray:
        n = self.env.num_trajectories
        lower = np.full(n, self.lower_offset, dtype=np.float32)
        upper = np.full(n, self.upper_offset, dtype=np.float32)
        ever_deployed = state[LP_EVER_DEPLOYED_KEY]
        hold_flag = np.where(ever_deployed, 1.0, -1.0).astype(np.float32)
        return np.column_stack([lower, upper, hold_flag])


class ArrivalRebalanceAgent(Agent):
    """
    Quotes a fixed tick range around the current price; rebalances after
    every ``rebalance_every`` liquidity-taking arrivals (sell or buy).

    Range:
      - Symmetric (default): ``width=W`` → ``[-W, +W]``.
      - Asymmetric: pass ``lower_offset`` and ``upper_offset`` to quote
        e.g. ``[-3, +7]``. When both are provided they take precedence
        over ``width``.

    Reads the arrivals that produced the *current* state directly from
    ``env.model_dynamics.last_arrivals`` (cached after each get_arrivals
    call), so the count is exact — no fee-delta or |Δtick| approximation.
    """
    def __init__(self, env: AMMEnvironment, rebalance_every: int = 10,
                 width: int = 2,
                 lower_offset: int = None, upper_offset: int = None):
        assert rebalance_every >= 1, f"rebalance_every must be >= 1, got {rebalance_every}"
        tau = env.model_dynamics.tau

        if lower_offset is not None or upper_offset is not None:
            assert lower_offset is not None and upper_offset is not None, (
                "pass both lower_offset and upper_offset, or neither"
            )
            self.lower_offset = int(lower_offset)
            self.upper_offset = int(upper_offset)
        else:
            assert 1 <= width <= tau, f"width must be in [1, tau={tau}], got {width}"
            self.lower_offset = -int(width)
            self.upper_offset = int(width)

        assert -tau <= self.lower_offset < self.upper_offset <= tau, (
            f"need -tau <= lower_offset < upper_offset <= tau, got "
            f"[{self.lower_offset}, {self.upper_offset}] with tau={tau}"
        )

        self.env = env
        self.rebalance_every = rebalance_every
        self.arrival_count = np.zeros(env.num_trajectories, dtype=np.int64)

    def get_action(self, state: dict) -> np.ndarray:
        n = self.env.num_trajectories
        is_episode_start = state[TIME_KEY][0] < self.env.step_size / 2

        if is_episode_start:
            self.arrival_count = np.zeros(n, dtype=np.int64)
            rebalance = np.ones(n, dtype=bool)  # initial deploy
        else:
            new_arrivals = self.env.model_dynamics.last_arrivals.sum(axis=1).astype(np.int64)
            self.arrival_count += new_arrivals
            rebalance = self.arrival_count >= self.rebalance_every
            self.arrival_count = np.where(rebalance, 0, self.arrival_count)

        lower = np.full(n, self.lower_offset, dtype=np.float32)
        upper = np.full(n, self.upper_offset, dtype=np.float32)
        hold_flag = np.where(rebalance, -1.0, 1.0).astype(np.float32)
        return np.column_stack([lower, upper, hold_flag])


class HoldToken0Agent(Agent):
    """
    Hold-token0 baseline: holds the initial wealth as the risky asset
    (token0) for the whole episode, so portfolio value marks to market with
    the external price:

        wealth_t  ≈  initial_wealth · (price_t / price_0)

    Mechanism: on the first step, deploys a one-tick range just above the
    current price (``[0, 1]``), then emits ``hold_flag = +1`` forever. As
    long as pool price stays below the position's lower bound, the LP sits at
    100% token0 with no swaps and no fees, so ``get_position_value_vec`` evaluates to
    ``W · external_price_t / external_price_0`` and the env's `PnL` reward
    produces the HODL-token0 path automatically.

    Caveat: if pool price crosses into the deployed one-tick range, the LP
    starts earning fees and converting to token1, so MTM can diverge from
    pure HODL.

    Set ``hold_cash=True`` for the alternative cash baseline: never deploy,
    portfolio value stays at ``initial_wealth`` (token1 is the numeraire
    so its value doesn't move), cumulative PnL ≡ 0.
    """
    def __init__(self, env: AMMEnvironment, hold_cash: bool = False):
        self.env = env
        self.hold_cash = hold_cash
        # Tightest "above current price" range available; this minimises the
        # chance of price crossing into the range during the episode.
        self.lower_offset = 0
        self.upper_offset = 1

    def get_action(self, state: dict) -> np.ndarray:
        n = self.env.num_trajectories
        if self.hold_cash:
            # Never deploy → portfolio_value = initial_wealth (constant).
            lower = np.zeros(n, dtype=np.float32)
            upper = np.ones(n, dtype=np.float32)
            hold_flag = np.ones(n, dtype=np.float32)
        else:
            # Deploy once just above current price -> 100% token0 -> MTM with price.
            lower = np.full(n, self.lower_offset, dtype=np.float32)
            upper = np.full(n, self.upper_offset, dtype=np.float32)
            ever_deployed = state[LP_EVER_DEPLOYED_KEY]
            hold_flag = np.where(ever_deployed, 1.0, -1.0).astype(np.float32)
        return np.column_stack([lower, upper, hold_flag])


class FullRangeRebalanceAgent(Agent):
    """
    Rebalances into the full active tick range around the current price.

    Action format: [lower_offset, upper_offset, hold_flag] = [-tau, +tau, -1.0]
    This spans 2*tau ticks centered on the current price. Always rebalances.
    """
    def __init__(self, env: AMMEnvironment):
        self.env = env
        self.tau = env.model_dynamics.tau


    def get_action(self, state: dict) -> np.ndarray:
        # Full-range band [-tau, +tau] around the current tick, always rebalance.
        action = np.array([[-self.tau, self.tau, -1.0]])
        return np.repeat(action, self.env.num_trajectories, axis=0)


# Backward-compatible aliases for older examples and notebooks.
DeployOnceAgent = DeployOnceWideAgent
DoNothingAgent = HoldToken0Agent
UniformAllocationAgent = FullRangeRebalanceAgent
