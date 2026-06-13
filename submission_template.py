"""
The Concentrator — submission template.

Copy this file, rename nothing (the grader looks for a top-level class named
``Agent``, or any class whose name ends in ``Agent``), and replace the body of
``get_action`` with your own liquidity-provision strategy.

---------------------------------------------------------------------------
GRADER RULES
---------------------------------------------------------------------------
• Your submission is ONE Python file, delivered as source. No companion files.
• The ONLY third-party dependency you can rely on is numpy.
    - The grader rejects imports other than numpy.
    - You also CANNOT import SAiFE_gym; use the raw string state keys below.
• NO file I/O: open() and obvious NumPy file-loading calls are blocked.
    - To ship a *trained* policy, inline its weights as numpy literals in this
      file (the grader cannot load a torch model or any weights file).
• The whole file must be ≤ 64 KB of source.
• Per-step and per-episode time limits apply — keep `get_action` cheap and
  vectorized (operate on whole arrays, never Python loops over trajectories).

---------------------------------------------------------------------------
THE CONTRACT
---------------------------------------------------------------------------
__init__(self, config):
    `config` is a read-only namespace of scalars describing the episode (NOT
    the live environment — you can't see the RNG/seed or the future). Fields:
        num_trajectories, n_steps, terminal_time, step_size, initial_wealth,
        tau, num_ticks, exponential_value, fee_tier, gas_cost, swap_fee_rate,
        drift, volatility
    Access as `config.tau`, `config.gas_cost`, etc. Use getattr fallbacks to stay
    robust if a field is ever absent.

get_action(self, state) -> np.ndarray of shape (num_trajectories, 3):
    Called once per step. `state` is a dict of numpy arrays (one row per
    trajectory). The useful scalar keys:
        "sqrt_price"        √P of the pool  (so pool price P = sqrt_price ** 2)
        "current_tick"      pool's current integer tick
        "midprice"          external market price (your fair value)
        "time"              elapsed simulation time
        "lp_tick_lower"     your position's lower tick bound (absolute)
        "lp_tick_upper"     your position's upper tick bound (absolute)
        "lp_ever_deployed"  bool — False until you first deploy
        "gas_cost"          cost charged per rebalance
        "portfolio_value"   your current mark-to-market wealth
    Return one action per trajectory: [lower_offset, upper_offset, hold_flag]
        lower_offset, upper_offset : tick bounds RELATIVE to current_tick,
            in [-tau, tau]. Must satisfy lower_offset < upper_offset.
            (The engine rounds them to integers and clips to the box for you.)
        hold_flag : <= 0  → rebalance into [current_tick+lower, current_tick+upper]
                    >  0  → hold the existing position (no gas this step)

The engine rounds/clips the two offsets, but it reads hold_flag by SIGN only —
so keep it in [-1, 1] yourself.
"""

import numpy as np


class Agent:
    def __init__(self, config):
        self.tau = int(getattr(config, "tau", 5))
        self.gas_cost = float(getattr(config, "gas_cost", 0.0))
        # Half-width of the band we quote around the current price, in ticks.
        self.width = max(1, self.tau // 2)
        # Rebalance once the price drifts within this many ticks of a band edge.
        self.edge_buffer = 1

    def get_action(self, state):
        n = state["sqrt_price"].shape[0]

        # How much room is left between the current price and each band edge.
        room_below = state["current_tick"] - state["lp_tick_lower"]   # ticks below price
        room_above = state["lp_tick_upper"] - state["current_tick"]   # ticks above price
        near_edge = np.minimum(room_below, room_above) <= self.edge_buffer

        ever_deployed = state.get("lp_ever_deployed", np.zeros(n, dtype=bool))
        rebalance = near_edge | ~ever_deployed   # deploy on step 0, recenter near edges

        # Re-quote a symmetric band ±width around the (new) current tick.
        lower = np.full(n, -self.width, dtype=np.float64)
        upper = np.full(n, self.width, dtype=np.float64)
        hold_flag = np.where(rebalance, -1.0, 1.0)

        return np.column_stack([lower, upper, hold_flag])   # (n, 3)


# ──────────────────────────────────────────────────────────────────────────
# Simplest possible starting point (uncomment to use instead): a full-width
# band that rebalances to re-center on the price every single step. Earns the
# most fees while in range, but pays gas every step — a baseline to beat.
#
# class Agent:
#     def __init__(self, config):
#         self.tau = int(getattr(config, "tau", 5))
#
#     def get_action(self, state):
#         n = state["sqrt_price"].shape[0]
#         lower = np.full(n, -self.tau, dtype=np.float64)
#         upper = np.full(n,  self.tau, dtype=np.float64)
#         hold_flag = np.full(n, -1.0)            # always rebalance
#         return np.column_stack([lower, upper, hold_flag])
