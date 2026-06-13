import sys
import csv
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import hackathon_simulation as hs
from SAiFE_gym import challenge


VALID_AGENT_SOURCE = """
import numpy as np

class Agent:
    def __init__(self, config):
        self.tau = config.tau

    def get_action(self, state):
        n = state["sqrt_price"].shape[0]
        lower = np.full(n, -self.tau, dtype=float)
        upper = np.full(n, self.tau, dtype=float)
        hold = np.full(n, -1.0, dtype=float)
        return np.column_stack([lower, upper, hold])
"""


def write_submission(tmp_path: Path, source: str, name: str = "agent.py") -> Path:
    path = tmp_path / name
    path.write_text(source)
    return path


def test_load_submission_agent_accepts_valid_agent(tmp_path):
    path = write_submission(tmp_path, VALID_AGENT_SOURCE)

    agent_class = hs.load_submission_agent(path)
    agent = agent_class(hs.ScenarioConfig(num_trajectories=2).submission_namespace())

    action = agent.get_action({"sqrt_price": np.ones(2)})
    assert action.shape == (2, 3)


def test_load_submission_agent_rejects_missing_agent(tmp_path):
    path = write_submission(tmp_path, "class Strategy:\n    pass\n")

    with pytest.raises(ValueError, match="Agent class"):
        hs.load_submission_agent(path)


def test_load_submission_agent_rejects_oversized_file(tmp_path):
    path = write_submission(tmp_path, "x = 1\n" + ("#" * hs.MAX_SUBMISSION_BYTES))

    with pytest.raises(ValueError, match="exceeds"):
        hs.load_submission_agent(path)


def test_load_submission_agent_rejects_non_numpy_import(tmp_path):
    path = write_submission(tmp_path, "import os\n\nclass Agent:\n    pass\n")

    with pytest.raises(ValueError, match="only import numpy"):
        hs.load_submission_agent(path)


def test_load_submission_agent_rejects_file_io(tmp_path):
    path = write_submission(
        tmp_path,
        "import numpy as np\n\nclass Agent:\n"
        "    def get_action(self, state):\n"
        "        return np.load('weights.npy')\n",
    )

    with pytest.raises(ValueError, match="blocked call"):
        hs.load_submission_agent(path)


def test_submission_runtime_getattr_blocks_dunder_access(tmp_path):
    path = write_submission(
        tmp_path,
        "import numpy as np\n\nclass Agent:\n"
        "    def __init__(self, config):\n"
        "        pass\n"
        "    def get_action(self, state):\n"
        "        return getattr((), '__class__')\n",
    )

    agent_class = hs.load_submission_agent(path)
    with pytest.raises(AttributeError, match="dunder"):
        agent_class(hs.ScenarioConfig(num_trajectories=1).submission_namespace()).get_action({})


def test_validate_action_rejects_bad_shape():
    with pytest.raises(ValueError, match="expected"):
        hs.validate_action(np.zeros((2, 2)), expected_rows=2, agent_name="BadAgent")


def test_validate_action_rejects_non_finite_values():
    action = np.zeros((2, 3))
    action[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        hs.validate_action(action, expected_rows=2, agent_name="BadAgent")


def test_official_mode_requires_hidden_seeds(monkeypatch):
    monkeypatch.delenv("HACKATHON_SEEDS", raising=False)

    with pytest.raises(ValueError, match="requires HACKATHON_SEEDS"):
        hs.parse_seeds("official")


def test_leaderboard_csv_does_not_write_seed_values(tmp_path):
    output_path = tmp_path / "leaderboard.csv"
    result = hs.EvaluationResult("agent", "submission", np.array([1.0, 2.0]))

    hs.write_leaderboard_csv([result], output_path, mode="official", seeds=[1001, 1002])

    with output_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert "seeds" not in rows[0]
    assert rows[0]["num_seeds"] == "2"
    assert "1001" not in output_path.read_text()


def test_evaluate_agent_is_deterministic_for_same_seed(tmp_path):
    path = write_submission(tmp_path, VALID_AGENT_SOURCE)
    builder = hs.submission_builders([path])[path.stem]
    config = hs.ScenarioConfig(num_trajectories=2, n_steps=4)

    first = hs.evaluate_agent("agent", "submission", builder, config, seeds=[7])
    second = hs.evaluate_agent("agent", "submission", builder, config, seeds=[7])

    np.testing.assert_allclose(first.pnl, second.pnl)


def test_scenario_config_defaults_match_official_challenge():
    config = challenge.ScenarioConfig()

    assert config.n_steps == 1000
    assert config.num_trajectories == 100
    assert config.tau == 10
    assert config.initial_price == 1000.0
    assert config.initial_wealth == 1000.0
    assert config.gas_cost == 3.0
    assert config.volatility == 0.009
    assert config.alpha1 == (15.0, 15.0)
    assert config.alpha3 == (20000.0, 20000.0)
    assert config.kernel_beta == 0.01
    assert config.kernel_k == 20


def test_official_observation_is_trimmed_and_copied():
    config = challenge.ScenarioConfig(num_trajectories=2, n_steps=2)
    env = challenge.create_environment(config, seed=11)
    state, _ = env.reset()

    obs = challenge.official_observation(state, env)

    assert set(obs) == set(challenge.OFFICIAL_OBSERVATION_KEYS) | {"active_liquidity"}
    assert "liquidity_array" not in obs
    assert "fees_0" not in obs
    assert "fees_1" not in obs
    for key, value in obs.items():
        assert value.shape == (config.num_trajectories,), key
        assert not np.shares_memory(value, state.get(key, np.empty(0)))


def test_run_episode_passes_official_observation_to_agent():
    seen_keys = []

    class AssertingAgent:
        def get_action(self, state):
            seen_keys.append(set(state))
            n = state["sqrt_price"].shape[0]
            return np.column_stack([
                np.full(n, -1.0),
                np.full(n, 1.0),
                np.full(n, -1.0),
            ])

    config = hs.ScenarioConfig(num_trajectories=2, n_steps=2)
    env = hs.create_environment(config, seed=5)

    pnl = hs.run_episode(env, AssertingAgent().get_action, "AssertingAgent")

    assert pnl.shape == (2,)
    assert seen_keys
    assert all(keys == set(challenge.OFFICIAL_OBSERVATION_KEYS) | {"active_liquidity"} for keys in seen_keys)
