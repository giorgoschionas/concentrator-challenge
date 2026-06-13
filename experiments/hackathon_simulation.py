"""Hackathon evaluation harness for participant LP agents.

Participants submit one Python file defining an ``Agent`` class with the
contract documented in ``submission_template.py``. This script evaluates that
agent in the same toxic, volatile scenario used for baseline comparisons and
writes a leaderboard CSV.

Examples:
    python experiments/hackathon_simulation.py --agent submission_template.py
    python experiments/hackathon_simulation.py --submissions-dir submissions
    HACKATHON_SEEDS=1001,1002 python experiments/hackathon_simulation.py --mode official --agent my_agent.py
"""

from __future__ import annotations

import argparse
import ast
import builtins
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from SAiFE_gym.agents.BaselineAgents import (  # noqa: E402
    DeployOnceWideAgent,
    FullRangeRebalanceAgent,
    HoldToken0Agent,
)
from SAiFE_gym.challenge import (  # noqa: E402
    ScenarioConfig,
    create_environment,
    official_observation,
    validate_action as validate_action_tuple,
)
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment  # noqa: E402


MAX_SUBMISSION_BYTES = 64 * 1024
PRACTICE_SEEDS = [42, 314, 2718]
DEFAULT_OUTPUT = Path("experiments/results/hackathon_leaderboard.csv")
ALLOWED_IMPORT_ROOTS = {"numpy"}
_MISSING = object()
BLOCKED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "fromfile",
    "fromregex",
    "genfromtxt",
    "globals",
    "help",
    "input",
    "load",
    "loadtxt",
    "locals",
    "memmap",
    "memoryview",
    "open",
    "open_memmap",
    "save",
    "savetxt",
    "savez",
    "savez_compressed",
    "vars",
}
SAFE_BUILTINS = {
    "__build_class__": builtins.__build_class__,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "object": object,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "zip": zip,
}


@dataclass
class EvaluationResult:
    name: str
    kind: str
    pnl: np.ndarray

    @property
    def mean_pnl(self) -> float:
        return float(np.mean(self.pnl))

    @property
    def std_pnl(self) -> float:
        return float(np.std(self.pnl))

    @property
    def median_pnl(self) -> float:
        return float(np.median(self.pnl))

    @property
    def min_pnl(self) -> float:
        return float(np.min(self.pnl))

    @property
    def max_pnl(self) -> float:
        return float(np.max(self.pnl))

    @property
    def profitable_pct(self) -> float:
        return float(100.0 * np.mean(self.pnl > 0))


def load_submission_agent(path: Path) -> type:
    """Load an Agent class from a single-file submission."""
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"submission file does not exist: {path}")
    if path.stat().st_size > MAX_SUBMISSION_BYTES:
        raise ValueError(f"submission exceeds {MAX_SUBMISSION_BYTES} bytes: {path}")

    source = path.read_text(encoding="utf-8")
    validate_submission_source(source, path)

    module_name = f"hackathon_submission_{abs(hash(path))}"
    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins(),
        "__file__": str(path),
        "__name__": module_name,
    }
    code = compile(source, str(path), "exec")
    exec(code, namespace)

    agent_class = namespace.get("Agent")
    if agent_class is None:
        candidates = [
            value for name, value in namespace.items()
            if name.endswith("Agent") and isinstance(value, type)
        ]
        if len(candidates) == 1:
            agent_class = candidates[0]

    if not isinstance(agent_class, type):
        raise ValueError(f"submission must define a top-level Agent class: {path}")
    if not callable(getattr(agent_class, "get_action", None)):
        raise ValueError(f"Agent must define get_action(self, state): {path}")
    return agent_class


def safe_builtins() -> dict[str, Any]:
    allowed = dict(SAFE_BUILTINS)
    allowed["getattr"] = safe_getattr
    allowed["__import__"] = restricted_import
    return allowed


def safe_getattr(obj, name: str, default: Any = _MISSING) -> Any:
    if isinstance(name, str) and name.startswith("__"):
        raise AttributeError("dunder attribute access is not allowed in submissions")
    if default is _MISSING:
        return getattr(obj, name)
    return getattr(obj, name, default)


def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0:
        raise ImportError("relative imports are not allowed in submissions")
    root = name.split(".", 1)[0]
    if root not in ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"submissions may only import numpy, got {name!r}")
    return builtins.__import__(name, globals, locals, fromlist, level)


def validate_submission_source(source: str, path: Path) -> None:
    """Reject submission source that violates the one-file numpy-only contract."""
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f"submission has invalid Python syntax: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                validate_import_name(alias.name, path, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                raise ValueError(f"relative imports are not allowed: {path}:{node.lineno}")
            validate_import_name(node.module, path, node.lineno)
        elif isinstance(node, ast.Call):
            call_name = called_name(node.func)
            if call_name in BLOCKED_CALLS:
                raise ValueError(f"blocked call {call_name!r} in submission: {path}:{node.lineno}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"dunder attribute access is not allowed: {path}:{node.lineno}")
        elif isinstance(node, ast.Name) and node.id.startswith("__") and node.id not in {"__name__"}:
            raise ValueError(f"dunder name access is not allowed: {path}:{node.lineno}")


def validate_import_name(module_name: str, path: Path, line_number: int) -> None:
    root = module_name.split(".", 1)[0]
    if root not in ALLOWED_IMPORT_ROOTS:
        raise ValueError(f"submissions may only import numpy, got {module_name!r}: {path}:{line_number}")


def called_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def validate_action(action: np.ndarray, expected_rows: int, agent_name: str) -> np.ndarray:
    """Normalize and validate an agent action before passing it to the env."""
    array, message = validate_action_tuple(action, expected_rows)
    if array is None:
        raise ValueError(f"{agent_name} returned invalid action: {message}")
    return array


def run_episode(env: AMMEnvironment, get_action: Callable[[dict], np.ndarray], agent_name: str) -> np.ndarray:
    """Run one episode and return cumulative PnL per trajectory."""
    state, _ = env.reset()
    rewards = []
    terminated = np.zeros(env.num_trajectories, dtype=bool)

    while not np.any(terminated):
        observation = official_observation(state, env)
        action = validate_action(get_action(observation), env.num_trajectories, agent_name)
        state, reward, terminated, _, _ = env.step(action)
        rewards.append(reward)

    return np.sum(np.asarray(rewards), axis=0)


def evaluate_agent(
    name: str,
    kind: str,
    build_agent: Callable[[AMMEnvironment, ScenarioConfig], object],
    config: ScenarioConfig,
    seeds: Iterable[int],
) -> EvaluationResult:
    """Evaluate one agent builder across all seeds."""
    pnl_by_seed = []
    for seed in seeds:
        env = create_environment(config, seed)
        agent = build_agent(env, config)
        pnl_by_seed.append(run_episode(env, agent.get_action, name))
    return EvaluationResult(name=name, kind=kind, pnl=np.concatenate(pnl_by_seed))


def baseline_builders(include_full_range: bool) -> dict[str, Callable[[AMMEnvironment, ScenarioConfig], object]]:
    builders = {
        "HoldToken0": lambda env, _config: HoldToken0Agent(env),
        "DeployOnceWide": lambda env, _config: DeployOnceWideAgent(env),
    }
    if include_full_range:
        builders["FullRangeRebalance"] = lambda env, _config: FullRangeRebalanceAgent(env)
    return builders


def submission_builders(paths: Iterable[Path]) -> dict[str, Callable[[AMMEnvironment, ScenarioConfig], object]]:
    builders = {}
    for path in paths:
        agent_class = load_submission_agent(path)
        name = path.stem

        def build_agent(_env, config, cls=agent_class):
            return cls(config.submission_namespace())

        builders[name] = build_agent
    return builders


def collect_submission_paths(agent_path: str | None, submissions_dir: str | None) -> list[Path]:
    paths = []
    if agent_path:
        paths.append(Path(agent_path))
    if submissions_dir:
        paths.extend(sorted(Path(submissions_dir).glob("*.py")))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def parse_seeds(mode: str) -> list[int]:
    if mode == "practice":
        return PRACTICE_SEEDS

    raw = os.environ.get("HACKATHON_SEEDS")
    if not raw:
        raise ValueError("official mode requires HACKATHON_SEEDS to be set")

    try:
        seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("HACKATHON_SEEDS must be a comma-separated list of integers") from exc
    if not seeds:
        raise ValueError("HACKATHON_SEEDS did not contain any seeds")
    return seeds


def rank_results(results: list[EvaluationResult]) -> list[tuple[int, EvaluationResult]]:
    ordered = sorted(results, key=lambda result: result.mean_pnl, reverse=True)
    return [(rank, result) for rank, result in enumerate(ordered, start=1)]


def print_leaderboard(results: list[EvaluationResult], mode: str, seeds: list[int]) -> None:
    ranked = rank_results(results)
    print("=" * 94)
    print(f"Hackathon leaderboard | mode={mode} | seeds={len(seeds)} | paths/result={len(results[0].pnl) if results else 0}")
    print("=" * 94)
    header = (
        f"{'Rank':>4} | {'Agent':<24} | {'Kind':<10} | {'Mean':>10} | {'Std':>10} | "
        f"{'Median':>10} | {'Min':>10} | {'Max':>10} | {'Profitable':>10}"
    )
    print(header)
    print("-" * len(header))
    for rank, result in ranked:
        print(
            f"{rank:>4} | {result.name:<24} | {result.kind:<10} | "
            f"{result.mean_pnl:>+10.2f} | {result.std_pnl:>10.2f} | "
            f"{result.median_pnl:>+10.2f} | {result.min_pnl:>+10.2f} | "
            f"{result.max_pnl:>+10.2f} | {result.profitable_pct:>9.1f}%"
        )
    print("-" * len(header))


def write_leaderboard_csv(results: list[EvaluationResult], output_path: Path, mode: str, seeds: list[int]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank",
                "name",
                "kind",
                "mode",
                "num_paths",
                "mean_pnl",
                "std_pnl",
                "median_pnl",
                "min_pnl",
                "max_pnl",
                "profitable_pct",
                "num_seeds",
            ],
        )
        writer.writeheader()
        for rank, result in rank_results(results):
            writer.writerow({
                "rank": rank,
                "name": result.name,
                "kind": result.kind,
                "mode": mode,
                "num_paths": len(result.pnl),
                "mean_pnl": result.mean_pnl,
                "std_pnl": result.std_pnl,
                "median_pnl": result.median_pnl,
                "min_pnl": result.min_pnl,
                "max_pnl": result.max_pnl,
                "profitable_pct": result.profitable_pct,
                "num_seeds": len(seeds),
            })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate hackathon LP agent submissions.")
    parser.add_argument("--agent", help="Path to one submitted Python file defining Agent.")
    parser.add_argument("--submissions-dir", help="Directory containing submitted *.py files.")
    parser.add_argument("--mode", choices=["practice", "official"], default="practice")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Leaderboard CSV output path.")
    parser.add_argument("--num-trajectories", type=int, default=ScenarioConfig.num_trajectories)
    parser.add_argument("--n-steps", type=int, default=ScenarioConfig.n_steps)
    parser.add_argument("--include-full-range", action="store_true", help="Include always-rebalance baseline.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    submission_paths = collect_submission_paths(args.agent, args.submissions_dir)
    try:
        seeds = parse_seeds(args.mode)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config = ScenarioConfig(num_trajectories=args.num_trajectories, n_steps=args.n_steps)

    builders: list[tuple[str, str, Callable[[AMMEnvironment, ScenarioConfig], object]]] = []
    builders.extend((name, "baseline", builder) for name, builder in baseline_builders(args.include_full_range).items())
    builders.extend((name, "submission", builder) for name, builder in submission_builders(submission_paths).items())

    if not builders:
        raise SystemExit("No agents to evaluate. Pass --agent or --submissions-dir.")

    results = [
        evaluate_agent(name, kind, builder, config, seeds)
        for name, kind, builder in builders
    ]

    print_leaderboard(results, args.mode, seeds)
    output_path = Path(args.output)
    write_leaderboard_csv(results, output_path, args.mode, seeds)
    print(f"\nSaved leaderboard: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
