"""Reward weight search using Optuna with Bayesian (TPE) optimization."""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import optuna
import torch
import yaml

import mjlab
import mjlab.tasks  # noqa: F401
import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends


# Config dataclasses mirroring the mandatory YAML schema.

@dataclass
class SearchRewardCfg:
    name: str
    group: str
    bounds: list[float]

@dataclass
class SearchMetricCfg:
    key: str
    weight: float
    direction: str  # "maximize" | "minimize"

@dataclass
class SearchConfig:
    study_name: str
    n_trials: int
    trial_iterations: int
    report_interval: int
    rewards: list[SearchRewardCfg]
    metrics: list[SearchMetricCfg]


def _load_search_config(path: Path) -> SearchConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return SearchConfig(
        study_name=raw["study_name"],
        n_trials=int(raw["n_trials"]),
        trial_iterations=int(raw["trial_iterations"]),
        report_interval=int(raw["report_interval"]),
        rewards=[SearchRewardCfg(**r) for r in raw["rewards"]],
        metrics=[SearchMetricCfg(**m) for m in raw["metrics"]],
    )


# Metric accumulation from env.step().
#
# _SingleMetricAccumulator is a dumb buffer for one scalar metric: it receives
# float values pushed from outside and returns their mean on pop().
#
# _StepInterceptor wraps RslRlVecEnvWrapper and owns a dict of accumulators
# keyed by metric name. On every step it computes the three tensor-derived
# metrics (mean_reward, mean_ep_length, reward_std) from rewards/dones and
# harvests any additional metrics from extras["episode"] — both sets are pushed
# into the corresponding accumulator if one exists for that key.
#
# The YAML drives which keys exist: SearchOnPolicyRunner creates one accumulator
# per metric listed in the config before constructing the interceptor.

class _SingleMetricAccumulator:
    """Collects scalar values for one metric and returns their mean on pop()."""

    def __init__(self) -> None:
        self._buffer: list[float] = []

    def push(self, value: float) -> None:
        self._buffer.append(value)

    def pop(self) -> float | None:
        """Return the mean of all pushed values since last call, or None if empty."""
        if not self._buffer:
            return None
        result = sum(self._buffer) / len(self._buffer)
        self._buffer.clear()
        return result


class _StepInterceptor:
    """Wraps RslRlVecEnvWrapper and feeds registered accumulators on every step.

    Two extraction paths, neither of which hardcodes metric names:
      - Tensor-derived: mean_reward, mean_ep_length, reward_std are computed
        per-episode from the rewards/dones tensors. These three keys are always
        available regardless of what the environment exposes.
      - extras["episode"]: any key written by MetricsManager is forwarded to
        the accumulator with the matching name, if one was registered.
    """

    # Internal keys for the three tensor-derived metrics.
    # These are only used inside this class to push into accumulators;
    # they are not exported or hardcoded anywhere else.
    _KEY_MEAN_REWARD = "mean_reward"
    _KEY_MEAN_EP_LENGTH = "mean_ep_length"
    _KEY_REWARD_STD = "reward_std"

    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        accumulators: dict[str, _SingleMetricAccumulator],
    ) -> None:
        self._env = env
        self._accumulators = accumulators
        # Per-env episode buffers; lazily initialized on first step.
        self._ep_rewards: torch.Tensor | None = None
        self._ep_lengths: torch.Tensor | None = None
        # Completed-episode reward buffer for std computation.
        self._done_rewards: list[float] = []

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def step(self, actions: torch.Tensor):
        obs, rewards, dones, extras = self._env.step(actions)

        if self._ep_rewards is None:
            n = rewards.shape[0]
            self._ep_rewards = torch.zeros(n, device=rewards.device)
            self._ep_lengths = torch.zeros(n, device=rewards.device)

        self._ep_rewards += rewards
        self._ep_lengths += 1

        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        ep_rewards_done: list[float] = []
        ep_lengths_done: list[float] = []

        for idx in done_ids:
            ep_rewards_done.append(float(self._ep_rewards[idx]))
            ep_lengths_done.append(float(self._ep_lengths[idx]))
            self._ep_rewards[idx] = 0.0
            self._ep_lengths[idx] = 0.0

        # Push tensor-derived metrics into their accumulators if registered.
        if ep_rewards_done:
            mean = sum(ep_rewards_done) / len(ep_rewards_done)
            self._done_rewards.extend(ep_rewards_done)

            if self._KEY_MEAN_REWARD in self._accumulators:
                self._accumulators[self._KEY_MEAN_REWARD].push(mean)

            if self._KEY_MEAN_EP_LENGTH in self._accumulators:
                self._accumulators[self._KEY_MEAN_EP_LENGTH].push(
                    sum(ep_lengths_done) / len(ep_lengths_done)
                )

            if self._KEY_REWARD_STD in self._accumulators and len(self._done_rewards) > 1:
                overall_mean = sum(self._done_rewards) / len(self._done_rewards)
                variance = sum((r - overall_mean) ** 2 for r in self._done_rewards) / len(self._done_rewards)
                self._accumulators[self._KEY_REWARD_STD].push(variance ** 0.5)

        # Forward anything MetricsManager wrote into extras["episode"].
        for key, val in extras.get("episode", {}).items():
            if key in self._accumulators:
                self._accumulators[key].push(
                    float(val.mean()) if hasattr(val, "mean") else float(val)
                )

        return obs, rewards, dones, extras


# Running composite score with per-metric normalization.
#
# Each metric is independently mapped to [0, 1] using its running range across
# all trials seen so far. This makes the score unit-agnostic, so mean_reward
# (typically O(1-10)) and mean_ep_length (typically O(100-1000)) can be
# combined with plain scalar weights in the YAML.
# Direction is applied after normalization, before weighting.

class ScoreNormalizer:

    def __init__(self, cfgs: list[SearchMetricCfg]) -> None:
        self._cfgs = {c.key: c for c in cfgs}
        self._min: dict[str, float] = {}
        self._max: dict[str, float] = {}

    def __call__(self, metrics: dict[str, float]) -> float:
        for key, val in metrics.items():
            self._min[key] = min(self._min.get(key, float("inf")), val)
            self._max[key] = max(self._max.get(key, float("-inf")), val)

        total = total_w = 0.0
        for key, cfg in self._cfgs.items():
            val = metrics.get(key)
            if val is None:
                continue
            span = self._max[key] - self._min[key]
            norm = (val - self._min[key]) / span if span > 1e-8 else 0.5
            if cfg.direction == "minimize":
                norm = 1.0 - norm
            total += cfg.weight * norm
            total_w += abs(cfg.weight)

        return total / (total_w + 1e-8)


# Search-aware runner.
#
# __init__ creates one _SingleMetricAccumulator per key listed in the YAML,
# passes them all to _StepInterceptor, then forwards the interceptor to the
# base runner as its env. The base learn() feeds every accumulator on every
# step with no changes to the base class.
#
# learn_with_search() calls learn() in report_interval-sized chunks.
# The base runner reads self.current_learning_iteration as start_it each call,
# so iteration numbering is continuous across chunks automatically.

class SearchOnPolicyRunner(MjlabOnPolicyRunner):

    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        metric_keys: list[str] | None = None,
        **kwargs,
    ) -> None:
        # One accumulator per YAML metric key; no names hardcoded here.
        self._accumulators: dict[str, _SingleMetricAccumulator] = {
            key: _SingleMetricAccumulator() for key in (metric_keys or [])
        }
        interceptor = _StepInterceptor(env, self._accumulators)
        super().__init__(interceptor, train_cfg, log_dir, device, **kwargs)

    def learn_with_search(
        self,
        num_learning_iterations: int,
        report_interval: int,
        trial: optuna.Trial,
        normalizer: ScoreNormalizer,
    ) -> float:
        remaining = num_learning_iterations
        first_chunk = True
        last_score = 0.0

        while remaining > 0:
            chunk = min(report_interval, remaining)
            super().learn(chunk, init_at_random_ep_len=first_chunk)
            first_chunk = False
            remaining -= chunk

            metrics = {k: v for k, v in (
                (k, acc.pop()) for k, acc in self._accumulators.items()
            ) if v is not None}
            if not metrics:
                # No episodes completed in this chunk yet; skip this report.
                continue

            last_score = normalizer(metrics)
            trial.report(last_score, self.current_learning_iteration)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return last_score


# Weight suggestion and injection.

def _inject_weights(
    env_cfg: ManagerBasedRlEnvCfg,
    reward_cfgs: list[SearchRewardCfg],
    trial: optuna.Trial,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for r in reward_cfgs:
        default_w = env_cfg.rewards[r.name].weight
        lo, hi = r.bounds[0], r.bounds[1]

        """
        if abs(default_w) < 1e-9:
            new_w = trial.suggest_float(r.name, lo, hi)
        else:
            sign = -1.0 if default_w < 0 else 1.0
            multiplier = trial.suggest_float(r.name, lo, hi, log=True)
            new_w = sign * abs(default_w) * multiplier
        """
        sign = -1.0 if default_w < 0 else 1.0
        multiplier = trial.suggest_float(r.name, lo, hi)
        new_w = sign * abs(default_w) * multiplier

        env_cfg.rewards[r.name].weight = new_w
        result[r.name] = new_w

    return result


# Best result persistence.

def _save_best(path: Path, trial_n: int, score: float, weights: dict[str, float]) -> None:
    with open(path, "w") as f:
        yaml.dump(
            {
                "trial": trial_n,
                "score": round(float(score), 6),
                "reward_weights": {k: round(float(v), 6) for k, v in weights.items()},
            },
            f,
            default_flow_style=False,
        )
    print(f"New best → trial {trial_n:04d}, score={score:.4f}, saved to {path}")


def _print_importances(study: optuna.Study) -> None:
    try:
        importances = optuna.importance.get_param_importances(study)
        print("\nParameter importances:")
        for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
            print(f"  {name:<40s} {imp:.3f}")
        print()
    except Exception:
        pass


# Objective factory.

def _make_objective(
    base_env_cfg: ManagerBasedRlEnvCfg,
    base_agent_cfg: RslRlBaseRunnerCfg,
    search_cfg: SearchConfig,
    device: str,
    log_root: Path,
    normalizer: ScoreNormalizer,
    best_state: dict,
    best_params_path: Path,
) -> Callable[[optuna.Trial], float]:

    def objective(trial: optuna.Trial) -> float:
        env_cfg = copy.deepcopy(base_env_cfg)
        agent_cfg = copy.deepcopy(base_agent_cfg)

        weights = _inject_weights(env_cfg, search_cfg.rewards, trial)

        print(f"\nTrial {trial.number:04d} starting.")
        for name, w in weights.items():
            print(f"  {name}: {w:.5f}")

        trial_dir = log_root / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        with open(trial_dir / "weights.yaml", "w") as f:
            yaml.dump(
                {"reward_weights": {k: round(float(v), 6) for k, v in weights.items()}},
                f,
                default_flow_style=False,
            )

        configure_torch_backends()

        env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        agent_dict = asdict(agent_cfg)
        for key in ("logger", "logger_type"):
            if key in agent_dict:
                agent_dict[key] = "tensorboard"
        agent_dict["save_interval"] = search_cfg.trial_iterations + 1

        runner = SearchOnPolicyRunner(env, agent_dict, str(trial_dir), device, metric_keys=[m.key for m in search_cfg.metrics])

        try:
            score = runner.learn_with_search(
                num_learning_iterations=search_cfg.trial_iterations,
                report_interval=search_cfg.report_interval,
                trial=trial,
                normalizer=normalizer,
            )
        except optuna.TrialPruned:
            print(f"Trial {trial.number:04d} pruned at iter {runner.current_learning_iteration}.")
            env.close()
            raise
        except Exception as exc:
            print(f"Trial {trial.number:04d} failed: {exc}")
            env.close()
            return float("-inf")

        env.close()
        print(f"Trial {trial.number:04d} complete. score={score:.4f}")

        if score > best_state["score"]:
            best_state.update({"score": score, "trial": trial.number, "weights": weights})
            _save_best(best_params_path, trial.number, score, weights)

        n_complete = sum(
            1 for t in trial.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        )
        if n_complete >= 10 and n_complete % 5 == 0:
            _print_importances(trial.study)

        return score

    return objective


# CLI config.

@dataclass(frozen=True)
class _SearchArgs:
    search_config: Path
    """Path to the mandatory YAML search configuration file."""
    num_envs: int | None = None
    """Override the number of parallel environments from the task default."""
    gpu_id: int = 0
    """CUDA device index to use for all trials."""
    study_db: str = "reward_search.db"
    """SQLite file for the Optuna study (created if absent, resumed if already present)."""


def main() -> None:
    all_tasks = list_tasks()
    chosen_task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    args = tyro.cli(
        _SearchArgs,
        args=remaining,
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    search_cfg = _load_search_config(args.search_config)
    base_env_cfg = load_env_cfg(chosen_task)
    base_agent_cfg = load_rl_cfg(chosen_task)

    if args.num_envs is not None:
        base_env_cfg.scene.num_envs = args.num_envs

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    device = "cuda:0"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = Path("logs") / "reward_search" / search_cfg.study_name / timestamp
    log_root.mkdir(parents=True, exist_ok=True)
    best_params_path = log_root / "best_params.yaml"

    normalizer = ScoreNormalizer(search_cfg.metrics)
    best_state: dict = {"score": float("-inf"), "trial": -1, "weights": {}}

    # Multivariate TPE: models joint parameter distributions rather than treating
    # each weight independently. Critical here because reward weights interact
    # (e.g. high task weight only works with proportionally scaled regularization).
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)

    # HyperbandPruner: brackets trials at multiple budgets so a trial that looks
    # weak at 500 iters but might recover isn't killed too aggressively.
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=search_cfg.report_interval,
        max_resource=search_cfg.trial_iterations,
        reduction_factor=3,
    )

    study = optuna.create_study(
        study_name=search_cfg.study_name,
        storage=f"sqlite:///{args.study_db}",
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,  # resume transparently if the db already has this study
    )

    print(f"Study  : {search_cfg.study_name}")
    print(f"Trials : {search_cfg.n_trials} × {search_cfg.trial_iterations} iters each")
    print(f"Report : every {search_cfg.report_interval} iters")
    print(f"Logs   : {log_root}")
    print(f"DB     : {args.study_db}\n")

    # Reset reward weights in config.yaml
    for reward in search_cfg.rewards:
        default_w = base_env_cfg.rewards[reward.name].weight
        sign = -1.0 if default_w < 0 else 1.0
        base_env_cfg.rewards[reward.name].weight = sign

    objective = _make_objective(
        base_env_cfg=base_env_cfg,
        base_agent_cfg=base_agent_cfg,
        search_cfg=search_cfg,
        device=device,
        log_root=log_root,
        normalizer=normalizer,
        best_state=best_state,
        best_params_path=best_params_path,
    )

    study.optimize(objective, n_trials=search_cfg.n_trials)

    print('-' * 80)
    print("Search complete")
    if best_state["trial"] >= 0:
        print(f"Best trial : {best_state['trial']:04d}, score={best_state['score']:.4f}")
        print(f"Best params: {best_params_path}")
    _print_importances(study)


if __name__ == "__main__":
    main()