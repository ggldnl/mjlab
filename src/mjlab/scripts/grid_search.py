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
import yaml

import mjlab.tasks  # noqa: F401
import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends


@dataclass
class SearchRewardCfg:
    name: str
    group: str
    bounds: list[float]

@dataclass
class SearchMetricCfg:
    key: str
    weight: float
    direction: str

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


def _read_tb_scalar(log_dir: str, key: str) -> float | None:
    """Return the most recent value of a tensorboard scalar key, or None on failure."""
    # TODO take metrics from environment config
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
        ea.Reload()
        events = ea.Scalars(key)
        return float(events[-1].value) if events else None
    except Exception:
        return None


def _collect_metrics(log_dir: str, cfgs: list[SearchMetricCfg]) -> dict[str, float]:
    """Read all configured metric keys from tensorboard and return available ones."""
    # TODO this doesn't work
    return {m.key: v for m in cfgs if (v := _read_tb_scalar(log_dir, m.key)) is not None}


class ScoreNormalizer:
    """Accumulates per-metric min/max across all trials and produces a normalized composite score.

    Each metric is independently mapped to [0, 1] using its running range.
    Direction (maximize/minimize) is applied before weighting.
    This makes the score unit-agnostic and task-agnostic.
    """

    def __init__(self, cfgs: list[SearchMetricCfg]) -> None:
        self._cfgs = cfgs
        self._min: dict[str, float] = {}
        self._max: dict[str, float] = {}

    def __call__(self, metrics: dict[str, float]) -> float:
        for key, val in metrics.items():
            self._min[key] = min(self._min.get(key, float("inf")), val)
            self._max[key] = max(self._max.get(key, float("-inf")), val)

        total = total_w = 0.0
        for c in self._cfgs:
            val = metrics.get(c.key)
            if val is None:
                continue
            span = self._max[c.key] - self._min[c.key]
            norm = (val - self._min[c.key]) / span if span > 1e-8 else 0.5
            if c.direction == "minimize":
                norm = 1.0 - norm
            total += c.weight * norm
            total_w += abs(c.weight)

        return total / (total_w + 1e-8)


class SearchOnPolicyRunner(MjlabOnPolicyRunner):
    """Extends MjlabOnPolicyRunner with chunked learning and Optuna pruning support.

    learn() is called repeatedly in report_interval-sized chunks.
    After each chunk, metrics are read from tensorboard and reported to Optuna.
    If Optuna's pruner decides the trial is poor, TrialPruned is raised immediately.
    """

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        self._search_log_dir = log_dir or ""  # stored separately for tb reading

    def learn_with_search(
        self,
        num_learning_iterations: int,
        report_interval: int,
        trial: optuna.Trial,
        metric_cfgs: list[SearchMetricCfg],
        normalizer: ScoreNormalizer,
    ) -> float:
        """Run learning in chunks, reporting to Optuna after each chunk.

        The base runner picks up from self.current_learning_iteration each call,
        so iteration numbering is continuous across chunks.
        """
        remaining = num_learning_iterations
        first_chunk = True
        last_score = 0.0

        while remaining > 0:
            chunk = min(report_interval, remaining)
            # Randomize episode lengths only at trial start.
            super().learn(chunk, init_at_random_ep_len=first_chunk)
            first_chunk = False
            remaining -= chunk

            metrics = _collect_metrics(self._search_log_dir, metric_cfgs)
            print(f"Found metrics: {metrics}")
            if not metrics:
                # Tb file not written yet; skip this report interval.
                continue

            last_score = normalizer(metrics)
            trial.report(last_score, self.current_learning_iteration)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return last_score


def _inject_weights(
    env_cfg: ManagerBasedRlEnvCfg,
    reward_cfgs: list[SearchRewardCfg],
    trial: optuna.Trial,
) -> dict[str, float]:
    """Suggest a weight for each reward term and apply it to env_cfg in-place.

    For nonzero defaults: Optuna suggests a log-uniform multiplier over bounds.
    The sign of the original weight is always preserved.
    For zero defaults: bounds are treated as absolute values (linear sampling).
    """
    result: dict[str, float] = {}
    for r in reward_cfgs:
        default_w = env_cfg.rewards[r.name].weight
        lo, hi = r.bounds[0], r.bounds[1]

        if abs(default_w) < 1e-9:
            new_w = trial.suggest_float(r.name, lo, hi)  # absolute, linear
        else:
            sign = -1.0 if default_w < 0 else 1.0
            multiplier = trial.suggest_float(r.name, lo, hi, log=True)  # log-uniform multiplier
            new_w = sign * abs(default_w) * multiplier

        env_cfg.rewards[r.name].weight = new_w
        result[r.name] = new_w

    return result


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

        # Persist suggested weights for this trial in case the study is interrupted.
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
        # Force tensorboard so each trial writes a local event file rather than
        # opening a new wandb run for every chunk of learn().
        for key in ("logger", "logger_type"):
            if key in agent_dict:
                agent_dict[key] = "tensorboard"
        # Suppress intermediate checkpoints to save disk.
        # The runner still saves one model at the end of the final chunk.
        agent_dict["save_interval"] = search_cfg.trial_iterations + 1

        runner = SearchOnPolicyRunner(env, agent_dict, str(trial_dir), device)

        try:
            score = runner.learn_with_search(
                num_learning_iterations=search_cfg.trial_iterations,
                report_interval=search_cfg.report_interval,
                trial=trial,
                metric_cfgs=search_cfg.metrics,
                normalizer=normalizer,
            )
        except optuna.TrialPruned:
            print(f"Trial {trial.number:04d} pruned at iter {runner.current_learning_iteration}.")
            env.close()
            raise
        except Exception as exc:
            # Catch unexpected errors so a single bad trial doesn't abort the study.
            print(f"Trial {trial.number:04d} failed: {exc}")
            env.close()
            return float("-inf")

        env.close()
        print(f"Trial {trial.number:04d} complete. score={score:.4f}")

        if score > best_state["score"]:
            best_state.update({"score": score, "trial": trial.number, "weights": weights})
            _save_best(best_params_path, trial.number, score, weights)

        # Print importances periodically after a warmup of 10 complete trials.
        n_complete = sum(
            1 for t in trial.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        )
        if n_complete >= 10 and n_complete % 5 == 0:
            _print_importances(trial.study)

        return score

    return objective


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

    # Multivariate TPE: models joint parameter distributions, not independent ones.
    # This is important because reward weights interact (e.g. high task weight
    # only works well with appropriately scaled regularization).
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)

    # HyperbandPruner brackets trials at multiple budgets so a trial that looks
    # bad at 500 iters but might recover gets a fair chance at the lower bracket.
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
        load_if_exists=True,  # resume an interrupted search transparently
    )

    print(f"Study  : {search_cfg.study_name}")
    print(f"Trials : {search_cfg.n_trials} x {search_cfg.trial_iterations} iters each")
    print(f"Report : every {search_cfg.report_interval} iters")
    print(f"Logs   : {log_root}")
    print(f"DB     : {args.study_db}\n")
    print('-' * 80)

    # Reset reward weights in config.yaml
    for reward in search_cfg.rewards:
        base_env_cfg.rewards[reward.name].weight = 1.0

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