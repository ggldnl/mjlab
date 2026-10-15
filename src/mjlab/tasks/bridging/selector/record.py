"""Record the rollouts entry points are taken from.

Every entry set runs its task under the settings the experiments run it with: the play
config, no rewards, metrics, curriculum or pushes, commands never resampled, a tracker
started at its first frame, and the set's own fixed command on top. Each environment runs
one episode, just long enough to cover the range and the future after it. Auto reset stays
on, so a run that fails starts a new trajectory instead of carrying on from the floor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.bridging.bridges.dataset import dataset
from mjlab.tasks.bridging.bridges.dataset.dataset import RolloutCfg
from mjlab.tasks.bridging.config import get_robot
from mjlab.tasks.bridging.config.g1.skills.jump_continuous.mdp.commands import (
  JumpCommandCfg,
)
from mjlab.tasks.bridging.selector import FUTURE_STATES
from mjlab.tasks.bridging.selector.specs import Spec
from mjlab.tasks.registry import load_env_cfg


def recorded_steps(spec: Spec) -> int:
  """Control steps per episode: the range, the future after its last step, and slack."""
  return spec.stop + FUTURE_STATES + 2


def experiment_cfg(
  spec: Spec, robot: str, runs: int, seed: int
) -> ManagerBasedRlEnvCfg:
  """The task's environment as the experiments run it, with the set's command fixed."""
  cfg = load_env_cfg(spec.task, play=True)
  cfg.scene.num_envs = runs
  cfg.seed = seed
  cfg.rewards, cfg.metrics, cfg.curriculum = {}, {}, {}
  cfg.events.pop("push_robot", None)
  for command in cfg.commands.values():
    command.resampling_time_range = (1.0e9, 1.0e9)
    command.gui = command.debug_vis = False
    if isinstance(command, JumpCommandCfg):
      command.sampling_mode = "start"
  if spec.condition is not None:
    spec.condition(cfg)
  step_dt = cfg.sim.mujoco.timestep * cfg.decimation
  cfg.episode_length_s = max(cfg.episode_length_s, (recorded_steps(spec) + 1) * step_dt)
  sensors = tuple(cfg.scene.sensors or ())
  if not any(sensor.name == "feet_ground_contact" for sensor in sensors):
    left, right = get_robot(robot).foot_body_names
    cfg.scene.sensors = (
      *sensors,
      ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
          mode="subtree", pattern=rf"^({left}|{right})$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
      ),
    )
  return cfg


def collect(
  specs: dict[str, Spec],
  robot: str,
  checkpoints: dict[str, Path],
  path: Path,
  runs: int = 64,
  seed: int = 0,
  device: str = "cuda:0",
) -> Path:
  """Record every entry set into one rollout file, one source per set."""
  feet = get_robot(robot).foot_body_names
  states: list[np.ndarray] = []
  env_ids: list[np.ndarray] = []
  trajectories: list[np.ndarray] = []
  frames: list[np.ndarray] = []
  phases: list[np.ndarray] = []
  sources: list[np.ndarray] = []
  commands: list[np.ndarray] = []
  metadata: list[dict[str, np.ndarray]] = []
  fps: float | None = None

  for source, (name, spec) in enumerate(specs.items()):
    env_cfg = experiment_cfg(spec, robot, runs, seed)
    rate = dataset.control_rate(env_cfg)
    if fps is not None and abs(rate - fps) > 1e-6:
      raise ValueError("All recorded skills must use the same control rate")
    fps = rate
    options = RolloutCfg(
      num_envs=runs, steps=recorded_steps(spec), settle=0, device=device
    )
    context: dict[str, np.ndarray] = {}
    state, env_id, trajectory, frame, phase, command = dataset.record(
      spec.task, env_cfg, checkpoints[name], options, name, metadata=context, feet=feet
    )
    if "foot_contact" not in context:
      raise ValueError(f"{name} has no recorded feet_ground_contact sensor")
    states.append(state)
    env_ids.append(env_id)
    trajectories.append(trajectory)
    frames.append(frame)
    phases.append(phase)
    sources.append(np.full(len(state), source, dtype=np.int16))
    commands.append(command)
    metadata.append(context)

  if fps is None:
    raise ValueError("At least one entry set is required")
  return dataset.write(
    path,
    states,
    env_ids,
    trajectories,
    frames,
    sources,
    tuple(specs),
    fps,
    commands,
    phases,
    metadata,
  )
