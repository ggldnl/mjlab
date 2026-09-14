"""Widen a tracking skill's reset noise and finetune it, so it survives a hand-over.

A skill is handed a robot by another policy, not by its own reset, and the two do not look
alike. Measured at kick entry 3 (frame 123), the bridge leaves the robot 4.3 tolerances out
on the worst leg joint while the skill's own reset perturbs it by 0.5, and a sweep says the
skill stops working past 2. The skill has never seen the states it is given.

The reset is already a simulated hand-over: a tracker writes the robot onto the reference
frame plus noise. This widens that noise and retrains, changing nothing else. The reward,
the observations, the network and the clip are the task's own, so the result is the same
skill with a larger initiation set, tied to no bridge and to no other skill.

How wide is one knob per bridge channel, in multiples of that channel's arrival tolerance,
so a number reads against the sweep in benchmarks/tolerance.py. See Scales. Joint noise is
drawn per joint and the tolerances measure the worst joint in a group, and the maximum of
15 leg draws from uniform(-a, a) sits at about 0.94a, so a scale of 4 produces a channel
error of about 4 tolerances.

The draws are independent across channels, which is close to what a real hand-over looks
like: measured over 19 recorded crossings, root position error and root linear velocity
error correlate at +0.23 and +0.25 on the horizontal axes and joint position and joint
velocity error at +0.12. Correlating them would need the tracker's reset to draw them
together, and this measurement does not ask for it.

Three things are handled that a plain `train --agent.resume` gets wrong:

    curriculum      a fresh run restarts the task's own curricula at step zero, which for
                    the kick drops the ball scatter to nothing and halves the ball reward
                    weights. That undoes the training being continued. The step counter is
                    advanced past the last stage instead, so every curriculum starts at the
                    value the loaded checkpoint was trained at
    ramp            a converged policy hit with the full noise on the first iteration
                    collapses, and collects the termination penalty for a state it could
                    not have avoided. The noise is ramped from the task's own ranges to the
                    target over the first part of the run
    far threshold   the reset offset alone eats a large share of the tracking failure
                    threshold before the policy has acted. It is scaled up here

What to check:

    Curriculum/reset_noise/scale      the ramp, 0 to 1. Reaches 1.0 at ramp_fraction of
                                      the run and stays there. The two ranges beside it are
                                      the joint noise it currently means, in rad and rad/s
    Train/mean_reward                 dips when the ramp starts and should recover. If it
                                      does not, the target scale is past what this skill
                                      can do and the run is measuring a broken task
    Episode_Termination/motion_far    the reset noise landing outside the tracking
                                      threshold. A large share early is the ramp working;
                                      a large share at the end is a threshold to widen
    Episode_Metrics/kick_ball_speed   the outcome under the widened reset, so it falls as
                                      the ramp comes on and that is the noise, not a
                                      regression. Nominal quality is the benchmark's own
                                      baseline case in step 2 below, measured from a clean
                                      reset. Judge the skill on that one

Run

1. Finetune. The checkpoint defaults to the newest under the skill's own log directory.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.skills.finetune --skill kick

2. Measure the new initiation set against the old, which is the test that does not involve
   a bridge. Point --checkpoint at each in turn.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.benchmarks.run \
        --skill kick --phases '(123,)' --seeds '(0,1,2)' \
        --channels "('root_pos_x','root_pos_y','root_rot_y','root_lin_vel_x','leg_joint_pos','leg_joint_vel','arm_joint_pos','arm_joint_vel')" \
        --scales '(1.0,2.0,4.0,6.0,8.0)' \
        --amplitudes.root-pos 0.05 --amplitudes.root-rot 0.05 \
        --amplitudes.root-lin-vel 0.15 --amplitudes.root-ang-vel 0.30 \
        --save-traces False --output logs/benchmarks/tolerance

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.benchmarks.tolerance \
        --directory logs/benchmarks/tolerance

3. Measure the hand-over itself, both kick policies in one sweep.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.benchmarks.kick.transitions \
        --entry 3 --switch-steps '(100,130,160)' \
        --kick-checkpoints '(BASELINE.pt,FINETUNED.pt)'

Only tracking skills are supported: the widening happens inside the clip tracker's reset.
The noise is drawn at every frame the skill resets into, not only at the frames a bridge
aims for. Narrowing it to an entry window would be cheaper and would tie the skill to the
bridge, which is what this is avoiding. If a skill will not converge at a useful scale,
that is the next thing to try.
"""

from __future__ import annotations

import json
from dataclasses import asdict, astuple, dataclass, field, replace
from datetime import datetime
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp.commands import (
  Tolerances,
)
from mjlab.tasks.bridging.experiments.humanoid.skills import SKILLS
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.mdp.commands import (
  JumpCommandCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import find_checkpoint
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.os import dump_yaml

Span = tuple[float, float]
Ranges = dict[str, Span]
Target = tuple[Ranges, Ranges, Span, Span]
"""What a reset draws from: root pose, root velocity, joint position, joint velocity."""

FAR_TERMINATION = "motion_far"
"""The termination the reset noise runs into first, and the threshold scaled to allow it."""


@dataclass(frozen=True)
class Scales:
  """How wide the reset opens, per bridge channel, in multiples of that channel's tolerance.

  One number per channel and not one for all of them, because what a skill survives and
  what the bridge misses by are both strongly anisotropic. The defaults are what the bridge
  measurably delivers at kick entry 3, so the reset covers the states a hand-over actually
  produces. Read them against benchmarks/tolerance.py for the skill being finetuned:
  widening a channel the bridge already satisfies spends training on nothing.

  Bridge delivers / kick survives, in tolerances, at entry 3:

      root position       5.7 / 8      already covered
      root orientation    1.6 / 2      already covered, and 2 is the pitch limit
      root linear vel     4.8 / 4      short, the reason this exists
      root angular vel    3.0 / 8      already covered
      leg joint position  4.3 / 2      short, the reason this exists
      leg joint velocity  1.9 / 4      already covered
  """

  root_pos: float = 6.0
  root_ori: float = 4.0
  """Covers roll, pitch and yaw together, and one knob cannot tell them apart.

  Set above what the bridge delivers, unlike the others, because the finetune that has
  been run says overshooting a channel is what moves it: trained at 4 while the baseline
  pitch limit was 2, the pitch limit came back at 4. A run at the delivered 1.6 would have
  trained the channel less than the run that improved it."""
  root_lin_vel: float = 5.0
  root_ang_vel: float = 3.0
  joint_pos: float = 4.0
  """Legs and arms share one range: the tracker draws a single span for every joint. Set
  from the leg tolerance, so the arms see about twice this in their own units."""
  joint_vel: float = 2.0
  """Lower than the positions on purpose. Joint velocity was never perturbed at a reset at
  all before this, and a full scale draw across 29 joints is a lot of energy to hand a
  robot that is also somewhere it did not expect to be."""


@dataclass
class Config:
  skill: str = "kick"
  checkpoint: Path | None = None
  """What to finetune. The newest under the skill's own log directory when not given."""

  scales: Scales = field(default_factory=Scales)
  """How wide to open each channel, in multiples of its bridge arrival tolerance."""
  ramp_fraction: float = 0.5
  """Share of the run spent reaching the target scale, from the task's own ranges."""
  far_threshold_scale: float = 1.5
  """How much to widen the tracking failure threshold, which the reset offset eats into."""

  iterations: int = 1500
  num_envs: int = 4096
  seed: int = 0
  device: str = "cuda:0"
  logger: str = "tensorboard"
  suffix: str = "robust"
  """Appended to the experiment name, so this logs beside the skill rather than into it."""
  log_root: Path = Path("logs") / "rsl_rl"


def widen(baseline: Ranges, target: Ranges) -> Ranges:
  """The wider of the two, per key. A finetune never narrows what the task already had."""
  keys = set(baseline) | set(target)
  return {
    key: (
      min(baseline.get(key, (0.0, 0.0))[0], target.get(key, (0.0, 0.0))[0]),
      max(baseline.get(key, (0.0, 0.0))[1], target.get(key, (0.0, 0.0))[1]),
    )
    for key in keys
  }


def targets(scales: Scales) -> Target:
  """The reset ranges the ramp ends at, from the bridge tolerances.

  Height and vertical velocity get half the horizontal scale. The clips are already shifted
  to stand a foot on the floor and a reset writes the robot into the pose, so a large
  vertical offset either buries it or drops it.
  """
  tolerance = Tolerances()
  flat = scales.root_pos * tolerance.root_pos
  turn = scales.root_ori * tolerance.root_ori
  drift = scales.root_lin_vel * tolerance.root_lin_vel
  spin = scales.root_ang_vel * tolerance.root_ang_vel
  pose: Ranges = {
    "x": (-flat, flat),
    "y": (-flat, flat),
    "z": (-0.5 * flat, 0.5 * flat),
    "roll": (-turn, turn),
    "pitch": (-turn, turn),
    "yaw": (-turn, turn),
  }
  speeds: Ranges = {
    "x": (-drift, drift),
    "y": (-drift, drift),
    "z": (-0.5 * drift, 0.5 * drift),
    "roll": (-spin, spin),
    "pitch": (-spin, spin),
    "yaw": (-spin, spin),
  }
  # One range for every joint, set from the leg tolerance, which is the looser of the two.
  # The arms are then perturbed by about twice the scale, which the sweep says they take
  joints = scales.joint_pos * tolerance.leg_joint_pos
  rates = scales.joint_vel * tolerance.leg_joint_vel
  return pose, speeds, (-joints, joints), (-rates, rates)


class reset_noise_curriculum:
  """Ramp the tracker's reset noise from where the task had it to where this run wants it.

  Linear in the environment step count from the iteration this run started at, which is not
  zero: the step counter is advanced before training so the task's own curricula start at
  the stage the loaded checkpoint was trained at. The origin is taken on the first call
  rather than configured, so the two cannot drift apart.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
    term_cfg = env.command_manager.get_term_cfg(cfg.params["command_name"])
    if not isinstance(term_cfg, JumpCommandCfg):
      raise TypeError("The reset noise ramp needs a clip tracker to widen")
    self._term_cfg: JumpCommandCfg = term_cfg
    self._from: Target = (
      dict(term_cfg.pose_range),
      dict(term_cfg.velocity_range),
      term_cfg.joint_position_range,
      term_cfg.joint_velocity_range,
    )
    self._to: Target = cfg.params["target"]
    self._steps: int = cfg.params["ramp_steps"]
    self._origin: int | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    target: Target,
    ramp_steps: int,
  ) -> dict[str, torch.Tensor]:
    del env_ids, command_name, target, ramp_steps
    if self._origin is None:
      self._origin = env.common_step_counter
    alpha = min((env.common_step_counter - self._origin) / max(self._steps, 1), 1.0)
    pose, speeds, joints, rates = self._from
    to_pose, to_speeds, to_joints, to_rates = self._to

    def blend(a: Span, b: Span) -> Span:
      return (a[0] + alpha * (b[0] - a[0]), a[1] + alpha * (b[1] - a[1]))

    self._term_cfg.pose_range = {
      key: blend(pose.get(key, (0.0, 0.0)), value) for key, value in to_pose.items()
    }
    self._term_cfg.velocity_range = {
      key: blend(speeds.get(key, (0.0, 0.0)), value) for key, value in to_speeds.items()
    }
    self._term_cfg.joint_position_range = blend(joints, to_joints)
    self._term_cfg.joint_velocity_range = blend(rates, to_rates)
    return {
      "scale": torch.tensor(alpha),
      "joint_pos": torch.tensor(self._term_cfg.joint_position_range[1]),
      "joint_vel": torch.tensor(self._term_cfg.joint_velocity_range[1]),
    }


def tracker(env_cfg: ManagerBasedRlEnvCfg) -> str:
  """The name of the task's clip tracker. Raises for a skill that has none."""
  found = [
    name for name, term in env_cfg.commands.items() if isinstance(term, JumpCommandCfg)
  ]
  if len(found) != 1:
    raise SystemExit(
      "Finetuning widens a clip tracker's reset, and this task has "
      f"{len(found)} of them. Only tracking skills are supported."
    )
  return found[0]


def relax_far_termination(env_cfg: ManagerBasedRlEnvCfg, scale: float) -> None:
  """Widen the tracking failure threshold, in the term and in whatever curriculum moves it.

  Both, because a curriculum rewrites its term's params on every tick: editing the term
  alone is undone on the first reset.
  """
  term = env_cfg.terminations.get(FAR_TERMINATION)
  if term is None:
    return
  term.params["threshold"] = float(term.params["threshold"]) * scale
  for curriculum in env_cfg.curriculum.values():
    if curriculum.params.get("termination_name") != FAR_TERMINATION:
      continue
    for stage in curriculum.params["stages"]:
      stage["params"]["threshold"] = float(stage["params"]["threshold"]) * scale


def build(cfg: Config) -> tuple[ManagerBasedRlEnvCfg, int]:
  """The finetuning environment, and the step count to start its curricula at."""
  env_cfg = load_env_cfg(SKILLS[cfg.skill])
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  name = tracker(env_cfg)
  command = env_cfg.commands[name]
  assert isinstance(command, JumpCommandCfg)

  pose, speeds, joints, rates = targets(cfg.scales)
  target = (
    widen(command.pose_range, pose),
    widen(command.velocity_range, speeds),
    (
      min(command.joint_position_range[0], joints[0]),
      max(command.joint_position_range[1], joints[1]),
    ),
    (
      min(command.joint_velocity_range[0], rates[0]),
      max(command.joint_velocity_range[1], rates[1]),
    ),
  )
  relax_far_termination(env_cfg, cfg.far_threshold_scale)

  # Past every stage of every curriculum the task carries, so a fresh run opens at the
  # values the loaded checkpoint was trained at instead of rewinding to the first stage
  start = 0
  for curriculum in env_cfg.curriculum.values():
    for stage in curriculum.params.get("stages", []):
      start = max(start, int(stage["step"]))
  start += 1

  # common_step_counter advances once per control step, so an iteration is
  # num_steps_per_env of it whatever the environment count. The task's own curriculum
  # thresholds are in the same unit
  steps_per_iteration = int(load_rl_cfg(SKILLS[cfg.skill]).num_steps_per_env)
  env_cfg.curriculum["reset_noise"] = CurriculumTermCfg(
    func=reset_noise_curriculum,
    params={
      "command_name": name,
      "target": target,
      "ramp_steps": max(
        int(cfg.ramp_fraction * cfg.iterations) * steps_per_iteration, 1
      ),
    },
  )
  return env_cfg, start


def run(cfg: Config) -> Path:
  if cfg.skill not in SKILLS:
    raise SystemExit(f"Unknown skill. Known: {', '.join(sorted(SKILLS))}.")
  if min(cfg.iterations, cfg.num_envs) < 1:
    raise SystemExit("Iterations and environment count must be positive")
  if min(astuple(cfg.scales)) < 0 or cfg.far_threshold_scale <= 0:
    raise SystemExit(
      "Noise scales must not be negative and the threshold scale positive"
    )
  if not 0.0 <= cfg.ramp_fraction <= 1.0:
    raise SystemExit("The ramp fraction is a share of the run")

  task = SKILLS[cfg.skill]
  agent = load_rl_cfg(task)
  source = find_checkpoint(agent.experiment_name, cfg.checkpoint)
  print(f"[finetune] continuing {source}")

  agent = replace(
    agent,
    experiment_name=f"{agent.experiment_name}_{cfg.suffix}",
    max_iterations=cfg.iterations,
    logger=cfg.logger,
    upload_model=False,
    resume=False,
  )
  env_cfg, start = build(cfg)

  log_dir = (
    cfg.log_root / agent.experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  )
  log_dir.mkdir(parents=True)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent))
  (log_dir / "finetune.json").write_text(
    json.dumps(
      {
        "config": asdict(cfg),
        "source_checkpoint": str(source),
        "curriculum_start_step": start,
      },
      indent=2,
      default=str,
    )
  )
  print(f"[finetune] logging to {log_dir}")

  torch.manual_seed(cfg.seed)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  try:
    env.common_step_counter = start
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent), str(log_dir), cfg.device)
    runner.load(
      str(source), load_cfg={"actor": True, "critic": True}, map_location=cfg.device
    )
    runner.current_learning_iteration = 0
    env.common_step_counter = start
    runner.learn(num_learning_iterations=cfg.iterations, init_at_random_ep_len=True)
  finally:
    env.close()
  print(f"[finetune] {log_dir.resolve()}")
  return log_dir


if __name__ == "__main__":
  run(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
