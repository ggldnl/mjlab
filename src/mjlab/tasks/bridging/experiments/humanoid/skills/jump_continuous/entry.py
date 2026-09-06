"""What the jump costs when it is entered somewhere other than the clip's first frame.

The clips open with a second or two of standing still, and a policy started at frame zero
plays that back before it does anything. Skipping it needs no retraining in principle: the
skill is trained with reference state initialization over the whole clip, so the crouch is
a frame it has been reset into and asked to continue from thousands of times. Whether that
holds in practice is what this measures, because "the policy saw those frames" and "the
policy opens cleanly from those frames cold" are not the same claim.

Each entry mode is run over the same clips and stretches, and four numbers come back:

    survived    reached the end of the clip without a tracking failure
    reached     landed within the command's success threshold of the target
    error       how far off the landing was, in metres, over the episodes that survived
    sole        how far the lowest foot sits above the floor the instant the reset writes
                the entry pose, negative being underground

A mode that survives as well as start is a mode that costs nothing, and the standing is
free to go. One that does not is the case for retraining with the entry frames weighted.

sole is here because it is the one thing separating the two landmarks, and none of the
other three columns show it. The clips are shifted to stand on the ground and that shift is
capped on the ankle body origin, which stops describing the sole once the foot pitches, so
the bottom of the crouch sits several centimetres into the floor. As a tracking target that
is free. As a reset pose it opens the episode with the contact solver pushing the robot
back out of the ground.

Run

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.entry
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous import (
  JUMP_CONTINUOUS_TASK_ID,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.jump_continuous_env_cfg import (
  JumpCommandCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.mdp.commands import (
  JumpCommand,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

FOOT_CAPSULE_RADIUS = 0.01
"""Radius of the G1's foot collision capsules, subtracted from a capsule axis to reach the
sole. A wrong value here shifts every row by the same amount, so it cannot flip a
comparison, only the absolute number."""

# Entry modes to compare, as (label, sampling_mode, entry_landmark, entry_offset).
# "start" is the baseline and the behaviour every checkpoint was played with until now
MODES = (
  ("start", "start", "load", 0),
  ("load", "entry", "load", 0),
  ("crouch-10", "entry", "crouch", 10),
  ("crouch-5", "entry", "crouch", 5),
  ("crouch", "entry", "crouch", 0),
)


def find_checkpoint(experiment: str, explicit: Path | None) -> Path:
  """The newest checkpoint of an experiment, printed rather than assumed.

  Modification time picks it when nothing is given, which has gone wrong here before: a
  stale run left in logs/ outranks the one you meant, silently.
  """
  if explicit is not None:
    if not explicit.exists():
      raise SystemExit(f"No checkpoint at {explicit}.")
    return explicit
  root = Path("logs") / "rsl_rl" / experiment
  found = sorted(root.rglob("model_*.pt"), key=lambda p: p.stat().st_mtime)
  if not found:
    raise SystemExit(f"No checkpoint under {root}. Train '{experiment}' first.")
  return found[-1]


def run_mode(
  checkpoint: Path,
  sampling_mode: str,
  entry_landmark: str,
  entry_offset: int,
  num_envs: int,
  device: str,
) -> dict[str, float]:
  """Build the env with one entry mode, run every clip to its end, and score it.

  The environment is rebuilt per mode rather than mutated between runs. The command term
  reads its config at reset and caches nothing, so mutating would probably work, but a
  policy comparison that silently shares state between arms is not worth the seconds saved.
  """
  env_cfg = load_env_cfg(JUMP_CONTINUOUS_TASK_ID, play=True)
  env_cfg.scene.num_envs = num_envs
  # An episode has to end when the clip does, or nothing is ever scored. Play sets this to
  # effectively infinite so a viewer can watch one jump forever
  env_cfg.episode_length_s = 20.0

  motion_cfg = env_cfg.commands["motion"]
  assert isinstance(motion_cfg, JumpCommandCfg)
  # The modes are table-driven, so these arrive as plain strings rather than as the
  # literals the config declares
  motion_cfg.sampling_mode = cast(Any, sampling_mode)
  motion_cfg.entry_landmark = cast(Any, entry_landmark)
  motion_cfg.entry_offset = entry_offset

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  agent_cfg = load_rl_cfg(JUMP_CONTINUOUS_TASK_ID)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(JUMP_CONTINUOUS_TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  command = env.command_manager.get_term("motion")
  assert isinstance(command, JumpCommand)

  robot = env.scene["robot"]
  foot_geoms = torch.tensor(
    [i for i, n in enumerate(robot.geom_names) if "foot" in n and "collision" in n],
    dtype=torch.long,
    device=device,
  )

  obs, _ = wrapped.reset()

  # Read before the first step, while the robot is still exactly where the reset put it
  sole = (
    robot.data.geom_pos_w[:, foot_geoms, 2]
    - FOOT_CAPSULE_RADIUS
    - env.scene.env_origins[:, None, 2]
  ).min(dim=1)[0]

  # Scored once per env, on its first episode only. mjlab resets a terminated env inside
  # step and returns the post-reset observation, so an env that fell is upright and back at
  # its entry frame by the time step returns. Anything read afterwards describes the fresh
  # episode, not the one being judged
  done_once = torch.zeros(num_envs, dtype=torch.bool, device=device)
  survived = torch.zeros(num_envs, dtype=torch.bool, device=device)
  error = torch.full((num_envs,), float("nan"), device=device)
  entered_at = command.time_steps.clone()

  max_steps = int(command.motion.time_step_total) + 5
  with torch.no_grad():
    for _ in range(max_steps):
      # Latched before stepping, because the pre-reset value is gone once step returns. One
      # control step stale, which for a landing error is 20 ms after touchdown on a robot
      # that has stopped moving
      pending_error = command.goal_pos_error.clone()

      actions = policy(obs)
      obs, _, dones, _ = wrapped.step(actions)

      # The termination buffers are the exception: they are written during step and are
      # what the reset itself reads, so they still describe the episode that just ended.
      # motion_ended is registered time_out=True, so timing out is the clip finishing
      finished = dones.bool() & ~done_once
      if finished.any():
        timed_out = env.termination_manager.time_outs
        survived[finished] = timed_out[finished]
        error[finished] = pending_error[finished]
        done_once |= finished

      if bool(done_once.all()):
        break

  threshold = motion_cfg.goal_success_threshold
  ok = survived & torch.isfinite(error)
  reached = ok & (error < threshold)
  scored = done_once.float().sum().item()

  result = {
    "survived": float(survived.float().mean().item()),
    "reached": float(reached.float().mean().item()),
    "error": float(error[ok].mean().item()) if bool(ok.any()) else float("nan"),
    "entry_frame": float(entered_at.float().mean().item()),
    "sole": float(sole.mean().item()),
    "scored": scored,
  }
  env.close()
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--experiment", type=str, default="g1_jump_continuous")
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--device", type=str, default="cuda:0")
  args = parser.parse_args()

  checkpoint = find_checkpoint(args.experiment, args.checkpoint)
  print(f"Checkpoint: {checkpoint}\n")

  rows = []
  for label, mode, landmark, offset in MODES:
    torch.manual_seed(0)
    stats = run_mode(checkpoint, mode, landmark, offset, args.num_envs, args.device)
    rows.append((label, stats))
    print(f"  {label:<10} done")

  print(
    f"\n{'entry':<10} {'frame':>7} {'survived':>9} {'reached':>8} {'error':>8} {'sole':>9}"
  )
  for label, s in rows:
    print(
      f"{label:<10} {s['entry_frame']:>7.0f} {s['survived']:>8.1%} "
      f"{s['reached']:>7.1%} {s['error']:>7.3f} m {s['sole']:>+8.4f} m"
    )


if __name__ == "__main__":
  main()
