"""Throwaway: run the T1 jump's teacher, the phase one tracker, before distillation.

Also runs the distilled student on the same clips with --policy student, which separates a
student that never learned the teacher from one that only breaks in the deploy env.

The teacher reads the reference (joint targets, phase, goal, anchor offset), so this runs on
the clip command, not the distance one. It loads a checkpoint saved during phase one, which
holds the teacher as a plain PPO actor. Without --checkpoint, the newest run's last phase
one checkpoint is picked and printed.

Two modes

    evaluate    headless. Every clip from its first frame with no stretch, several envs
                per clip, and a table per clip: how many survived, where the failures
                happened relative to takeoff and landing, which termination fired, and
                how far from the goal the survivors landed, and how many left the ground
                with both feet during the clip's flight
    view        the viewer, with the reference ghost next to the robot

Run

1. The table.

    uv run python -m mjlab.tasks.bridging.config.t1.skills.jump.play_teacher

2. Watch it, clips drawn at random, each started where training starts them.

    uv run python -m mjlab.tasks.bridging.config.t1.skills.jump.play_teacher \
      --mode view --num-envs 4

3. The distilled student on the same table, the newest run's last checkpoint.

    uv run python -m mjlab.tasks.bridging.config.t1.skills.jump.play_teacher       --policy student

4. A given checkpoint.

    uv run python -m mjlab.tasks.bridging.config.t1.skills.jump.play_teacher \
      --checkpoint logs/rsl_rl/t1_jump_continuous/<run>/model_3999.pt
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import (
  MjlabOnPolicyRunner,
  MjlabTeacherStudentRunner,
  RslRlVecEnvWrapper,
)
from mjlab.tasks.bridging.config.g1.skills.jump_continuous import (
  jump_ppo_runner_cfg,
  jump_runner_cfg,
)
from mjlab.tasks.bridging.config.g1.skills.jump_continuous.mdp import (
  JumpCommand,
  JumpCommandCfg,
)
from mjlab.tasks.bridging.config.t1.skills.jump.jump_env_cfg import (
  t1_jump_continuous_env_cfg,
)

EXPERIMENT = "t1_jump_continuous"


def find_checkpoint(explicit: Path | None, student: bool) -> Path:
  """The newest run's last checkpoint of the asked for phase."""
  if explicit is not None:
    return explicit
  runs = sorted(
    (Path("logs") / "rsl_rl" / EXPERIMENT).glob("*/"), key=lambda p: p.stat().st_mtime
  )
  if not runs:
    raise SystemExit(f"No runs under logs/rsl_rl/{EXPERIMENT}.")
  checkpoints = sorted(runs[-1].glob("model_*.pt"), key=lambda p: int(p.stem[6:]))
  for path in reversed(checkpoints):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if ("student_state_dict" in saved) == student:
      return path
  raise SystemExit(
    f"No {'student' if student else 'teacher'} checkpoint in {runs[-1]}."
  )


def build(num_envs: int, device: str, checkpoint: Path, student: bool):
  """The clip env in play mode, and the policy reading its group."""
  cfg = t1_jump_continuous_env_cfg(play=True)
  cfg.scene.num_envs = num_envs
  # Play makes episodes endless for watching. The clip ending still ends them
  cfg.episode_length_s = 20.0
  motion = cfg.commands["motion"]
  assert isinstance(motion, JumpCommandCfg)
  # Where training starts most episodes that reach the jump: the trimmed clip's first frame
  motion.sampling_mode = "start"
  env = ManagerBasedRlEnv(cfg=cfg, device=device)

  if student:
    # The deployment shape: the student as the actor, reading "actor"
    agent = jump_runner_cfg(EXPERIMENT)
    runner_cls = MjlabTeacherStudentRunner
  else:
    agent = jump_ppo_runner_cfg(EXPERIMENT)
    agent.obs_groups = {"actor": ("teacher",), "critic": ("critic",)}
    runner_cls = MjlabOnPolicyRunner
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = runner_cls(wrapped, asdict(agent), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  return env, wrapped, runner.get_inference_policy(device=device)


def evaluate(env: ManagerBasedRlEnv, wrapped: RslRlVecEnvWrapper, policy) -> None:
  """Every clip from frame zero, one table row per clip."""
  command = env.command_manager.get_term("motion")
  assert isinstance(command, JumpCommand)
  lib = command.motion
  n = env.num_envs
  ids = torch.arange(n, device=env.device)

  wrapped.reset()
  command.motion_ids[:] = ids % lib.num_motions
  command.scales[:] = 1.0
  command.time_steps[:] = 0
  command.motion_done[:] = False
  command.place_on_reference(ids)
  env.sim.forward()
  command.update_relative_body_poses()
  obs = wrapped.get_observations()

  done = torch.zeros(n, dtype=torch.bool, device=env.device)
  survived = torch.zeros(n, dtype=torch.bool, device=env.device)
  end_frame = torch.zeros(n, dtype=torch.long, device=env.device)
  error = torch.full((n,), float("nan"), device=env.device)
  cause = [""] * n
  names = env.termination_manager.active_terms
  # Both feet off the ground at some step of the reference's flight
  airborne = torch.zeros(n, dtype=torch.bool, device=env.device)
  feet = env.scene["feet_ground_contact"]
  takeoff = command.takeoff_steps_all[command.motion_ids]
  land = command.land_steps_all[command.motion_ids]

  with torch.no_grad():
    for _ in range(lib.time_step_total + 5):
      # Read before stepping: a finished env is reset inside step
      frame, goal_error = command.time_steps.clone(), command.goal_pos_error.clone()
      in_flight = (frame >= takeoff) & (frame <= land) & ~done
      obs, _, dones, _ = wrapped.step(policy(obs))
      off = (feet.data.found == 0).all(dim=-1)
      airborne |= in_flight & off & ~dones.bool()
      new = dones.bool() & ~done
      for i in new.nonzero(as_tuple=True)[0].tolist():
        fired = [k for k in names if bool(env.termination_manager.get_term(k)[i])]
        cause[i] = ",".join(fired)
        survived[i] = bool(env.termination_manager.time_outs[i])
        end_frame[i] = frame[i]
        error[i] = goal_error[i]
      done |= new
      if bool(done.all()):
        break

  print(
    f"\n{'clip':<44} {'dist':>5} {'ok':>5} {'air':>5} {'fail at':>18} "
    f"{'causes':<28} {'land err':>8}"
  )
  for m, meta in enumerate(lib.metadata):
    rows = (ids % lib.num_motions == m).nonzero(as_tuple=True)[0]
    ok = survived[rows]
    failed = rows[~ok]
    where = ""
    if len(failed):
      f = end_frame[failed].float().mean().item()
      where = f"{f:5.0f} (to {meta.takeoff_step}, ld {meta.land_step})"
    causes = Counter(c for i in failed.tolist() for c in cause[i].split(",") if c)
    err = error[rows[ok]]
    print(
      f"{meta.name[:44]:<44} {meta.distance:5.2f} {int(ok.sum()):>2}/{len(rows):<2} "
      f"{int(airborne[rows].sum()):>2}/{len(rows):<2} "
      f"{where:>18} {', '.join(f'{k} {v}' for k, v in causes.items())[:28]:<28} "
      f"{(err.mean().item() if len(err) else float('nan')):8.3f}"
    )
  print(
    f"\nsurvived {survived.float().mean().item():.0%}, airborne "
    f"{airborne.float().mean().item():.0%}, landing error of the survivors "
    f"{error[survived].mean().item():.3f} m. 'fail at' is the mean frame of the failures, "
    "against the clip's takeoff and landing frames."
  )


def main(
  mode: Literal["evaluate", "view"] = "evaluate",
  policy: Literal["teacher", "student"] = "teacher",
  checkpoint: Path | None = None,
  num_envs: int = 256,
  viewer: Literal["native", "viser"] = "native",
  device: str = "cuda:0",
) -> None:
  """Run the teacher.

  Args:
    mode: "evaluate" for the per clip table, "view" for the viewer.
    policy: The phase one teacher, or the distilled student.
    checkpoint: A checkpoint of that phase. Defaults to the newest run's last one.
    num_envs: Environments. In evaluate, spread evenly over the clips.
    viewer: Viewer backend for "view".
    device: Torch device.
  """
  student = policy == "student"
  path = find_checkpoint(checkpoint, student)
  print(f"{policy.capitalize()}: {path}")
  env, wrapped, act = build(num_envs, device, path, student)
  if mode == "evaluate":
    evaluate(env, wrapped, act)
  else:
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    if viewer == "native":
      NativeMujocoViewer(wrapped, act).run()
    else:
      ViserPlayViewer(wrapped, act).run()
  env.close()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
