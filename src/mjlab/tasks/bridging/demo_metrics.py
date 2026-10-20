"""Headless demo outcomes and the phase in which execution stopped."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from mjlab.tasks.bridging.tests.stage import state


def perturb(env, seed: int) -> None:
  """Apply paired, small joint perturbations without changing task geometry."""
  robot = env.scene["robot"]
  generator = torch.Generator(device=env.device).manual_seed(seed)
  q, v = robot.data.joint_pos.clone(), robot.data.joint_vel.clone()
  q += 0.01 * torch.randn(q.shape, device=env.device, generator=generator)
  v += 0.02 * torch.randn(v.shape, device=env.device, generator=generator)
  robot.write_joint_state_to_sim(q, v)
  env.sim.forward()
  env.sim.sense()


@torch.inference_mode()
def trials(
  wrapped,
  controller,
  steps: int,
  demo: Literal["soccer", "sokoban"],
  path: Path,
  method: str,
  seeds: tuple[int, ...],
) -> None:
  """Reuse loaded policies, resetting the full scene and controller for each run."""
  from mjlab.tasks.bridging.evaluate import save

  if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
    raise ValueError("Seeds must be nonempty, nonnegative and unique")
  paths = [path.with_stem(f"{path.stem}_{seed}") for seed in seeds]
  if any(target.exists() for target in paths):
    raise FileExistsError("Use a new output directory for another demo measurement")
  for seed, target in zip(seeds, paths, strict=True):
    env = wrapped.unwrapped
    env.reset(seed=seed)
    perturb(env, seed)
    controller.reset()
    env.observation_manager.reset()
    env.observation_manager.compute(update_history=True)
    torch.manual_seed(seed)
    result = run(wrapped, controller, steps, demo)
    result.update(
      seed=seed,
      method=method,
      randomization={"joint_position_std_rad": 0.01, "joint_velocity_std_rad_s": 0.02},
    )
    save(target, result)
    print(
      f"{demo} {method} seed={seed}: {result['outcome']} ({result['seconds']:.2f}s)",
      flush=True,
    )


def outcome(demo: str, completed: bool, switch: str | None) -> str:
  if completed:
    return "completed"
  if demo == "sokoban":
    return "switch" if switch is not None else "action"
  return switch or "elsewhere"


@torch.inference_mode()
def run(wrapped, controller, steps: int, demo: Literal["soccer", "sokoban"]) -> dict:
  """Attribute falls during bridges and the next two seconds to that switch."""
  env = wrapped.unwrapped
  obs = wrapped.get_observations()
  initial = {
    name: torch.cat((entity.data.root_link_pos_w, entity.data.root_link_quat_w), -1)[0]
    .cpu()
    .tolist()
    for name, entity in env.scene.entities.items()
    if name in ("robot", "ball", "fallen_robot") or name.startswith("box_")
  }
  initial_state = state(env)[0].cpu().tolist() if "robot" in initial else None
  events: list[dict] = []
  switch, guard_until = None, -1
  reason = "Step limit reached"
  failed_at_bridge = False
  elapsed = 0
  for tick in range(steps):
    if tick > guard_until:
      switch = None
    if controller.done:
      reason = controller.reason
      break
    before = controller.phase
    handoffs = len(controller.handoffs) if demo == "soccer" else 0
    action = controller(obs)
    after = controller.phase
    handover = (
      len(controller.handoffs) > handoffs
      if demo == "soccer"
      else before == "bridge" and after == "execute"
    )
    if before == "bridge" or after == "bridge" or handover:
      switch = (
        controller.transition.split("_to_")[1]
        if demo == "soccer"
        else controller.entering
      )
      guard_until = tick + round(2.0 / env.step_dt)
    if before != after or handover:
      events.append(
        {"seconds": tick * env.step_dt, "from": before, "to": after, "switch": switch}
      )
    if controller.done:
      reason = controller.reason
      failed_at_bridge = before == "bridge" and after == "failed"
      break
    obs, _, _, _ = wrapped.step(action)
    elapsed += 1
    if tick + 1 > guard_until:
      switch = None
    robot = env.scene["robot"]
    actual = state(env)
    fallen = (
      not bool(torch.isfinite(actual).all())
      or float(actual[0, 2] - env.scene.env_origins[0, 2]) < 0.25
      or float(robot.data.projected_gravity_b[0, 2]) > -0.2
    )
    if fallen:
      reason = "Robot fell or state became nonfinite"
      break
  completed = controller.phase == "done"
  # Other task failures, such as a missed goal or action timeout, are separate
  fall = "fell" in reason.lower() or "nonfinite" in reason.lower()
  active = switch if fall or failed_at_bridge or controller.phase == "bridge" else None
  return {
    "outcome": outcome(demo, completed, active),
    "seconds": elapsed * env.step_dt,
    "phase": controller.phase,
    "reason": reason,
    "events": events,
    "initial_scene": initial,
    "initial_robot_state": initial_state,
    "attribution": "Falls during a bridge or within 2 seconds of its handover count at that switch; other task failures count elsewhere",
  }
