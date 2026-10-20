"""Replay a request, hand over independently, then measure only the incoming skill.

One request per source execution, shared by every method that uses it (TRIGGERS):

- timed: no-op, bridge and the walk2jump transition
- approach: only the walk2kick transition
- landing: every jump2walk method

Where the incoming entry goes per method: placement.
"""

# pyright: reportPrivateUsage=false, reportPrivateImportUsage=false

from collections.abc import Callable

import torch

from mjlab.tasks.bridging.baseline.model import TransitionPolicy
from mjlab.tasks.bridging.baseline.outcome import KICK_MIN_SPEED_M_S
from mjlab.tasks.bridging.baseline.scene import TransitionScene
from mjlab.tasks.bridging.config.g1.skills.kick import mdp as kick_mdp
from mjlab.tasks.bridging.scoring import metrics
from mjlab.tasks.bridging.tests.stage import state
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply

# warmup_s of walking before any request, also the history kept with it. A timed
# request adds up to phase_offset_s per seed, about one gait cycle, so each run
# switches at a different point of its stride. An approach starts warmup_s of
# walking plus up to start_offset_m per seed before its distance, for the same reason
WALK_SAMPLING = {"warmup_s": 3.0, "phase_offset_s": 1.25, "start_offset_m": 1.0}

TRIGGERS = {
  "timed": "walk straight from the reset pose, switch after warmup_s plus a seeded phase offset",
  "approach": "walk at the world fixed entry, switch on the first step within request_placement_m of it",
  "landing": "jump until the first landing",
}


def world_fixed(scene: TransitionScene, method: str) -> bool:
  """Whether the incoming reference and ball stay where the scene reset put them.

  Nominal starts on that placement. The walk2kick transition was trained to reach it
  from 0.4 m, so it keeps it. Every other method places the entry from the robot.
  """
  return method == "nominal" or (
    method == "transition" and scene.target_skill == "kick"
  )


def request_kind(scene: TransitionScene, method: str) -> str:
  if scene.source != "walk":
    return "landing"
  return "approach" if world_fixed(scene, method) else "timed"


def placement(scene: TransitionScene, method: str) -> str:
  """Where the incoming entry is, in words, for the report."""
  if world_fixed(scene, method):
    return "fixed in the world as in transition training, with the kick's ball"
  if method == "bridge":
    return "entry sequence placed distance_m ahead of the robot at the request, facing its heading, with the kick's ball"
  if method == "transition":
    return "placed on the robot when the transition hands over"
  return "placed on the robot at the switch, with the kick's ball"


def entry_gap(scene: TransitionScene) -> torch.Tensor:
  """Planar distance from the robot to the entry."""
  target = scene.run.command.target[:, :2]
  position = scene.robot.data.root_link_pos_w[:, :2]
  return torch.linalg.vector_norm(target - position, dim=-1)


def seeded_uniform(seeds: tuple[int, ...], device: str) -> torch.Tensor:
  """One draw in [0, 1) per seed, independent of the simulation RNG."""
  return torch.tensor(
    [float(torch.rand((), generator=torch.Generator().manual_seed(s))) for s in seeds],
    device=device,
  )


def _merge(request: dict | None, sample: dict, ready: torch.Tensor) -> dict:
  if request is None:
    return sample
  for name, value in sample.items():
    if name == "history":
      for saved, current in zip(request[name], value, strict=True):
        saved[ready] = current[ready]
    elif name == "objects":
      for entity, root in value.items():
        request[name][entity][ready] = root[ready]
    else:
      request[name][ready] = value[ready]
  return request


@torch.no_grad()
def walk_until(
  scene: TransitionScene,
  due: Callable[[int], torch.Tensor],
  heading: torch.Tensor | None = None,
) -> dict:
  """Walk and capture each robot on the first step it is due.

  heading None steers at the entry, otherwise the walk holds that world heading.
  A robot that falls is captured where it fell, one never due fails at the timeout.
  """
  dt = scene.env.step_dt
  device = scene.env.device
  keep = round(WALK_SAMPLING["warmup_s"] / dt)
  if scene.run.history_length > keep:
    raise ValueError("Source warmup must cover the bridge history")
  captured = torch.zeros_like(scene.failed)
  elapsed = torch.zeros(scene.env.num_envs, device=device)
  request: dict | None = None
  last = round(scene.cfg.source_timeout_s / dt)
  for tick in range(last + 1):
    if tick == last:
      scene.failed |= ~captured
    ready = ~captured & (due(tick) | scene.failed)
    if bool(ready.any()):
      sample = scene.capture_request()
      history = sample["history"][-keep:]
      if len(history) < keep:
        first = history[0] if history else sample["state"]
        history = [first.clone() for _ in range(keep - len(history))] + history
      sample["history"] = history
      request = _merge(request, sample, ready)
      captured |= ready
      elapsed[ready] = tick * dt
    if bool(captured.all()):
      break
    scene.hold_target()
    if heading is not None:
      scene.steer(heading)
    scene.step(scene.skill_action(scene.source))
  assert request is not None
  request["source_walk_s"] = elapsed
  return request


def timed_request(scene: TransitionScene, seeds: tuple[int, ...]) -> dict:
  """Walk straight along the reset heading and switch at a seeded time.

  The robot starts where transition training starts it. The kick's ball is parked
  behind it, it is placed for the incoming skill at the switch.
  """
  dt = scene.env.step_dt
  offsets = WALK_SAMPLING["phase_offset_s"] * seeded_uniform(seeds, scene.env.device)
  due_s = WALK_SAMPLING["warmup_s"] + offsets
  if scene.cfg.source_timeout_s <= float(due_s.max()):
    raise ValueError("Source timeout is too short for the timed request")
  due_tick = (due_s / dt).round().long()
  if scene.target_skill == "kick":
    scene.park_ball()
  _, _, heading = euler_xyz_from_quat(state(scene.env)[:, 3:7])
  request = walk_until(scene, lambda tick: tick >= due_tick, heading)
  request["source_phase_offset_s"] = offsets
  return request


def approach_request(
  scene: TransitionScene, seeds: tuple[int, ...], distance: float
) -> dict:
  """Walk at the world fixed entry and switch within distance of it.

  The reference stays frozen on its entry frame and the ball stays where the scene
  put it. The robot starts back along x by the warmup walk plus a seeded offset.
  """
  offsets = WALK_SAMPLING["start_offset_m"] * seeded_uniform(seeds, scene.env.device)
  back = distance + scene.cfg.walk_speed * WALK_SAMPLING["warmup_s"] + offsets
  if scene.cfg.source_timeout_s * scene.cfg.walk_speed <= float(back.max()):
    raise ValueError("Source timeout is too short to walk to the request")
  initial = state(scene.env).clone()
  initial[:, 0] = scene.run.command.target[:, 0] - back
  scene.run._teleport(initial)
  scene.env.sim.forward()
  request = walk_until(scene, lambda tick: entry_gap(scene) <= distance)
  request["source_start_offset_m"] = offsets
  return request


@torch.no_grad()
def landing_request(scene: TransitionScene) -> dict:
  """Jump until the first landing."""
  airborne = torch.zeros(scene.env.num_envs, dtype=torch.bool, device=scene.env.device)
  landed = airborne.clone()
  warmup = round(1.0 / scene.env.step_dt)
  if scene.run.history_length > warmup:
    raise ValueError("Source warmup must cover the bridge history")
  for _ in range(round(scene.cfg.source_timeout_s / scene.env.step_dt)):
    scene.run._set_walk()
    found = scene.env.scene["feet_ground_contact"].data.found
    grounded = found.reshape(scene.env.num_envs, -1).gt(0).any(-1)
    height = scene.robot.data.body_link_pos_w[:, scene.run.feet, 2]
    height = height - scene.env.scene.env_origins[:, 2, None]
    airborne |= ~grounded & (height.min(-1).values > 0.08)
    landed |= airborne & grounded & scene.run.motion.has_landed
    if bool((landed | scene.failed).all()):
      break
    scene.step(scene.skill_action(scene.source))
  else:
    scene.failed |= ~landed
  return scene.capture_request()


def add_noise(scene: TransitionScene, seeds: tuple[int, ...]) -> None:
  q, v = scene.robot.data.joint_pos.clone(), scene.robot.data.joint_vel.clone()
  for row, seed in enumerate(seeds):
    generator = torch.Generator(device=scene.env.device).manual_seed(seed)
    q[row] += 0.01 * torch.randn(
      q.shape[1], device=scene.env.device, generator=generator
    )
    v[row] += 0.02 * torch.randn(
      v.shape[1], device=scene.env.device, generator=generator
    )
  scene.robot.write_joint_state_to_sim(q, v)
  scene.env.sim.forward()


@torch.no_grad()
def source_request(
  scene: TransitionScene,
  seeds: tuple[int, ...],
  kind: str,
  distance: float | None = None,
) -> dict:
  """Run the source from a perturbed reset up to its request, see the module docstring."""
  if len(seeds) != scene.env.num_envs or len(set(seeds)) != len(seeds):
    raise ValueError("One distinct seed is required per environment")
  if (kind == "landing") != (scene.source != "walk"):
    raise ValueError(f"A {scene.source} source has no {kind} request")
  scene.reset(seeds[0])
  add_noise(scene, seeds)
  if kind == "timed":
    return timed_request(scene, seeds)
  if kind == "approach":
    if distance is None:
      raise ValueError("An approach request needs a distance")
    return approach_request(scene, seeds, distance)
  return landing_request(scene)


def kick_direction(scene: TransitionScene) -> torch.Tensor | None:
  """Direction of the kick reference in world axes, None for other skills."""
  if scene.target_skill != "kick":
    return None
  local = scene.run.motion.kick_direction
  return quat_apply(
    scene.run.motion.anchor_yaw_quat,
    torch.cat((local, local.new_zeros(len(local), 1)), -1),
  )


def bridge_action(scene: TransitionScene, duration: float):
  history = torch.stack(scene.history[-scene.run.history_length :], 1)
  missing = scene.run.history_length - history.shape[1]
  if missing > 0:
    history = torch.cat((history[:, :1].expand(-1, missing, -1), history), 1)
  output = scene.run.runtime_bridges[scene.bridge_name](
    history,
    scene.run.target_window,
    torch.full((scene.env.num_envs,), duration, device=scene.env.device),
  )
  return output.action, output.handoff


@torch.no_grad()
def measure(
  scene: TransitionScene,
  method: str,
  seeds: tuple[int, ...],
  errors: metrics.FailureErrors,
  duration: float | None = None,
  policy: TransitionPolicy | None = None,
  transition_steps: int = 100,
  distance: float | None = None,
) -> list[dict]:
  n, device, dt = scene.env.num_envs, scene.env.device, scene.env.step_dt
  if len(seeds) != n:
    raise ValueError("One distinct seed is required per environment")
  if method not in ("nominal", "no-op", "transition", "bridge"):
    raise ValueError(f"Unknown scoring method: {method}")
  if method == "no-op" and duration is not None:
    raise ValueError("An instant switch has no duration")
  if method == "bridge" and (duration is None or distance is None):
    raise ValueError("A bridge needs a distance and a duration")
  request = scene.capture_request()
  # Joint targets the robot holds at the request, then one per applied action
  targets = [scene.joint_targets(scene.env.action_manager.action)]
  scene.hold_target()
  limits = scene.downstream_steps()
  faults = {
    name: torch.zeros(n, dtype=torch.bool, device=device)
    for name in (
      "source_failed",
      "transition_failed",
      "no_handoff",
      "contact_before_handoff",
      "missed_ball",
      "weak_kick",
      "no_landing",
    )
  }
  faults["source_failed"] |= scene.failed
  finished = scene.failed.clone()
  instant = method in ("nominal", "no-op")
  active = torch.zeros_like(finished) if instant else ~finished
  incoming = ~finished if instant else torch.zeros_like(finished)
  handed = incoming.clone()
  age = torch.zeros(n, dtype=torch.long, device=device)
  elapsed = torch.zeros(n, dtype=torch.long, device=device)
  start_state = torch.zeros_like(request["state"])
  start_state[incoming] = request["state"][incoming]
  success = torch.zeros_like(finished)
  names = metrics.METRICS[scene.target_skill]
  totals = {name: torch.zeros(n, device=device) for name in names}
  touched = torch.zeros_like(finished)
  native_faults = {
    name: torch.zeros_like(finished) for name in scene.target_failures.active_terms
  }
  budget = (
    0 if instant else round(duration / dt) if duration is not None else transition_steps
  )
  fixed = world_fixed(scene, method)
  if method == "no-op":
    # Nothing is placed for an instant switch, the incoming skill starts on the robot
    scene.place_incoming(~finished)
  if method == "bridge":
    assert duration is not None and distance is not None
    # The entry sequence goes distance metres ahead, the bridge has duration to reach it
    scene.place_incoming(torch.ones_like(finished), distance)
    scene.open_bridge(duration)
  else:
    scene.run.command.stop()
  if scene.target_skill == "kick":
    # A source or bridge start already touching the ball cannot be credited
    faults["contact_before_handoff"] |= kick_mdp.touching_ball(scene.env) & ~finished
  direction = kick_direction(scene)
  reason = (
    "instant_switch"
    if method == "no-op"
    else "nominal"
    if method == "nominal"
    else "not_reached"
  )
  termination = [reason for _ in seeds]
  for tick in range(budget + int(limits.max()) + 1):
    if bool(finished.all()):
      break
    scene.hold_target(active)
    action = scene.env.action_manager.action.clone()
    stop = torch.zeros_like(finished)
    forced = torch.zeros_like(finished)
    if bool(active.any()):
      if method == "transition":
        if policy is None:
          raise ValueError("A trained transition checkpoint is required")
        action, _, decision = policy.act(scene.observe())
        forced = active & (tick == budget - 1)
        stop = decision.bool() | forced
      elif method == "bridge":
        assert duration is not None
        action, stop = bridge_action(scene, duration)
        timed_out = active & ~stop & (tick == budget)
        faults["no_handoff"] |= timed_out
        elapsed[timed_out] = tick
        for row in torch.nonzero(timed_out).flatten().tolist():
          termination[row] = "bridge_timeout"
        finished |= timed_out
        active &= ~timed_out
      else:
        raise ValueError(f"Unknown scoring method: {method}")
    starting = active & stop
    if bool(starting.any()):
      if method == "transition" and not fixed:
        # The transition chose where the incoming skill starts
        scene.place_incoming(starting)
      start_state[starting] = state(scene.env)[starting]
      elapsed[starting] = tick
      for row in torch.nonzero(starting).flatten().tolist():
        termination[row] = (
          "transition_limit"
          if bool(forced[row])
          else "learned_stop"
          if method == "transition"
          else "bridge_stop"
        )
    handed |= starting
    incoming |= starting
    active &= ~starting
    if bool(finished.all()):
      break
    successor = scene.skill_action(scene.target_skill)
    if not bool(active.any()):
      scene.run.command.stop()
    applied = torch.where(active[:, None], action, successor)
    targets.append(scene.joint_targets(applied))
    scene.step(applied)
    native = scene.target_failures.compute() & incoming
    scene.failed |= native
    for name, mask in native_faults.items():
      mask |= scene.target_failures.get_term(name) & incoming
    if scene.target_skill == "walk":
      data = scene.robot.data
      desired = torch.tensor([scene.run.walk_speed, 0.0, 0.0], device=device).expand(
        n, -1
      )
      values = metrics.walk(
        data.root_link_lin_vel_b, data.root_link_ang_vel_b[:, 2], desired
      )
    elif scene.target_skill == "jump":
      values = metrics.jump(
        scene.run.motion.robot_body_pos_w, scene.run.motion.body_pos_relative_w
      )
    else:
      contact = kick_mdp.touching_ball(scene.env)
      faults["contact_before_handoff"] |= contact & active
      touched |= contact & incoming
      assert direction is not None
      speed = metrics.kick(kick_mdp.ball_vel_w(scene.env), direction, touched)
      name = "forward_ball_speed_m_s"
      totals[name] = torch.where(
        incoming, torch.maximum(totals[name], speed), totals[name]
      )
      values = {}
    for name, value in values.items():
      if not bool(torch.isfinite(value[incoming]).all()):
        raise RuntimeError(f"Nonfinite {name}; refusing to publish an average")
      totals[name][incoming] += value[incoming]
    age += incoming.long()
    crashed = active & scene.failed
    faults["transition_failed"] |= crashed
    elapsed[crashed] = tick + 1
    finished |= crashed
    active &= ~crashed
    complete = incoming & ((age >= limits) | scene.failed)
    if scene.target_skill == "kick":
      faults["missed_ball"] |= complete & ~touched
      faults["weak_kick"] |= (
        complete & touched & (totals["forward_ball_speed_m_s"] <= KICK_MIN_SPEED_M_S)
      )
    if scene.target_skill == "jump":
      faults["no_landing"] |= complete & ~scene.run.motion.has_landed
    valid = ~scene.failed
    for mask in faults.values():
      valid &= ~mask
    success[complete] = valid[complete]
    finished |= complete
    incoming &= ~complete
  if not bool(finished.all()):
    raise RuntimeError("Scoring loop did not finish every run")
  # A trial's own steps end where it finished, the batch may step on without it
  step_rms, step_peak = metrics.action_steps(torch.stack(targets, 1).cpu())
  rows = []
  for row, seed in enumerate(seeds):
    measured = {
      name: (
        float(value[row])
        if scene.target_skill == "kick"
        else float(value[row] / age[row])
      )
      if int(age[row])
      else None
      for name, value in totals.items()
    }
    reasons = [
      name for name, mask in {**faults, **native_faults}.items() if bool(mask[row])
    ]
    rows.append(
      {
        "seed": seed,
        "success": bool(success[row]),
        "handoff": bool(handed[row]),
        "termination": termination[row],
        "failure_reasons": reasons,
        "transition_s": float(elapsed[row] * dt),
        "post_steps": int(age[row]),
        "post_horizon_s": float(limits[row] * dt),
        "contact": bool(touched[row]) if scene.target_skill == "kick" else None,
        "measurements": measured,
        "scores": metrics.failure_scores(
          scene.target_skill, measured, bool(success[row]), errors
        ),
        "action": metrics.action_discontinuity(
          step_rms[row],
          step_peak[row],
          int(elapsed[row] + age[row]),
          int(elapsed[row]) if bool(handed[row]) else None,
        ),
        "request_state": request["state"][row].cpu().tolist(),
        # Where the incoming entry ended up, after any placement
        "reference_entry": scene.run.target[row].cpu().tolist(),
        "incoming_start_state": start_state[row].cpu().tolist()
        if bool(handed[row])
        else None,
      }
    )
  return rows
