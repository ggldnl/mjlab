"""What the bridge is being asked for, and what it would be worth, before there is a bridge.

Run:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.handoff
    uv run python -m ...tests.handoff --couple walk2jump --walk-steps 200

Needs the two skills' checkpoints and nothing else. No bridge policy and no bridge corpus.

##
# What it does
##

Walking is interrupted mid-stride, and from that one interrupt the same hand-over is run
twice:

    none      the entering skill takes over from wherever walking left the robot. No bridge.
              This is the problem the project exists for, measured rather than asserted.
    perfect   the robot is teleported into the entry state the selector asks for, and the
              entering skill takes over from there. An oracle bridge, one that arrives
              exactly, instantly and for free.

Neither is a bridge. `perfect` is the ceiling a real one is chasing and `none` is the floor
it has to clear, and the gap between them is the whole value of the component. If that gap is
small the hand-over was never hard and this is the wrong couple to be testing against; if
`perfect` itself fails, the entry state is wrong and no bridge will rescue it.

Which is the point of running this first. A trained bridge that fails tells you nothing on
its own, because a bad entry state and a bad bridge fail identically. This separates them
while the bridge does not exist yet, so when one does, its number has two known-good
reference points either side of it.

##
# Why walk to the punch combination
##

Because the combination's opening is not a standing pose. It is tracked from a LAFAN1 fight
clip, and its tracker resets with the reference at frame zero, which is a guard: knees near
seventy degrees, hips folded unevenly, torso turned into the lead shoulder, both elbows drawn
in. Sixty-eight degrees from the robot's default pose at the worst joint.

So the bridge is asked for a posture that walking never passes through and that no amount of
standing up straight approximates. Compare that with handing over to walk or run, where the
entry is the pose the robot is already more or less in whenever walking stops: a bridge can
score well on those having learned nothing.

Nothing is on the floor for this couple either. A ball or a crate makes a hand-over partly
about where the robot ends up in the world, and a good arrival with the object out of place
still fails. Here the only question left is whether the robot arrived in the pose and at the
velocities the entry frame asks for, which is the question the bridge exists to answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.commands import (
  CHANNELS,
  Tolerances,
  arm_mask,
  channel_errors,
)
from mjlab.tasks.bridging.experiments.humanoid.selector import Selector
from mjlab.tasks.bridging.experiments.humanoid.skills.jump.mdp.commands import (
  JumpCommand,
)
from mjlab.tasks.bridging.experiments.humanoid.tests import actors
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import (
  ROBOT,
  Couple,
  Policy,
  arena,
  crossing,
  facing,
  find_checkpoint,
  state,
)
from mjlab.tasks.registry import load_rl_cfg
from mjlab.utils.lab_api.math import yaw_quat

COUPLES: dict[str, Couple] = {
  "walk2punch_combo": Couple(leaving=actors.WALK, entering=actors.PUNCH_COMBO),
  "walk2front_kick": Couple(leaving=actors.WALK, entering=actors.FRONT_KICK),
  "walk2jump": Couple(leaving=actors.WALK, entering=actors.JUMP, duration_s=0.7),
}
"""The couples worth staging this way: an entering skill whose opening is not a stand.

walk2punch_combo first, and by some distance. The combination's entry is the furthest from
anything walking produces, it needs nothing on the floor, and its tracker is anchored to a
single clip so there is no goal to get wrong on top of the pose.
"""


@dataclass
class HandoffCfg:
  couple: str = "walk2punch_combo"
  entry: int = 0
  """Which state off the entering skill's shortlist to aim at."""

  walk_steps: int = 150
  """Control steps of walking before the interrupt. Three seconds at 50 Hz, long enough that
  the gait has settled and the robot is genuinely mid-stride rather than still leaving its
  reset."""
  entering_steps: int = 220
  """How long the entering skill drives once it has taken over."""
  speed: float = 1.0
  """Forward command for the walk, in m/s."""

  device: str = "cuda:0"
  seed: int = 0
  walk_checkpoint: Path | None = None
  entering_checkpoint: Path | None = None


@dataclass
class Outcome:
  """One hand-over, scored."""

  mode: str
  errors: torch.Tensor
  """(8,) how far the robot was from the entry state when the skill took over."""
  earned: float
  """Discounted return the entering skill collected, over the window's own discount mass."""
  fell: bool
  steps: int


def _drive(
  env: ManagerBasedRlEnv, policy: Policy, obs, steps: int, run: Run | None = None
):
  """Step one policy for a while. Returns the last observation."""
  for _ in range(steps):
    obs, _, _, _, _ = env.step(policy(obs))
    if run is not None and run.watch():
      break
  return obs


class Run:
  """Accumulates the entering skill's discounted return, the way a hand-over is judged.

  Discounted rather than averaged, for the reason the selector was rebuilt around: a skill
  that stumbles out of a bad entry and recovers a second later still collects a good flat
  mean, and that is exactly the hand-over this is supposed to catch. At the skills' own gamma
  the recovery is worth almost nothing against the stumble.

  A fall stops the accumulation instead of ending the run. Nothing in this arena terminates,
  so a fallen robot lies there collecting whatever a prone robot collects, and counting that
  would flatter the transition.
  """

  def __init__(self, env: ManagerBasedRlEnv, discount: float, steps: int) -> None:
    self.env, self.discount, self.limit = env, discount, steps
    self.robot: Entity = env.scene[ROBOT]
    self.earned, self.scored, self.fell = 0.0, 0, False
    self.mass = (1.0 - discount**steps) / max(1.0 - discount, 1e-9)

  def watch(self) -> bool:
    """Score this step. True once there is nothing left to score."""
    if self.fell or self.scored >= self.limit:
      return True
    # Projected gravity, not root height: a deep crouch and a fall reach the same height and
    # only one of them is a failure, and half these entry states are deep crouches
    if float(self.robot.data.projected_gravity_b[0, 2]) > -0.7:
      self.fell = True
      return True
    self.earned += self.discount**self.scored * float(self.env.reward_buf[0])
    self.scored += 1
    return False


def reference_state(env: ManagerBasedRlEnv) -> torch.Tensor | None:
  """Where a clip tracker says it wants the robot, right now. (1, 13 + 2J), or None.

  For a skill trained by imitation this outranks anything the selector reconstructs, and the
  difference is not small. `anchor_to_robot` takes the direction the clip should travel, and
  a clip's travel direction is not the yaw its pelvis holds at frame zero: the two differ by
  whatever the performer's hips were doing, and here that is most of a right angle. Aiming at
  a state built by stripping the yaw off the clip and re-applying the robot's meant the robot
  arrived turned away from its own reference, which the tracker reads as an enormous tracking
  error on frame one and never recovers from. The oracle fell in half a second while the same
  policy ran the whole clip cleanly in its own environment.

  So the reference is asked instead of reconstructed. `body_quat_w` is the anchored clip's
  actual world pose, which is the thing the tracking reward compares the robot against, and
  therefore the only definition of "arrived" that skill agrees with.

  None when the entering skill has no reference to ask, which is every skill that was trained
  by reward rather than by imitation. Those keep the selector's own state.
  """
  if "motion" not in env.command_manager.active_terms:
    return None
  command = env.command_manager.get_term("motion")
  if not isinstance(command, JumpCommand):
    return None
  return torch.cat(
    [
      command.body_pos_w[:, 0],
      command.body_quat_w[:, 0],
      command.body_lin_vel_w[:, 0],
      command.body_ang_vel_w[:, 0],
      command.joint_pos,
      command.joint_vel,
    ],
    dim=-1,
  )


def teleport(env: ManagerBasedRlEnv, robot: Entity, target: torch.Tensor) -> None:
  """Put the robot exactly in this state. The oracle bridge.

  Everything `BridgeCommand.place` does except drawing a window, including the entity reset:
  qpos and qvel are not the whole state, since the action term holds the last action it
  applied and the observation terms hold their history. A robot moved without clearing them
  hands the next skill a step of somebody else's episode.

  Then a forward pass, which is the part that is easy to leave out and expensive to leave
  out. `write_root_state_to_sim` writes qpos and qvel; `root_link_pos_w` and the rest are
  derived from them by the simulator, so until it runs they still describe the robot that was
  there before. Reading the arrival straight after the write said the oracle had missed the
  root by half a metre and was carrying a metre per second of walking momentum, which is also
  what the robot then did: the pose was the guard and the velocity was still the walk, and it
  fell over in half a second. The teleport was fine. The read was a step early.
  """
  joints = robot.data.joint_pos.shape[1]
  env_ids = torch.arange(1, device=robot.data.joint_pos.device)
  robot.write_joint_state_to_sim(
    target[:, 13 : 13 + joints], target[:, 13 + joints :], env_ids=env_ids
  )
  robot.write_root_state_to_sim(target[:, 0:13], env_ids=env_ids)
  robot.reset(env_ids=env_ids)
  env.sim.forward()


def stage(cfg: HandoffCfg, couple: Couple, mode: str, env, policies, entry) -> Outcome:
  """Walk, interrupt, optionally teleport into the entry state, then hand over.

  The walk is replayed from the same seed rather than the interrupt being saved and restored,
  so both modes are interrupted in the same state without this file having to know what all
  of a state is. It is slower and it cannot be subtly wrong.
  """
  robot: Entity = env.scene[ROBOT]
  torch.manual_seed(cfg.seed)
  obs, _ = env.reset(seed=cfg.seed)

  # Both skills get their world before anything moves. The entering one needs its clip
  # anchored somewhere even during the walk, or its observation is reading a reference that
  # was never placed
  here = state(robot)
  for actor in (couple.entering, couple.leaving):
    if actor.enter:
      actor.enter(env, here[:, 0:3], here[:, 3:7])

  twist = env.command_manager.get_term("twist")
  twist.vel_command_b[:, 0] = cfg.speed
  twist.vel_command_b[:, 1:] = 0.0
  obs = _drive(env, policies[couple.leaving.name], obs, cfg.walk_steps)

  here = state(robot)
  target = facing(entry.state, here)
  target[:, 0:2] = crossing(here, target, entry.duration_s)

  # Anchored once, here, at where the robot is meant to end up and with the heading it has
  # now. Two details, each of which cost a run to find.
  #
  # A heading, not the robot's orientation. `anchor_to_robot` takes the direction the clip
  # plays along, and a target pose carries the entry's own tilt: the guard leans, so handing
  # it the full quaternion rotates the entire reference by that lean and the policy spends
  # the episode chasing a clip pitched into the floor.
  #
  # And before the crossing, not after it. Anchoring again once the robot has arrived slides
  # the clip onto wherever it actually got to, which erases the arrival error instead of
  # leaving it for the entering skill to cope with. Erasing it is the one thing this test
  # must not do: the arrival error is the entire subject.
  heading = yaw_quat(here[:, 3:7])
  if couple.entering.enter:
    couple.entering.enter(env, target[:, 0:3], heading)

  # Now ask the entering skill where it actually wants the robot, and believe it over the
  # reconstruction. See `reference_state`
  reference = reference_state(env)
  if reference is not None:
    target = reference.clone()

  if mode == "perfect":
    teleport(env, robot, target)
    obs = env.get_observations()

  errors = channel_errors(
    state(robot), target, arm_mask(tuple(robot.joint_names), env.device)
  )

  discount = float(
    getattr(load_rl_cfg(couple.entering.task).algorithm, "gamma", 0.99)  # ty: ignore[unresolved-attribute]
  )
  run = Run(env, discount, cfg.entering_steps)
  _drive(env, policies[couple.entering.name], obs, cfg.entering_steps, run)
  return Outcome(mode, errors[0], run.earned / run.mass, run.fell, run.scored)


@dataclass
class Entry:
  """The state the selector asks for, and how long the bridge would get to reach it."""

  state: torch.Tensor
  name: str
  why: str
  duration_s: float


def report(couple: Couple, entry: Entry, outcomes: list[Outcome]) -> None:
  requirements = Tolerances().as_tensor(outcomes[0].errors.device)
  width = 11

  print()
  print(f"{couple.leaving.name} -> {couple.entering.name}, aiming at '{entry.name}'")
  print(f"  {entry.why}")
  print(f"  the bridge would get {entry.duration_s:.2f} s to close this:")
  print()
  head = f"{'channel':<16}{'requires':>10}" + "".join(
    o.mode.rjust(width) for o in outcomes
  )
  print(head)
  print("-" * len(head))
  for index, channel in enumerate(CHANNELS):
    cells = "".join(f"{float(o.errors[index]):>{width}.3f}" for o in outcomes)
    print(f"{channel:<16}{float(requirements[index]):>10.2f}{cells}")

  print()
  print(f"{'entering skill':<26}" + "".join(o.mode.rjust(width) for o in outcomes))
  print("-" * len(head))
  print(
    f"{'discounted return':<26}" + "".join(f"{o.earned:>{width}.3f}" for o in outcomes)
  )
  print(
    f"{'fell':<26}"
    + "".join(("yes" if o.fell else "no").rjust(width) for o in outcomes)
  )
  print(f"{'steps scored':<26}" + "".join(f"{o.steps:>{width}d}" for o in outcomes))

  by_mode = {o.mode: o for o in outcomes}
  floor, ceiling = by_mode.get("none"), by_mode.get("perfect")
  print()
  if floor is None or ceiling is None:
    return
  if ceiling.fell:
    print(
      "The oracle fell. The entry state is wrong, or the entering skill cannot open from "
      "it, and no bridge fixes either. Fix the table before training anything."
    )
    return
  gap = ceiling.earned - floor.earned
  print(
    f"Headroom {gap:+.3f}: what a bridge is worth on this couple, from handing over cold "
    f"to arriving exactly. A trained bridge lands between the two columns, and where it "
    f"lands is the only number about it that means anything."
  )
  if gap <= 0.05 * max(abs(ceiling.earned), 1e-6):
    print(
      "That is nearly nothing, so this hand-over was never hard and it is the wrong couple "
      "to judge a bridge by. Pick an entering skill whose opening is further from a walk."
    )


def main(cfg: HandoffCfg) -> None:
  if cfg.couple not in COUPLES:
    raise SystemExit(f"Unknown couple. Known: {', '.join(sorted(COUPLES))}.")
  couple = COUPLES[cfg.couple]

  # Before the simulator: this is the one thing a couple can be missing, and building the
  # arena first means waiting a minute to be told a posture is not written down
  selector = Selector.load(couple.entering.name)
  for line in selector.lines():
    print(f"{couple.entering.name}: {line}")

  torch.manual_seed(cfg.seed)
  env_cfg = arena(couple)
  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  try:
    shortlist = selector.shortlist(env)
    row, posture, duration = shortlist[cfg.entry]
    entry = Entry(
      state=torch.as_tensor(row, dtype=torch.float32, device=cfg.device),
      name=posture.name,
      why=posture.why,
      duration_s=couple.duration_s if couple.duration_s is not None else duration,
    )

    policies = {}
    for actor, explicit in (
      (couple.leaving, cfg.walk_checkpoint),
      (couple.entering, cfg.entering_checkpoint),
    ):
      checkpoint = find_checkpoint(load_rl_cfg(actor.task).experiment_name, explicit)
      print(f"{actor.name:12s} {checkpoint}")
      policies[actor.name] = Policy(actor.task, checkpoint, env, actor.name, cfg.device)

    outcomes = [
      stage(cfg, couple, mode, env, policies, entry) for mode in ("none", "perfect")
    ]
    report(couple, entry, outcomes)
  finally:
    env.close()


if __name__ == "__main__":
  main(tyro.cli(HandoffCfg, config=mjlab.TYRO_FLAGS))
