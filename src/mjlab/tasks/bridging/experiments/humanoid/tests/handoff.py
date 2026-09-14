"""Compare cold handoff, exact physical arrival, and arrival with the recorded action.

Run:
    uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.handoff --couple walk2kick

Requires fresh selector.record and selector.build outputs. All modes use the same entry
placement as real transitions. Perfect restores the recorded preceding action;
physical_only retains the outgoing action. Neither mode replaces the target with mocap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.bridges import (
  DEFAULT_BRIDGE,
  BridgeKind,
  resolve,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp.commands import (
  CHANNELS,
  Tolerances,
  arm_mask,
  channel_errors,
)
from mjlab.tasks.bridging.experiments.humanoid.selector import EntryTable
from mjlab.tasks.bridging.experiments.humanoid.selector import resume as entry_resume
from mjlab.tasks.bridging.experiments.humanoid.selector.table import (
  TABLE_PATH,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.table import (
  Entry as RecordedEntry,
)
from mjlab.tasks.bridging.experiments.humanoid.tests import actors
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import (
  ROBOT,
  Couple,
  Policy,
  aim,
  arena,
  defaults,
  find_checkpoint,
  fresh_obs,
  state,
)
from mjlab.tasks.registry import load_rl_cfg

COUPLES: dict[str, Couple] = {
  "walk2punch_combo": Couple(leaving=actors.WALK, entering=actors.PUNCH_COMBO),
  "walk2front_kick": Couple(leaving=actors.WALK, entering=actors.FRONT_KICK),
  "walk2jump": Couple(leaving=actors.WALK, entering=actors.JUMP, duration_s=0.7),
  "walk2kick": Couple(leaving=actors.WALK, entering=actors.KICK),
}
"""The couples worth staging this way: an entering skill whose opening is not a stand.

walk2punch_combo first, and by some distance. The combination's entry is the furthest from
anything walking produces, it needs nothing on the floor, and its tracker is anchored to a
single clip so there is no goal to get wrong on top of the pose.
"""


@dataclass
class HandoffCfg:
  couple: str = "walk2punch_combo"
  table: Path = TABLE_PATH
  entry: int = 0
  """Which row of the entering skill's entry table to aim at, in table order."""

  duration_s: float = 0.6
  """How long a bridge would get to reach the entry, in seconds.

  Only places the target here, since neither mode crosses anything: `none` hands over where
  the walk left the robot and `perfect` teleports. It decides how far ahead of the interrupt
  the target sits, which is where a real crossing would end up. The couple's own duration
  wins when it has one."""

  walk_steps: int = 150
  """Control steps of walking before the interrupt. Three seconds at 50 Hz, long enough that
  the gait has settled and the robot is genuinely mid-stride rather than still leaving its
  reset."""
  entering_steps: int = 220
  """How long the entering skill drives once it has taken over."""
  speed: float = 1.0
  """Forward command for the walk, in m/s."""

  bridge: BridgeKind = DEFAULT_BRIDGE
  """Which bridge architecture's arena this is staged in. See bridges/__init__.py.

  No bridge policy is loaded here, since neither mode crosses anything. It still decides the
  robot and the terrain the two skills run on, so the ceiling this prints is the ceiling for
  that architecture's arena."""

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
      actor.enter(env, here[:, 0:3], here[:, 3:7], 0, defaults(actor))

  twist = env.command_manager.get_term("twist")
  twist.vel_command_b[:, 0] = cfg.speed
  twist.vel_command_b[:, 1:] = 0.0
  obs = _drive(env, policies[couple.leaving.name], obs, cfg.walk_steps)

  here = state(robot)
  target = aim(
    env,
    env.command_manager.get_term("bridge"),
    couple.entering,
    entry.state,
    here,
    entry.duration_s,
    entry.frame,
    couple.entering.arrive,
    defaults(couple.entering),
    recorded=entry.recorded,
  )
  if mode in ("perfect", "physical_only"):
    previous = env.action_manager.action.clone()
    teleport(env, robot, target)
    if mode == "perfect":
      previous = torch.as_tensor(
        entry.recorded.previous_action, device=env.device, dtype=target.dtype
      ).unsqueeze(0)
    entry_resume.restore_action(env, previous)
  obs = fresh_obs(env)

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
  recorded: RecordedEntry
  frame: int
  """Step of the skill's own trajectory this state was recorded at. The oracle resumes the
  entering skill there rather than at its first frame."""


def report(couple: Couple, entry: Entry, outcomes: list[Outcome]) -> None:
  requirements = Tolerances().as_tensor(outcomes[0].errors.device)
  width = 11

  print()
  print(
    f"{couple.leaving.name} -> {couple.entering.name}, aiming at '{entry.name}', "
    f"resuming at frame {entry.frame}"
  )
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
      "The recorded-state oracle fell. Check entry reconstruction, object placement "
      "and skill robustness in this environment before attributing this failure to the bridge."
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
  table = EntryTable.load(cfg.table)
  rows = table.of(couple.entering.name)
  row = rows[min(max(cfg.entry, 0), len(rows) - 1)]
  entry_resume.require_context(row)
  for line in table.lines(couple.entering.name):
    print(line)

  torch.manual_seed(cfg.seed)
  env_cfg = arena(couple, resolve(cfg.bridge))
  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  try:
    entry = Entry(
      state=torch.as_tensor(row.state[None], dtype=torch.float32, device=cfg.device),
      name=row.name,
      why=row.why,
      duration_s=(
        couple.duration_s if couple.duration_s is not None else cfg.duration_s
      ),
      frame=row.frame,
      recorded=row,
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
      stage(cfg, couple, mode, env, policies, entry)
      for mode in ("none", "physical_only", "perfect")
    ]
    report(couple, entry, outcomes)
  finally:
    env.close()


if __name__ == "__main__":
  main(tyro.cli(HandoffCfg, config=mjlab.TYRO_FLAGS))
