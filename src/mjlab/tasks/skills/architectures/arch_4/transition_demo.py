"""Walk, bridge, jump: one robot, three policies, handed over twice.

    uv run python -m mjlab.tasks.skills.architectures.arch_4.transition_demo

One episode, on loop: the robot walks straight ahead at a fixed speed, the bridge takes
over for a fixed number of steps, the jump takes over from there, and when the clip runs
out everything resets and it happens again.

##
# What the bridge is aimed at
##

The jump is a motion tracking policy whose clip opens from a stand, so a robot arriving at
walking speed cannot start it at the beginning. What it can be handed instead is a later
frame: the bottom of the crouch, which is a state a moving body can plausibly be delivered
into. To know what that state is, the jump is rolled out from its own nominal starting
conditions and every frame recorded, and the crouch is picked out of the recording as the
lowest the root gets before the feet leave the ground. That recording is what the ghost
draws.

The bridge is then told two things: arrive at this, within this many steps. Both are what
it was trained on, a strip of poses relative to where the robot is now plus how much of
the window is left, so the instruction is assembled here in exactly the layout its own
command term produces (see bridge/mdp/commands.py). The number of steps is the tunable
one, and it belongs inside the range the bridge saw in training.

##
# Where things are in the world
##

A policy returns joint targets and nothing else, so none of the three cares where the
robot is. Two things here do.

The **ghost** has no physics. It is the recorded rollout, moved so that its first frame
sits at the pose the robot held when the switch was issued, in xy and heading only. That
move is exact rather than approximate: the jump policy never observes an absolute position
or heading, so the same states come out wherever the clip is anchored. It is also why the
rollout is recorded once at startup rather than at every switch.

The **clip** the jump will track is pinned to where the robot is meant to *arrive*, not to
where it is when the bridge starts, and it is pinned once. Anchoring again at hand-over
would slide the reference onto wherever the robot actually got to, which erases the
arrival error instead of leaving the jump to cope with it.

Neither of those has a height in it. Both robots are assumed to be on the same flat
ground.

##
# What this does not do
##

Nothing here chooses anything. The moment to aim at is the crouch because that is the one
worth aiming at with two skills in the pool, the walk runs for a fixed time before letting
go, and the window is a constant. Those three are what arch_4's chooser is supposed to
decide, and having them written down by hand is what lets the bridge be watched at all.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro
import viser
from tensordict import TensorDict

import mjlab
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.tasks.skills.architectures.arch_4.bridge import BRIDGE_TASK_ID, frames
from mjlab.tasks.skills.architectures.arch_4.bridge.dataset import G1, ROOT_STATE_DIM
from mjlab.tasks.skills.architectures.arch_4.bridge.env_cfg import bridge_env_cfg
from mjlab.tasks.skills.architectures.arch_4.bridge.mdp.commands import BridgeCommandCfg
from mjlab.tasks.skills.architectures.arch_4.bridge.view import Ghost, visual_meshes
from mjlab.tasks.skills.experiments.parkour.arena import (
  JUMP_COMMAND,
  JUMP_OBS_GROUP,
  TWIST_COMMAND,
  VELOCITY_OBS_GROUP,
  parkour_arena_env_cfg,
)
from mjlab.tasks.skills.experiments.parkour.jump import JUMP_TASK_ID
from mjlab.tasks.skills.experiments.parkour.jump.mdp.commands import (
  JumpCommand,
  JumpCommandCfg,
)
from mjlab.tasks.skills.experiments.parkour.walk import WALK_TASK_ID
from mjlab.tasks.skills.utils import retrieve_latest_checkpoint
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat

# The arena carries one observation group per policy. This is the third.
BRIDGE_OBS_GROUP = "bridge_actor"

WALK, BRIDGE, JUMP = 0, 1, 2
PHASE_NAME = {WALK: "walk", BRIDGE: "bridge", JUMP: "jump"}

ROBOT_COLOR = (70, 150, 255)
GHOST_COLOR = (245, 165, 60)


@dataclass(frozen=True)
class Config:
  walk_checkpoint: Path | None = None
  jump_checkpoint: Path | None = None
  bridge_checkpoint: Path | None = None
  """Newest under logs/rsl_rl/<experiment> when not given."""

  walk_speed: float = 1.0
  """Forward speed the walk is held at, m/s. Straight ahead, no turning."""

  jump_distance: float = 1.5
  """How far the jump is asked to travel, m. Picks the clip and how much it is
  stretched; the height it reaches comes with the clip and cannot be asked for."""

  walk_steps: int = 150
  """Steps of walking before the bridge is handed control. Three seconds at 50 Hz."""

  bridge_steps: int = 30
  """Steps the bridge is given to reach the target. The tunable one. Trained over
  10 to 50 steps, so outside that range the phase input is off distribution."""

  port: int = 8080
  device: str | None = None


class Policy:
  """A frozen checkpoint, callable on an observation.

  A checkpoint stores weights and not architecture, so something has to build a network
  of the right shape before the weights have anywhere to go. That something is the
  runner, and it sizes the network from the environment it is handed, which is why this
  needs an environment even though nothing here steps one.

  `obs_group` is which of that environment's observation groups this policy reads. Every
  task calls its own group `actor`, and the arena cannot: three policies trained in three
  different environments act in it, and no one of those observations is a superset of the
  others. So each policy is pointed at its own group and is otherwise untouched, which is
  also why checkpoints trained before the arena existed still load.
  """

  def __init__(
    self,
    task_id: str,
    env: ManagerBasedRlEnv,
    device: str,
    obs_group: str,
    checkpoint: Path | None = None,
  ) -> None:
    agent_cfg = load_rl_cfg(task_id)
    agent_cfg.obs_groups = {"actor": (obs_group,), "critic": (obs_group,)}

    path = checkpoint or Path(retrieve_latest_checkpoint(task_id))
    print(f"{task_id}: {path}")

    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(str(path), load_cfg={"actor": True}, strict=True, map_location=device)

    self.policy = runner.get_inference_policy(device=device)
    self.clip = agent_cfg.clip_actions

  @torch.no_grad()
  def __call__(self, obs: TensorDict) -> torch.Tensor:
    """The action, clipped the way this policy's own training clipped it."""
    action = self.policy(obs)
    return action if self.clip is None else action.clamp(-self.clip, self.clip)


class Goal:
  """What the bridge is told: where to arrive, and how much of the window is left.

  It lives here rather than in the environment because nothing in the environment
  computes it. In training a command term draws both halves from a corpus of recorded
  motion; here they are written from the jump's own rollout. The environment only needs
  somewhere to read them from, which is `term`.
  """

  def __init__(self, samples: int, future: int, max_steps: int) -> None:
    # Which frames of the target strip are shown. A handful spread across it rather than
    # all of them, at the same offsets training used.
    self.offsets = torch.linspace(0, future - 1, samples).long()
    self.max_steps = max_steps
    self.reach = future - 1
    """How far past the aimed-at frame the strip reaches. A rollout has to have that
    much left in it or there is no strip to cut."""
    self.value: torch.Tensor | None = None

  def term(self) -> ObservationTermCfg:
    """This, as an observation term the environment can read.

    A closure and not a bound method, which is not a style choice. The observation
    manager deep-copies the config it is given, and deep-copying a bound method copies
    the object behind it too, so the environment would end up reading a buffer that
    nothing writes. A function is copied as itself and keeps pointing here.
    """

    def read(env: ManagerBasedRlEnv) -> torch.Tensor:
      # Allocated on the first call, which is when the widths are known: the manager
      # calls every term once at startup to find out how wide it is.
      if self.value is None:
        joints = env.scene["robot"].data.joint_pos.shape[1]
        width = self.offsets.numel() * frames.pose_dim(joints) + 2
        self.value = torch.zeros(env.num_envs, width, device=env.device)
        self.offsets = self.offsets.to(env.device)
      return self.value

    return ObservationTermCfg(func=read)

  def write(
    self,
    states: torch.Tensor,
    frame: int,
    pos: torch.Tensor,
    quat: torch.Tensor,
    step: int,
    steps: int,
  ) -> None:
    """Aim at `frame` of `states`, from where the robot is now, with `steps` to get there.

    Rewritten every step, because the target is stated relative to the robot: as it moves,
    what is left to cover shrinks, and that is the number the policy acts on.
    """
    assert self.value is not None, "the environment has not been built yet"
    strip = states[frame + self.offsets].unsqueeze(0)
    poses = frames.encode(strip, pos, yaw_quat(quat)).flatten(start_dim=1)
    phase = torch.tensor(
      [[min(step / steps, 1.0), steps / self.max_steps]], device=poses.device
    )
    self.value[:] = torch.cat([poses, phase], dim=-1)


def arena_cfg(
  goal: Goal, walk_speed: float, nominal: bool = False
) -> ManagerBasedRlEnvCfg:
  """The environment all three policies act in: a bare plane and every observation.

  The corridor is left out, so this is the arena only in the sense that matters here:
  one scene carrying the walk's commanded twist and the jump's reference clip side by
  side, because one robot has to be driven by policies trained against both.

  `nominal` builds the second copy, the one the jump is rolled in from its own starting
  conditions. The only difference is that its reset teleports the robot onto the clip's
  first frame instead of leaving it where it stands.
  """
  cfg = parkour_arena_env_cfg(obstacles=None)

  # The arena already pins the twist to +x and holds that heading. All that is left is
  # the speed, and it goes in the range rather than being written every step: nothing
  # resamples, so what a reset leaves behind is what stays.
  twist = cfg.commands[TWIST_COMMAND]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges = replace(twist.ranges, lin_vel_x=(walk_speed, walk_speed))

  motion = cfg.commands[JUMP_COMMAND]
  assert isinstance(motion, JumpCommandCfg)
  motion.reset_robot_to_clip = nominal

  # The bridge's observation, taken from the bridge's own task so the layout cannot drift
  # away from what the checkpoint was fitted to. Its one term that reads a command is
  # swapped for the buffer above; everything else is proprioception and is already right.
  actor = bridge_env_cfg(play=True).observations["actor"]
  terms = dict(actor.terms)
  terms["command"] = goal.term()
  cfg.observations[BRIDGE_OBS_GROUP] = replace(actor, terms=terms)
  return cfg


def motion_of(env: RslRlVecEnvWrapper) -> JumpCommand:
  """The clip the jump tracks, and where it is pinned."""
  term = env.unwrapped.command_manager.get_term(JUMP_COMMAND)
  assert isinstance(term, JumpCommand)
  return term


def state_of(env: RslRlVecEnvWrapper) -> torch.Tensor:
  """The robot right now, in the corpus layout: root pose, root velocity, joints.

  (13 + 2J,), which is what `frames.encode` slices and what a recorded rollout is made
  of, so one layout serves the ghost, the target and the drawing.
  """
  d = env.unwrapped.scene["robot"].data
  return torch.cat(
    [
      d.root_link_pos_w[0],
      d.root_link_quat_w[0],
      d.root_link_lin_vel_w[0],
      d.root_link_ang_vel_w[0],
      d.joint_pos[0],
      d.joint_vel[0],
    ]
  )


def root_pose(env: RslRlVecEnvWrapper) -> tuple[torch.Tensor, torch.Tensor]:
  """Where the robot is and which way it faces, as (1, 3) and (1, 4)."""
  d = env.unwrapped.scene["robot"].data
  return d.root_link_pos_w[0:1], d.root_link_quat_w[0:1]


def place(
  states: torch.Tensor, at_pos: torch.Tensor, at_quat: torch.Tensor
) -> torch.Tensor:
  """The rollout, moved so its first frame sits at a given pose. (T, 13 + 2J).

  A rigid motion in xy and heading, and nothing else. Height is untouched, which is what
  assumes both ends are on the same flat ground.
  """
  turn = quat_mul(yaw_quat(at_quat), quat_inv(yaw_quat(states[0:1, 3:7])))
  turn = turn.expand(states.shape[0], 4)

  offset = states[:, 0:3] - states[0, 0:3]
  offset[:, 2] = 0.0
  turned = quat_apply(turn, offset)

  moved = states.clone()
  moved[:, 0:2] = at_pos[0, 0:2] + turned[:, 0:2]
  moved[:, 3:7] = quat_mul(turn, states[:, 3:7])
  moved[:, 7:10] = quat_apply(turn, states[:, 7:10])
  moved[:, 10:ROOT_STATE_DIM] = quat_apply(turn, states[:, 10:ROOT_STATE_DIM])
  return moved


def draw(ghost: Ghost, g1: G1, state: torch.Tensor) -> None:
  """Put one ghost at one state.

  The ghost is a second copy of the robot, a plain MuJoCo model that is never stepped.
  A root and a set of joint angles go in, forward kinematics turns them into a position
  per geom, and the meshes are moved there.
  """
  q = np.asarray(state.cpu(), dtype=np.float64)
  g1.data.qpos[0:7] = q[0:7]
  g1.data.qpos[7:] = q[ROOT_STATE_DIM : ROOT_STATE_DIM + g1.num_joints]
  mujoco.mj_kinematics(g1.model, g1.data)
  ghost.pose(g1.data)


def aim(env: RslRlVecEnvWrapper, distance: float) -> tuple[int, float]:
  """Pin which jump this is: the same clip, stretched the same way, on every reset.

  Distance is the whole command. The stretch scales the clip's horizontal travel and
  nothing else, so the height the jump reaches is whatever the chosen clip reaches.
  """
  term = motion_of(env)
  term.request_goal(distance)
  return term.solve_goal(distance)


def record(env: RslRlVecEnvWrapper, policy: Policy, steps: int) -> torch.Tensor:
  """Run the jump from its own starting conditions and keep every frame. (T, 13 + 2J).

  Done once. The jump policy reads nothing in world coordinates, so this rollout is the
  same wherever it happens, and a switch only decides where to put it.
  """
  env.reset()
  states = [state_of(env)]
  for _ in range(steps - 1):
    _, _, dones, _ = env.step(policy(env.get_observations()))
    states.append(state_of(env))
    # A fall ends the rollout rather than recording the reset that follows it. Whether
    # what is left reaches far enough is checked by the caller.
    if bool(dones[0]):
      break
  return torch.stack(states)


def crouch_frame(states: torch.Tensor, takeoff: int) -> int:
  """The bottom of the crouch: the lowest the root gets before the feet leave the ground.

  Measured off the rollout rather than written down, so a different clip does not mean
  finding its crouch by hand. selector.py computes the same thing and says why it is the
  moment worth aiming at.
  """
  return int(states[:takeoff, 2].argmin())


def main(cfg: Config) -> None:
  # Registering a task is a side effect of importing the package that defines it, so
  # nothing is in the registry until the tasks are imported.
  import mjlab.tasks  # noqa: F401

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Widths and offsets the bridge was trained with, read off its own config rather than
  # written down again here.
  bridge_cfg = bridge_env_cfg(play=True).commands["bridge"]
  assert isinstance(bridge_cfg, BridgeCommandCfg)
  goal = Goal(
    samples=bridge_cfg.target_samples,
    future=bridge_cfg.windows.future,
    max_steps=bridge_cfg.windows.gap_range[1],
  )

  # Two copies of the same world: one the robot really lives in, one the jump is rolled
  # in from its own start. The second is a whole environment rather than a replay because
  # a rollout is what the policy does under physics, and the ghost is that rollout. Both
  # read the same instruction buffer, and only one of them ever runs the bridge.
  world = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=arena_cfg(goal, cfg.walk_speed), device=device),
    clip_actions=None,
  )
  nominal = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=arena_cfg(goal, cfg.walk_speed, nominal=True), device=device),
    clip_actions=None,
  )

  walk = Policy(
    WALK_TASK_ID, world.unwrapped, device, VELOCITY_OBS_GROUP, cfg.walk_checkpoint
  )
  jump = Policy(
    JUMP_TASK_ID, world.unwrapped, device, JUMP_OBS_GROUP, cfg.jump_checkpoint
  )
  bridge = Policy(
    BRIDGE_TASK_ID, world.unwrapped, device, BRIDGE_OBS_GROUP, cfg.bridge_checkpoint
  )
  act = {WALK: walk, BRIDGE: bridge, JUMP: jump}

  # Both worlds jump the same jump, so both are pinned to the same clip.
  clip, scale = aim(world, cfg.jump_distance)
  aim(nominal, cfg.jump_distance)
  meta = motion_of(world).motion.metadata[clip]
  print(
    f"jump: {meta.name} stretched {scale:.2f}x, "
    f"{scale * meta.distance:.2f} m out, apex {meta.goal_apex:.2f} m"
  )

  rollout = record(nominal, jump, meta.num_frames)
  crouch = crouch_frame(rollout, meta.takeoff_step)
  print(
    f"crouch at frame {crouch} of {rollout.shape[0]}, takeoff at {meta.takeoff_step}"
  )
  assert rollout.shape[0] > crouch + goal.reach, (
    "The rollout stops before the target strip does. The jump policy fell over inside "
    "its own environment, which is a broken jump checkpoint rather than a bridging "
    "problem."
  )

  g1 = G1()
  meshes = visual_meshes(g1.model)
  server = viser.ViserServer(port=cfg.port, label="walk, bridge, jump")
  server.scene.add_grid("/ground", width=40.0, height=40.0, cell_size=0.5)
  solid = Ghost(server, meshes, "robot", ROBOT_COLOR, opacity=1.0)
  ghost = Ghost(server, meshes, "jump", GHOST_COLOR, visible=False, opacity=0.4)

  play = server.gui.add_checkbox("Play", initial_value=True)
  rate = server.gui.add_slider("Speed", min=0.1, max=1.5, step=0.1, initial_value=0.5)
  print(f"\nViewer at http://localhost:{cfg.port} -- Ctrl-C to quit.")

  # A requested goal is applied at the next reset, and the environment's own first reset
  # already happened while it was being built. Without this the robot would be tracking
  # whichever clip came up then, while the ghost and the target came from the one asked
  # for. The rollout got its reset inside `record`.
  world.reset()

  step_dt = world.unwrapped.step_dt
  env_ids = torch.zeros(1, dtype=torch.long, device=device)
  target = rollout
  phase, tick = WALK, 0
  last = time.time()

  try:
    while True:
      now = time.time()
      if not play.value or now - last < step_dt / max(rate.value, 0.1):
        time.sleep(1.0 / 240.0)
        continue
      last = now

      # Hand-overs happen before anything acts, so the step below is already step zero of
      # the phase it moves into. `tick` counts steps taken in the current phase.
      if phase == WALK and tick >= cfg.walk_steps:
        # The ghost is spawned where the switch is issued: the rollout is put down at the
        # pose the robot holds right now, and everything downstream reads it from there.
        target = place(rollout, *root_pose(world))
        phase, tick = BRIDGE, 0
      elif phase == BRIDGE and tick >= cfg.bridge_steps:
        # The clip is pinned to where the robot was asked to arrive, not to where it got
        # to, and wound to the frame the bridge was aiming at. Whatever distance is left
        # between the two is the jump's problem now, which is the point of leaving it.
        motion_of(world).anchor_to_robot(
          env_ids,
          start_frame=crouch,
          at_pos=target[crouch : crouch + 1, 0:3],
          at_quat=target[crouch : crouch + 1, 3:7],
        )
        phase, tick = JUMP, 0
      elif phase == JUMP and bool(motion_of(world).motion_done[0]):
        world.reset()
        ghost.visible = False
        phase, tick = WALK, 0
      if tick == 0:
        print(f"[{PHASE_NAME[phase]}]")

      if phase == BRIDGE:
        # Written before the observation is computed, since this is part of it.
        goal.write(target, crouch, *root_pose(world), tick, cfg.bridge_steps)

      _, _, dones, _ = world.step(act[phase](world.get_observations()))

      with server.atomic():
        draw(solid, g1, state_of(world))
        if phase != WALK:
          # The ghost walks into the crouch and waits there while the bridge works, then
          # carries on through the jump alongside the robot, where it is the reference
          # the robot is tracking and the gap between the two is the tracking error.
          index = min(tick, crouch) if phase == BRIDGE else crouch + tick
          draw(ghost, g1, target[min(index, target.shape[0] - 1)])
          ghost.visible = True
      tick += 1

      # An episode the robot ended by itself, by falling or by timing out. The
      # environment has already put it back; only the sequence has to be told.
      if bool(dones[0]):
        ghost.visible = False
        phase, tick = WALK, 0
  except KeyboardInterrupt:
    print("\nShutting down.")
  finally:
    world.unwrapped.close()
    nominal.unwrapped.close()


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
