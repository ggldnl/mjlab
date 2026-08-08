"""Stitch a walk into a jump with the bridge, and watch the seam.

    uv run python -m mjlab.tasks.skills.architectures.arch_4.transition_demo_2

evaluate.py asks whether the bridge can cross a hole cut out of a recording, where the
motion on both sides is a real body and the answer is known. This asks the same question
of a hole cut between two policies, which is the question a composition actually has.
Everything else is deliberately the same, down to the colours: the seam is the only thing
that changed.

The window is made up rather than drawn from the corpus:

    before   the walk policy's own rollout, up to a frame you pick
    hole     nothing, which is the point
    after    the jump policy's own rollout, from its crouch onward

and it is handed to the bridge through the same buffers its command term fills during
training, so the observation the policy reads here is computed by the same code that
computed it then. Nothing about the instruction is assembled by hand.

The **solid** robot is the composition: the walk's last frames are replayed onto it, the
bridge takes over and drives through the hole under physics, and the jump's rollout picks
up on the far side. That last handover is the one that matters, and the distance between
where the robot got to and where the jump needed it is printed beside it.

The **ghost** is the reference. Green where it is motion someone actually produced, red
across the hole, where it is only the straight line between the two ends and the bridge is
under no obligation to follow it. Only the hole is simulated; the two rollouts are replayed,
because they came from policies that already ran under physics in their own environments.

##
# Where the jump gets put
##

The two rollouts happened in different worlds and neither knows about the other. The walk
is rebased so its hand-off frame sits at the environment origin, exactly as a corpus window
is rebased. The jump is then placed so that its crouch lands where the walk would have got
to had it simply kept walking for the length of the hole, facing the way it was facing.
That distance is measured off the walk rollout itself rather than computed from the
commanded speed, and `lead_scale` scales it: 1.0 asks the bridge to carry the body forward,
0.0 asks it to stop where it stands.

None of this is a claim that the placement is right. It is the one free choice in the whole
stitch, it is a slider, and it is the thing to play with first when the seam looks wrong.
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

import mjlab
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.skills.architectures.arch_4.bridge import BRIDGE_TASK_ID
from mjlab.tasks.skills.architectures.arch_4.bridge.dataset import G1, ROOT_STATE_DIM
from mjlab.tasks.skills.architectures.arch_4.bridge.env_cfg import bridge_env_cfg
from mjlab.tasks.skills.architectures.arch_4.bridge.mdp.commands import BridgeCommand
from mjlab.tasks.skills.architectures.arch_4.bridge.view import (
  CONTEXT_COLOR,
  MASKED_COLOR,
  MODEL_COLOR,
  Ghost,
  visual_meshes,
)
from mjlab.tasks.skills.experiments.parkour.jump import JUMP_TASK_ID
from mjlab.tasks.skills.experiments.parkour.jump.mdp.commands import JumpCommand
from mjlab.tasks.skills.experiments.parkour.walk import WALK_TASK_ID
from mjlab.tasks.skills.utils import retrieve_latest_checkpoint
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import (
  normalize,
  quat_apply,
  quat_inv,
  quat_mul,
  yaw_quat,
)

# The three stretches of one loop, as in evaluate.py.
BEFORE, BRIDGING, AFTER = 0, 1, 2


@dataclass(frozen=True)
class Config:
  walk_checkpoint: Path | None = None
  jump_checkpoint: Path | None = None
  bridge_checkpoint: Path | None = None
  """Newest under logs/rsl_rl/<experiment> when not given. Resolved paths are printed."""

  walk_speed: float = 1.0
  """Forward speed the walk rollout is recorded at, m/s."""

  jump_distance: float = 1.5
  """How far the jump is asked to travel, m. Picks the clip and its stretch."""

  walk_steps: int = 200
  """Frames of walking to record. The hand-off is picked from inside this."""

  hole: int = 30
  """Length of the hole, in frames, before the slider moves it."""

  lead_scale: float = 1.0
  """How much of the walk's own progress over the hole the jump is placed ahead by."""

  port: int = 8080
  device: str | None = None


def walk_env_cfg(speed: float) -> ManagerBasedRlEnvCfg:
  """The walk task's environment with the twist held at one forward speed.

  Every part of the command is pinned, the yaw rate included. The rollout runs free once
  it starts, so anything left with a range is sampled once at the reset and held for the
  whole recording, and a yaw rate sampled at random is a walk that quietly turns.
  """
  cfg = load_env_cfg(WALK_TASK_ID, play=True)
  cfg.scene.num_envs = 1
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  # Heading control would rewrite the yaw rate every step to hold a sampled heading,
  # which is a second thing writing the command; the term refuses to carry a heading
  # range without it.
  twist.heading_command = False
  twist.ranges = replace(
    twist.ranges,
    lin_vel_x=(speed, speed),
    lin_vel_y=(0.0, 0.0),
    ang_vel_z=(0.0, 0.0),
    heading=None,
  )
  twist.resampling_time_range = (1.0e9, 1.0e9)
  twist.rel_standing_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.gui = False
  twist.debug_vis = False
  return cfg


def jump_env_cfg() -> ManagerBasedRlEnvCfg:
  """The jump task's own environment, where a reset puts the robot on the clip's first
  frame. That reset is what makes the recorded rollout the nominal jump."""
  cfg = load_env_cfg(JUMP_TASK_ID, play=True)
  cfg.scene.num_envs = 1
  motion = cfg.commands["motion"]
  motion.gui = False
  motion.debug_vis = False
  return cfg


def load_policy(
  task_id: str, env: ManagerBasedRlEnv, device: str, checkpoint: Path | None
) -> tuple[object, RslRlVecEnvWrapper]:
  """The trained actor, loaded the way every other frozen policy here is loaded.

  evaluate.py's `load_policy` with the task id as an argument, because this one has three
  to load. A checkpoint stores weights and not architecture, so the runner has to build a
  network first, and it sizes that network from the environment it is handed: each policy
  is therefore loaded against the environment its own task registers.
  """
  agent_cfg = load_rl_cfg(task_id)
  path = checkpoint or Path(retrieve_latest_checkpoint(task_id))
  print(f"{task_id}: {path}")

  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(str(path), load_cfg={"actor": True}, strict=True, map_location=device)
  return runner.get_inference_policy(device=device), wrapped


def state_of(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The robot right now, in the corpus layout: (13 + 2J,).

  Root position, orientation, linear and angular velocity, joint angles, joint speeds.
  The same layout a corpus frame has, so a recorded rollout can be dropped straight into
  the buffers the bridge reads.
  """
  d = env.scene["robot"].data
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


def record(wrapped: RslRlVecEnvWrapper, policy, steps: int) -> torch.Tensor:
  """Run a policy in its own environment and keep every frame. (T, 13 + 2J)."""
  env = wrapped.unwrapped
  wrapped.reset()
  states = [state_of(env)]
  for _ in range(steps - 1):
    with torch.inference_mode():
      action = policy(wrapped.get_observations())
    _, _, dones, _ = wrapped.step(action)
    states.append(state_of(env))
    if bool(dones[0]):
      break
  return torch.stack(states)


def place(
  states: torch.Tensor, frame: int, at_pos: torch.Tensor, at_quat: torch.Tensor
) -> torch.Tensor:
  """The rollout, moved so that `frame` of it sits at a given pose. (T, 13 + 2J).

  A rigid motion in xy and heading and nothing else, so heights are left exactly as the
  rollout produced them and both ends are assumed to be on the same flat ground. The
  rotation is the target heading relative to that frame's own, since the pelvis is not
  aligned with the direction the motion travels.
  """
  turn = quat_mul(yaw_quat(at_quat), quat_inv(yaw_quat(states[frame : frame + 1, 3:7])))
  turn = turn.expand(states.shape[0], 4)

  offset = states[:, 0:3] - states[frame, 0:3]
  offset[:, 2] = 0.0
  turned = quat_apply(turn, offset)

  moved = states.clone()
  moved[:, 0:2] = at_pos[0, 0:2] + turned[:, 0:2]
  moved[:, 3:7] = quat_mul(turn, states[:, 3:7])
  moved[:, 7:10] = quat_apply(turn, states[:, 7:10])
  moved[:, 10:ROOT_STATE_DIM] = quat_apply(turn, states[:, 10:ROOT_STATE_DIM])
  return moved


def crouch_frame(states: torch.Tensor, takeoff: int) -> int:
  """The bottom of the crouch: the lowest the root gets before the feet leave the ground.

  Measured off the rollout rather than written down, so a different clip does not mean
  finding its crouch by hand. selector.py computes the same thing and says why it is the
  moment worth aiming at: a body arriving with speed can plant and sink into it, instead
  of stopping, standing up, settling, and only then crouching.
  """
  return int(states[:takeoff, 2].argmin())


def tween(start: torch.Tensor, end: torch.Tensor, steps: int) -> torch.Tensor:
  """The straight line between two states, one frame per step. (steps, 13 + 2J).

  This is what the red ghost draws and all the policy is *not* given. It exists because
  the environment's reward, its metrics and its lost-tracking termination all read a
  reference every step and there is no real one inside a hole at inference. Straight
  interpolation is the honest stand-in: it is also the baseline every in-betweener has to
  beat (see frames.blend).
  """
  alpha = torch.linspace(0.0, 1.0, steps + 2, device=start.device)[1:-1].unsqueeze(-1)
  path = start.unsqueeze(0) * (1.0 - alpha) + end.unsqueeze(0) * alpha
  path[:, 3:7] = normalize(path[:, 3:7])
  return path


@dataclass
class Stitch:
  """One made-up window: a walk in, a hole, a jump out."""

  before: torch.Tensor
  """The walk's last frames up to and including the hand-off. (past, 13 + 2J)."""

  hole: torch.Tensor
  """The straight line across the hole. (gap, 13 + 2J)."""

  after: torch.Tensor
  """The jump from its crouch onward. (T, 13 + 2J)."""

  gap: int


def stitch(
  walk: torch.Tensor,
  hand_off: int,
  jump: torch.Tensor,
  crouch: int,
  gap: int,
  past: int,
  lead_scale: float,
  origin: torch.Tensor,
) -> Stitch:
  """Cut a window out of two rollouts that never met.

  The walk is rebased so its hand-off sits at the environment origin, which is a pure
  translation because the robot is put down at the walk's own heading and there is no
  rotation to undo. The jump is then placed so its crouch lands where the walk itself got
  to `gap` frames later, which is the closest thing to what a corpus window would have
  had there.
  """
  walk = walk.clone()
  walk[:, 0:2] += origin[0:2] - walk[hand_off, 0:2]

  ahead = walk[min(hand_off + gap, walk.shape[0] - 1), 0:2] - walk[hand_off, 0:2]
  at_pos = walk[hand_off : hand_off + 1, 0:3].clone()
  at_pos[0, 0:2] += lead_scale * ahead
  jump = place(jump, crouch, at_pos, walk[hand_off : hand_off + 1, 3:7])

  after = jump[crouch:]
  return Stitch(
    before=walk[hand_off - past + 1 : hand_off + 1],
    hole=tween(walk[hand_off], after[0], gap),
    after=after,
    gap=gap,
  )


def install(command: BridgeCommand, window: Stitch) -> None:
  """Put a made-up window into the buffers the command term would have filled.

  Everything the policy reads is derived from these three: `reference` from the hand-off
  onward, how long the hole is, and how far into it we are. `target_in_base` gathers rows
  `gap + target_index` out of the reference and expresses them relative to wherever the
  robot is at that moment, so filling the rows is the whole of the instruction.
  """
  span = command.reference.shape[1]
  rows = torch.cat([window.hole, window.after])[:span]
  command.reference[0, : rows.shape[0]] = rows
  # Past the end of what was stitched, hold the last frame, which is what padding a short
  # clip does everywhere else in this architecture.
  command.reference[0, rows.shape[0] :] = rows[-1]
  command.gap[0] = window.gap
  command.step[0] = 0


def teleport(command: BridgeCommand, state: torch.Tensor) -> None:
  """Put the robot on the hand-off, momentum included.

  The velocities are the point. The whole architecture exists to exploit the momentum the
  outgoing skill leaves behind, and a hand-over that dropped it would be posing the robot
  rather than handing it over. Same sequence the command term's own reset uses.
  """
  robot = command.robot
  env_ids = torch.zeros(1, dtype=torch.long, device=state.device)
  joints = ROOT_STATE_DIM + command.num_joints
  limits = robot.data.soft_joint_pos_limits[env_ids]
  joint_pos = state[None, ROOT_STATE_DIM:joints].clamp(limits[:, :, 0], limits[:, :, 1])
  robot.write_joint_state_to_sim(joint_pos, state[None, joints:], env_ids=env_ids)
  robot.write_root_state_to_sim(state[None, 0:ROOT_STATE_DIM], env_ids=env_ids)
  robot.reset(env_ids=env_ids)


class StitchViewer:
  """One seam at a time: walk in, bridge across, jump out, reference alongside.

  evaluate.py's BridgeViewer with the window coming from two policies instead of the
  corpus. The phases, the colours and the loop are the same.
  """

  def __init__(
    self,
    server: viser.ViserServer,
    env: ManagerBasedRlEnv,
    policy,
    wrapped: RslRlVecEnvWrapper,
    g1: G1,
    walk: torch.Tensor,
    jump: torch.Tensor,
    crouch: int,
    cfg: Config,
  ) -> None:
    self.server = server
    self.env = env
    self.policy = policy
    self.wrapped = wrapped
    self.g1 = g1
    self.walk = walk
    self.jump = jump
    self.crouch = crouch
    self.cfg = cfg

    command = env.command_manager.get_term("bridge")
    assert isinstance(command, BridgeCommand)
    self.command = command
    self.num_joints = command.num_joints

    meshes = visual_meshes(g1.model)
    self.robot = Ghost(server, meshes, "robot", MODEL_COLOR, opacity=1.0)
    self.reference = Ghost(server, meshes, "reference", CONTEXT_COLOR, opacity=0.45)
    self.reference_hole = Ghost(
      server, meshes, "reference_hole", MASKED_COLOR, visible=False, opacity=0.45
    )
    server.scene.add_grid("/ground", width=20.0, height=20.0, cell_size=0.5)

    # The hand-off needs a full context behind it and the widest hole's worth of walking
    # still in front of it, since that is what the jump is placed against. Read off the
    # recording rather than off the config, because a walk that fell is shorter than asked.
    self.last_hand_off = walk.shape[0] - 1 - command.cfg.windows.gap_range[1]
    assert self.last_hand_off >= command.past, (
      f"The walk rollout is {walk.shape[0]} frames, too short to cut a hand-off out of. "
      "The walk policy fell over in its own environment."
    )

    self.phase = BEFORE
    self.cursor = 0
    self.gap = cfg.hole
    self.arrival_error = (0.0, 0.0)
    self._build_gui()
    self.load()

  def _build_gui(self) -> None:
    gui = self.server.gui
    with gui.add_folder("Seam"):
      self.sl_hand_off = gui.add_slider(
        "Hand-off frame",
        min=self.command.past - 1,
        max=self.last_hand_off,
        step=1,
        initial_value=min(self.cfg.walk_steps - 1, self.last_hand_off),
        hint="Which frame of the walk the bridge takes over from, so which phase of "
        "the stride it inherits.",
      )
      self.sl_hole = gui.add_slider(
        "Hole [frames]",
        min=self.command.cfg.windows.gap_range[0],
        max=self.command.cfg.windows.gap_range[1],
        step=1,
        initial_value=self.cfg.hole,
        hint="How long the bridge is given. Outside the trained range the phase input "
        "is off distribution.",
      )
      self.sl_lead = gui.add_slider(
        "Lead",
        min=0.0,
        max=1.5,
        step=0.05,
        initial_value=self.cfg.lead_scale,
        hint="How far ahead the jump is placed, as a share of what the walk itself "
        "covers over the hole. 0 puts the crouch where the robot stands.",
      )
      self.btn_again = gui.add_button("Replay")
      self.html = gui.add_html("")

      @self.sl_hand_off.on_update
      def _(_) -> None:
        self.load()

      @self.sl_hole.on_update
      def _(_) -> None:
        self.load()

      @self.sl_lead.on_update
      def _(_) -> None:
        self.load()

      @self.btn_again.on_click
      def _(_) -> None:
        self.load()

    with gui.add_folder("Playback"):
      self.cb_play = gui.add_checkbox("Play", initial_value=True)
      self.sl_speed = gui.add_slider(
        "Speed", min=0.1, max=1.5, step=0.1, initial_value=0.5
      )
      self.cb_resume = gui.add_checkbox(
        "Resume jump after bridge",
        initial_value=True,
        hint="On: the jump's rollout is written back, so arrival error shows as a "
        "jump. Off: the bridge keeps driving.",
      )
      self.cb_reference = gui.add_checkbox("Show reference ghost", initial_value=True)

  def load(self) -> None:
    """Cut a fresh window, put the robot on its hand-off, rewind to the start."""
    self.gap = int(self.sl_hole.value)
    hand_off = int(self.sl_hand_off.value)
    window = stitch(
      self.walk,
      hand_off,
      self.jump,
      self.crouch,
      self.gap,
      self.command.past,
      float(self.sl_lead.value),
      self.env.scene.env_origins[0],
    )

    # The reset draws a corpus window and puts the robot on it; both are then replaced.
    # Going through reset rather than around it is what clears the episode buffers, the
    # last action and the termination flags, which is the state training hands over in.
    self.env.reset()
    install(self.command, window)
    teleport(self.command, window.before[-1])

    self.before = window.before.cpu().numpy()
    # What the ghost draws from the hand-off onward. The same rows the environment holds,
    # except that they are not cut off at the width of its buffer: the policy only ever
    # looks a hole and a context ahead, and we want to watch the jump to the end.
    self.after = torch.cat([window.hole, window.after]).cpu().numpy()
    self.tail = self.command.tail
    self.phase = BEFORE
    self.cursor = 0
    self.arrival_error = (0.0, 0.0)
    self._draw()

  def _qpos_of(self, state: np.ndarray) -> np.ndarray:
    """The rendering pose of a state: root pose then joint angles."""
    return np.concatenate(
      [state[0:3], state[3:7], state[ROOT_STATE_DIM : ROOT_STATE_DIM + self.num_joints]]
    )

  def _robot_qpos(self) -> np.ndarray:
    """The simulated robot's own pose, read back out of the environment."""
    data = self.command.robot.data
    return np.concatenate(
      [
        data.root_link_pos_w[0].cpu().numpy(),
        data.root_link_quat_w[0].cpu().numpy(),
        data.joint_pos[0].cpu().numpy(),
      ]
    )

  def _pose(self, ghost: Ghost, qpos: np.ndarray) -> None:
    data = self.g1.data
    data.qpos[0:3] = qpos[0:3]
    data.qpos[3:7] = qpos[3:7]
    data.qpos[7:] = qpos[7:]
    mujoco.mj_kinematics(self.g1.model, data)
    ghost.pose(data)

  def advance(self) -> None:
    """One frame of the loop, stepping physics only where the bridge is responsible."""
    if self.phase == BEFORE:
      self.cursor += 1
      if self.cursor >= self.before.shape[0]:
        self.phase, self.cursor = BRIDGING, 0
      return

    if self.phase == BRIDGING:
      obs = self.wrapped.get_observations()
      with torch.inference_mode():
        action = self.policy(obs)
      self.env.step(action)
      self.cursor = int(self.command.step[0])
      if self.cursor >= self.gap:
        # The hand-back. How far the robot is from the crouch the jump needs to start in
        # is the number this whole architecture is judged on.
        target = self.after[min(self.gap, self.after.shape[0] - 1)]
        actual = self._robot_qpos()
        self.arrival_error = (
          float(np.linalg.norm(actual[0:3] - target[0:3])),
          float(np.abs(actual[7:] - self._qpos_of(target)[7:]).mean()),
        )
        self.phase = AFTER
      return

    # After the hole: either the jump's rollout is written back, or the bridge carries on.
    if self.cb_resume.value:
      self.cursor += 1
      # Played to the end of the jump rather than to the end of the training tail. The
      # tail is how long the policy was scored for after a hole; the jump is what we came
      # to watch.
      over = self.cursor >= self.after.shape[0] - 1
    else:
      obs = self.wrapped.get_observations()
      with torch.inference_mode():
        action = self.policy(obs)
      self.env.step(action)
      self.cursor = int(self.command.step[0])
      over = self.cursor >= self.gap + self.tail
    if over:
      self.load()

  def _draw(self) -> None:
    reference_index = min(
      self.cursor if self.phase != BEFORE else 0, self.after.shape[0] - 1
    )
    if self.phase == BEFORE:
      solid = self._qpos_of(self.before[self.cursor])
      ghost = solid
    elif self.phase == BRIDGING:
      solid = self._robot_qpos()
      ghost = self._qpos_of(self.after[reference_index])
    else:
      ghost = self._qpos_of(self.after[reference_index])
      solid = ghost if self.cb_resume.value else self._robot_qpos()

    in_hole = self.phase == BRIDGING
    with self.server.atomic():
      self._pose(self.robot, solid)
      shown = self.reference_hole if in_hole else self.reference
      if self.cb_reference.value:
        self._pose(shown, ghost)
      self.reference.visible = self.cb_reference.value and not in_hole
      self.reference_hole.visible = self.cb_reference.value and in_hole
    self._update_info()

  def _update_info(self) -> None:
    name = ("walk", "bridge", "jump")[self.phase]
    colour = ("#3cc85a", "#e13a2d", "#3cc85a")[self.phase]
    distance, joints = self.arrival_error
    self.html.content = (
      '<div style="font-size:0.85em;line-height:1.4;padding:0 1em 0.5em 1em;">'
      f"<b>Hand-off:</b> walk frame {int(self.sl_hand_off.value)}"
      f"<br/><b>Entry:</b> jump frame {self.crouch} (crouch)"
      f"<br/><b>Hole:</b> {self.gap} frames ({self.gap / 50.0:.2f} s)"
      f"<br/><b>Now:</b> <span style='color:{colour}'>{name}</span> "
      f"({self.cursor + 1})"
      f"<br/><b>At hand-back:</b> {distance:.3f} m, {joints:.3f} rad"
      "</div>"
    )

  def step(self) -> None:
    if self.cb_play.value:
      self.advance()
    self._draw()


def rollouts(cfg: Config, device: str) -> tuple[torch.Tensor, torch.Tensor, int]:
  """Record both skills in their own environments, then close them.

  Neither environment is needed after this. What the demo works on from here is two
  arrays of states, which is the same thing the corpus hands evaluate.py.
  """
  walk_env = ManagerBasedRlEnv(cfg=walk_env_cfg(cfg.walk_speed), device=device)
  policy, wrapped = load_policy(WALK_TASK_ID, walk_env, device, cfg.walk_checkpoint)
  # Long enough that the hand-off slider always has a hole's worth of walking left in
  # front of it, since that is what the jump is placed against.
  walk = record(wrapped, policy, cfg.walk_steps + 64)
  walk_env.close()

  jump_env = ManagerBasedRlEnv(cfg=jump_env_cfg(), device=device)
  policy, wrapped = load_policy(JUMP_TASK_ID, jump_env, device, cfg.jump_checkpoint)
  motion = jump_env.command_manager.get_term("motion")
  assert isinstance(motion, JumpCommand)
  # Pin the clip and its stretch before recording, so this is one jump and not whichever
  # came up. Applied at the next reset, which `record` does first thing.
  motion.request_goal(cfg.jump_distance)
  clip, scale = motion.solve_goal(cfg.jump_distance)
  meta = motion.motion.metadata[clip]
  print(
    f"jump: {meta.name} stretched {scale:.2f}x, "
    f"{scale * meta.distance:.2f} m out, apex {meta.goal_apex:.2f} m"
  )
  jump = record(wrapped, policy, meta.num_frames)
  jump_env.close()

  crouch = crouch_frame(jump, meta.takeoff_step)
  print(f"walk: {walk.shape[0]} frames, jump: {jump.shape[0]} frames")
  print(f"crouch at jump frame {crouch}, takeoff at {meta.takeoff_step}")
  return walk, jump, crouch


def main(cfg: Config) -> None:
  # Registering a task is a side effect of importing the package that defines it.
  import mjlab.tasks  # noqa: F401

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  walk, jump, crouch = rollouts(cfg, device)

  # The stage: the bridge's own environment, which is where its command term lives and
  # therefore the only place its observation is computed by the code that trained it.
  env_cfg = bridge_env_cfg(play=True)
  env_cfg.scene.num_envs = 1
  # There is no reference inside a hole at inference, so a termination that measures the
  # robot against one is measuring it against a straight line nobody promised to follow.
  env_cfg.terminations.pop("lost_tracking", None)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  policy, wrapped = load_policy(BRIDGE_TASK_ID, env, device, cfg.bridge_checkpoint)

  server = viser.ViserServer(port=cfg.port, label="Walk to jump")
  viewer = StitchViewer(server, env, policy, wrapped, G1(), walk, jump, crouch, cfg)

  print(f"\nViewer at http://localhost:{cfg.port} -- Ctrl-C to quit.")
  last = time.time()
  try:
    while True:
      now = time.time()
      if now - last >= (1.0 / 50.0) / max(viewer.sl_speed.value, 0.1):
        viewer.step()
        last = now
      time.sleep(1.0 / 240.0)
  except KeyboardInterrupt:
    print("\nShutting down.")
  finally:
    env.close()


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
