"""Watch a trained bridge cross a hole, against the body that actually crossed it.

    uv run python -m mjlab.tasks.skills.architectures.arch_4.bridge.evaluate
    uv run python -m mjlab.tasks.skills.architectures.arch_4.bridge.evaluate --split eval

Two robots, one window, played on loop.

The **solid** one is the composition, end to end: the clip runs into the hand-off, the
bridge takes over and drives through the hole under physics, and then the clip picks up
again on the far side. That last handover is the one that matters. If the bridge has left
the robot somewhere the next thing cannot start from, the resume is a visible jump, and
the size of that jump is printed beside it.

The **ghost** is the reference: what the subject did, every frame, including inside the
hole. Green while the clip is driving, red across the hole, which is where the two are
allowed to differ and where the whole question lives.

Only the hole is simulated. The context stretches on either side stand in for skills this
component does not have yet, so they are replayed rather than controlled, and physics is
switched on exactly where the bridge is responsible. That boundary is the point: the
supervised in-betweener this replaced looked fine frame by frame and put a foot through
the floor in two windows out of five, and no amount of playback would have made that
obvious. Here the robot is standing on the floor because the floor is pushing back.

`Resume clip after bridge` decides what happens on the far side. On, the clip is written
back onto the robot, which is the composition as it will really be assembled and makes the
arrival error visible as a jump. Off, the policy keeps driving through the second window
the way training scores it, which shows whether it could have carried on by itself.

The number to read is `hand-off score`: the arrival across every channel a skill taking
over would care about, squashed into (0, 1]. It is the same number training multiplies the
whole second window by, so this panel and the reward agree by construction.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
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

# The three stretches of one loop.
BEFORE, BRIDGING, AFTER = 0, 1, 2


def find_checkpoint(explicit: Path | None, experiment: str) -> Path:
  """The policy to load, and a loud statement of which one it is.

  Picked by modification time when not given, which is convenient and has bitten this
  project before: a stale run left in `logs/` outranks the checkpoint you meant. The path
  is printed rather than assumed, so a comparison against the wrong policy is at least a
  visible mistake.
  """
  if explicit is not None:
    if not explicit.exists():
      raise SystemExit(f"No checkpoint at {explicit}.")
    return explicit
  root = Path("logs") / "rsl_rl" / experiment
  found = sorted(root.rglob("model_*.pt"), key=lambda p: p.stat().st_mtime)
  if not found:
    raise SystemExit(
      f"No checkpoint under {root}. Train one first:\n"
      f"  uv run train {BRIDGE_TASK_ID} --env.scene.num-envs 4096"
    )
  print(f"[INFO]: Loading checkpoint: {found[-1]} (auto)")
  return found[-1]


def load_policy(env: ManagerBasedRlEnv, checkpoint: Path, device: str):
  """The trained actor, loaded the way every other frozen policy here is loaded."""
  agent_cfg = load_rl_cfg(BRIDGE_TASK_ID)
  runner_cls = load_runner_cls(BRIDGE_TASK_ID) or MjlabOnPolicyRunner
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  return runner.get_inference_policy(device=device), wrapped


class BridgeViewer:
  """One window at a time: clip in, bridge across, clip out, reference alongside."""

  def __init__(self, server, env: ManagerBasedRlEnv, policy, wrapped, g1: G1) -> None:
    self.server = server
    self.env = env
    self.policy = policy
    self.wrapped = wrapped
    self.g1 = g1
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
    server.scene.add_grid("/ground", width=14.0, height=14.0, cell_size=0.5)

    self.index = 0
    self.phase = BEFORE
    self.cursor = 0
    self._showing_hole = False
    self._syncing = False
    self.arrival_error = (0.0, 0.0)
    self._build_gui()
    self.load(0)

  def _build_gui(self) -> None:
    gui = self.server.gui
    with gui.add_folder("Window"):
      self.sl_index = gui.add_slider(
        "Entry",
        min=0,
        max=max(self.command.num_windows - 1, 1),
        step=1,
        initial_value=0,
      )
      self.btn_prev = gui.add_button("Previous")
      self.btn_next = gui.add_button("Next")
      self.btn_again = gui.add_button("Replay")
      self.html = gui.add_html("")

      @self.sl_index.on_update
      def _(_) -> None:
        if not self._syncing:
          self.load(int(self.sl_index.value))

      @self.btn_prev.on_click
      def _(_) -> None:
        self.load(self.index - 1)

      @self.btn_next.on_click
      def _(_) -> None:
        self.load(self.index + 1)

      @self.btn_again.on_click
      def _(_) -> None:
        self.load(self.index)

    with gui.add_folder("Playback"):
      self.cb_play = gui.add_checkbox("Play", initial_value=True)
      self.sl_speed = gui.add_slider(
        "Speed", min=0.1, max=1.5, step=0.1, initial_value=0.5
      )
      self.cb_resume = gui.add_checkbox(
        "Resume clip after bridge",
        initial_value=True,
        hint="On: the clip is written back, so arrival error shows as a jump. "
        "Off: the policy keeps driving.",
      )
      self.cb_reference = gui.add_checkbox("Show reference ghost", initial_value=True)

  def load(self, index: int) -> None:
    """Pin window `index`, reset the environment onto its hand-off, rewind to the start."""
    self.index = index % self.command.num_windows
    self.command.force_window = self.index
    self.env.reset()

    self.gap = int(self.command.gap[0])
    self.before = self.command.context_before(0).cpu().numpy()
    self.after = self.command.reference[0].cpu().numpy()
    self.resume = self.command.resume
    self.phase = BEFORE
    self.cursor = 0
    self.arrival_error = (0.0, 0.0)

    self._syncing = True
    self.sl_index.value = self.index
    self._syncing = False
    self._draw()

  def _qpos_of(self, state: np.ndarray) -> np.ndarray:
    """The rendering pose of a corpus state: root pose then joint angles."""
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
      self.cursor = max(int(self.command.step[0]), 0)
      if self.cursor >= self.gap:
        # The hand-back. How far the robot is from where the clip resumes is the number
        # this whole architecture is judged on.
        target = self.after[min(self.gap, self.after.shape[0] - 1)]
        actual = self._robot_qpos()
        self.arrival_error = (
          float(np.linalg.norm(actual[0:3] - target[0:3])),
          float(np.abs(actual[7:] - self._qpos_of(target)[7:]).mean()),
        )
        self.phase = AFTER
      return

    # After the hole: either the clip is written back, or the policy carries on.
    if self.cb_resume.value:
      self.cursor += 1
    else:
      obs = self.wrapped.get_observations()
      with torch.inference_mode():
        action = self.policy(obs)
      self.env.step(action)
      self.cursor = max(int(self.command.step[0]), 0)
    if self.cursor >= self.gap + self.resume:
      self.load(self.index)

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
      self._showing_hole = in_hole
    self._update_info()

  def _update_info(self) -> None:
    name = ("clip", "bridge", "clip again")[self.phase]
    colour = ("#3cc85a", "#e13a2d", "#3cc85a")[self.phase]
    distance, joints = self.arrival_error
    self.html.content = (
      '<div style="font-size:0.85em;line-height:1.4;padding:0 1em 0.5em 1em;">'
      f"<b>Window:</b> {self.index + 1} / {self.command.num_windows}"
      f"<br/><b>Hole:</b> {self.gap} frames ({self.gap / 50.0:.2f} s)"
      f"<br/><b>Now:</b> <span style='color:{colour}'>{name}</span> "
      f"({self.cursor + 1})"
      f"<br/><b>At hand-back:</b> {distance:.3f} m, {joints:.3f} rad"
      f"<br/><b>Hand-off score:</b> {float(self.command.hand_off_score[0]):.3f}"
      "</div>"
    )

  def step(self, dt: float) -> None:
    if self.cb_play.value:
      self.advance()
    self._draw()


@dataclass(frozen=True)
class Config:
  checkpoint: Path | None = None
  """Defaults to the newest under logs/<experiment>. The resolved path is printed."""

  split: str = "train"
  """Which subjects the windows come from. 'eval' is motion the policy never trained on."""

  port: int = 8080
  device: str | None = None


def main(cfg: Config) -> None:
  import viser

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = bridge_env_cfg(play=True, split=cfg.split)
  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  agent_cfg = load_rl_cfg(BRIDGE_TASK_ID)
  checkpoint = find_checkpoint(cfg.checkpoint, agent_cfg.experiment_name)
  print(f"policy: {checkpoint}")
  policy, wrapped = load_policy(env, checkpoint, device)

  server = viser.ViserServer(port=cfg.port, label="Bridge evaluation")
  viewer = BridgeViewer(server, env, policy, wrapped, G1())

  print(f"\nViewer at http://localhost:{cfg.port} -- Ctrl-C to quit.")
  last = time.time()
  try:
    while True:
      now = time.time()
      if now - last >= (1.0 / 50.0) / max(viewer.sl_speed.value, 0.1):
        viewer.step(now - last)
        last = now
      time.sleep(1.0 / 240.0)
  except KeyboardInterrupt:
    print("\nShutting down.")
  finally:
    env.close()


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
