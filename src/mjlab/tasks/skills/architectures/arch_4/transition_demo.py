"""Throwaway: one walk-to-jump transition, bridged and unbridged, side by side.

    uv run python -m mjlab.tasks.skills.architectures.arch_4.transition_demo

Not part of the architecture and not wired into anything. It exists to make one question
answerable by looking: does putting the bridge between walking and jumping produce a
better hand-off than dropping the jump on the robot mid-stride.

Two environments, same arena, same commands, same walking, and they differ only in what
happens at the switch:

    solid       walk, then the bridge drives, then the jump entered at its crouch
    ghost       walk, then the jump entered at its own first frame, with nothing between

The ghost is a real robot in a real simulation, not a replayed clip. That matters: a
replay would show what the clips look like, and what is actually in question is what the
skills do to a body. It is drawn translucent and moved on top of the solid one so the two
can be compared rather than merely displayed, and it is coloured by whichever skill is
driving it, green for walk and amber for jump.

The two never run at once. The robot walks, the switch fires, and only then does the jump
matter, which is the whole shape of the problem.

One honest caveat about what the crouch entry can buy. Every ASAP jump clip opens with the
subject standing still, so its crouch is a stationary crouch, and arriving there at walking
speed hands the jump a reference that wants a vertical launch from a standstill. What this
should show is the stand-up-and-settle being skipped, roughly a second and a half of it,
not a running jump. A running jump needs a clip that opens with a run-up.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.architectures.arch_4.bridge.dataset import G1
from mjlab.tasks.skills.architectures.arch_4.bridge.view import Ghost, visual_meshes
from mjlab.tasks.skills.architectures.arch_4.selector import crouch_frame

BRIDGED, NAIVE = 0, 1
WALKING, BRIDGING, JUMPING = 0, 1, 2

SOLID_COLOR = (70, 150, 255)
GHOST_COLOR = (150, 150, 155)


@dataclass(frozen=True)
class Config:
  walk_steps: int = 120
  """Control steps of walking before the switch fires."""

  walk_speed: float = 1.0
  """Forward speed held constant throughout, in m/s. The jump is aimed the same way."""

  bridge_steps: int = 25
  """How long the bridge is given. Inside the 10 to 50 it was trained over."""

  jump_steps: int = 160
  """Steps to keep the jump running afterwards."""

  bridge_checkpoint: Path | None = None
  """Newest under logs/rsl_rl/bridge_in_betweener when not given."""

  port: int = 8080
  device: str | None = None


def _jump_states(command) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
  """The jump clip as (frames, 13 + 2J) states, plus root height and airborne mask."""
  motion = command.motion
  clip = int(command.motion_ids[0])
  states = torch.cat(
    [
      motion.body_pos_w[clip, :, 0],
      motion.body_quat_w[clip, :, 0],
      motion.body_lin_vel_w[clip, :, 0],
      motion.body_ang_vel_w[clip, :, 0],
      motion.joint_pos[clip],
      motion.joint_vel[clip],
    ],
    dim=-1,
  )
  root_z = motion.body_pos_w[clip, :, 0, 2].cpu().numpy()
  lowest = motion.body_pos_w[clip, :, :, 2].min(dim=-1).values.cpu().numpy()
  airborne = lowest > (np.median(lowest[:20]) + 0.06)
  return states, root_z, airborne


def load_bridge_actor(path: Path, device: str):
  """Rebuild the bridge's actor from its checkpoint, with no environment involved.

  The usual loader goes through the runner, which sizes the network from whatever
  environment it is handed. Here the environment is the arena, whose observation is a
  different width, so that path raises rather than quietly loading a mismatched network.
  The checkpoint describes the actor completely, so it is rebuilt directly.

  Only the mean action is taken; the exploration standard deviation in the checkpoint is
  a training artefact and sampling from it here would add noise to a demonstration.
  """
  blob = torch.load(path, map_location=device, weights_only=False)
  state = blob["actor_state_dict"]

  widths = [
    state[k].shape for k in state if k.startswith("mlp.") and k.endswith("weight")
  ]
  layers: list[torch.nn.Module] = []
  for i, (out_dim, in_dim) in enumerate(widths):
    if i:
      layers.append(torch.nn.ELU())
    layers.append(torch.nn.Linear(in_dim, out_dim))
  mlp = torch.nn.Sequential(*layers).to(device)
  mlp.load_state_dict(
    {k[len("mlp.") :]: v for k, v in state.items() if k.startswith("mlp.")}
  )
  mlp.eval()

  mean = state["obs_normalizer._mean"].to(device)
  std = state["obs_normalizer._std"].to(device)
  eps = 1e-2  # rsl_rl's EmpiricalNormalization default, and it is not negligible.
  obs_dim = int(mean.shape[-1])

  def act(obs: torch.Tensor) -> torch.Tensor:
    if obs.shape[-1] != obs_dim:
      raise ValueError(
        f"The bridge was trained on a {obs_dim}-wide observation and this one is "
        f"{obs.shape[-1]}. The layout in `_bridge_obs` has drifted from the one in "
        f"bridge/env_cfg.py; they have to be changed together."
      )
    with torch.inference_mode():
      return mlp((obs - mean) / (std + eps))

  print(f"bridge actor: {obs_dim} -> {widths[-1][0]}, {len(widths)} layers")
  return act


class TransitionDemo:
  def __init__(self, server, env: ManagerBasedRlEnv, pool, bridge, cfg: Config) -> None:
    self.server = server
    self.env = env
    self.pool = pool
    self.bridge = bridge
    self.cfg = cfg
    self.walk, self.jump = pool[0], pool[2]
    self.robot = env.scene["robot"]

    self.jump_command = self.jump.command
    states, root_z, airborne = _jump_states(self.jump_command)
    self.clip = states
    self.crouch = crouch_frame(root_z, airborne)
    print(f"jump clip: {states.shape[0]} frames, crouch at {self.crouch}")

    self.g1 = G1()
    meshes = visual_meshes(self.g1.model)
    self.solid = Ghost(server, meshes, "solid", SOLID_COLOR, opacity=1.0)
    self.ghost = Ghost(server, meshes, "ghost", GHOST_COLOR, opacity=0.35)
    server.scene.add_grid("/ground", width=20.0, height=20.0, cell_size=0.5)
    self._build_gui()
    self.restart()

  def _build_gui(self) -> None:
    gui = self.server.gui
    self.cb_play = gui.add_checkbox("Play", initial_value=True)
    self.sl_speed = gui.add_slider(
      "Speed", min=0.1, max=1.5, step=0.1, initial_value=0.4
    )
    self.cb_ghost = gui.add_checkbox("Show unbridged ghost", initial_value=True)
    self.btn_restart = gui.add_button("Restart")
    self.html = gui.add_html("")

    @self.btn_restart.on_click
    def _(_) -> None:
      self.restart()

  def restart(self) -> None:
    self.env.reset()
    self.pool.reset(
      torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
    )
    self.phase = WALKING
    self.tick = 0
    self.origins = self.env.scene.env_origins.clone()
    if not hasattr(self, "last_outcome"):
      self.last_outcome = ""
    self._draw()

  def _walk_command(self) -> None:
    """Hold the twist at a constant forward walk, the way the corridor does.

    Rewritten every step rather than once: the arena resamples the twist on its own
    schedule, and a walk that changes direction halfway through would make the two
    environments differ for a reason that has nothing to do with the bridge.
    """
    term = self.env.command_manager.get_term("twist")
    assert term is not None
    term.command[:] = torch.tensor(
      [self.cfg.walk_speed, 0.0, 0.0], device=self.env.device
    )

  def _bridge_obs(self) -> torch.Tensor:
    """The bridge's observation, rebuilt in the arena's coordinates.

    The layout has to match what the bridge was trained on, term for term and in order:
    the command first, then proprioception. The arena's own `actor` group is close but
    carries a base linear velocity the bridge never saw, so this is assembled from the
    robot rather than borrowed.
    """
    from mjlab.tasks.skills.architectures.arch_4.bridge import frames

    data = self.robot.data
    env_ids = torch.tensor([BRIDGED], device=self.env.device)

    # The strip was placed once, at the switch, and does not move while the bridge runs.
    # Recomputing it per step against the robot's live pose is what made the target chase
    # the robot instead of the robot chasing the target.
    index = torch.linspace(0, self.target_strip.shape[0] - 1, 5, device=self.env.device)
    target = self.target_strip[index.long()].unsqueeze(0)
    base_pos = data.root_link_pos_w[env_ids]
    base_yaw = frames.yaw_quat(data.root_link_quat_w[env_ids])
    goal = frames.encode(target, base_pos, base_yaw).flatten(start_dim=1)

    progress = torch.tensor(
      [[min(self.tick / self.cfg.bridge_steps, 1.0), self.cfg.bridge_steps / 75.0]],
      device=self.env.device,
    )
    return torch.cat(
      [
        goal,
        progress,
        data.root_link_ang_vel_b[env_ids],
        data.projected_gravity_b[env_ids],
        (data.joint_pos - data.default_joint_pos)[env_ids],
        (data.joint_vel - data.default_joint_vel)[env_ids],
        self.env.action_manager.action[env_ids],
      ],
      dim=-1,
    )

  def _begin_switch(self) -> None:
    """Decide, once, where the jump is going to happen. Everything else follows.

    Three things have to agree from here on: where the bridge is aiming, where the jump's
    reference sits, and where the robot actually ends up. The first two are fixed now, at
    the same time and from the same numbers, and the third is what the bridge is judged
    on. Deciding the anchor later, or twice, is what makes a transition look like the
    skills teleport: each one places its clip against a robot that has since moved.

    The predicted hand-off pose is the robot's current one carried forward at its current
    velocity for as long as the bridge is given. Crude, and the stand-in for a selector
    that will eventually get this from where the obstacle is.
    """
    from mjlab.utils.lab_api.math import quat_apply, quat_conjugate, quat_mul, yaw_quat

    data = self.robot.data
    device = self.env.device
    self.solid_ids = torch.tensor([BRIDGED], device=device)
    self.naive_ids = torch.tensor([NAIVE], device=device)

    duration = self.cfg.bridge_steps * self.env.step_dt
    self.handoff_pos = data.root_link_pos_w[self.solid_ids].clone()
    self.handoff_pos[:, :2] += data.root_link_lin_vel_w[self.solid_ids, :2] * duration
    self.handoff_quat = yaw_quat(data.root_link_quat_w[self.solid_ids]).clone()

    # The clip, rotated to the hand-off heading and translated so its crouch frame lands
    # on the hand-off pose. This is the same transform anchor_to_robot will apply at
    # hand-over, which is why the bridge and the jump end up talking about one place.
    clip = self.clip.to(device)
    # The command stretches the clip horizontally to reach its commanded distance, so a
    # strip built from the raw clip would aim at a jump the skill is not going to make.
    stretch = self.jump_command.stretch_offset[self.solid_ids]
    clip = clip.clone()
    clip[:, 0:2] += stretch
    entry = clip[self.crouch]
    delta = quat_mul(
      self.handoff_quat, quat_conjugate(yaw_quat(entry[3:7].unsqueeze(0)))
    )
    strip = clip[self.crouch : self.crouch + 25].clone()
    span = strip.shape[0]
    rot = delta.expand(span, 4)
    # Only the horizontal placement comes from the hand-off; height stays the clip's own,
    # because that is what anchor_to_robot does and the two have to agree exactly.
    placement = self.handoff_pos.clone()
    placement[:, 2] = entry[2]
    strip[:, 0:3] = quat_apply(rot, strip[:, 0:3] - entry[0:3]) + placement
    strip[:, 3:7] = quat_mul(rot, strip[:, 3:7])
    strip[:, 7:10] = quat_apply(rot, strip[:, 7:10])
    strip[:, 10:13] = quat_apply(rot, strip[:, 10:13])
    self.target_strip = strip

    # The ghost is the naive composition: the jump lands on it at once, from the clip's
    # own first frame, anchored to wherever it happens to be. It is allowed to teleport,
    # which is exactly what a hand-off with nothing in between looks like.
    self.jump.entry_frame = 0
    self.jump.reset(torch.tensor([False, True], device=device))

  def advance(self) -> None:
    self._walk_command()
    obs = self.env.observation_manager.compute()
    actions = self.walk.act(
      obs, torch.ones_like(self.env.episode_length_buf, dtype=torch.bool)
    )

    if self.phase == WALKING and self.tick >= self.cfg.walk_steps:
      self._begin_switch()
      self.phase, self.tick = BRIDGING, 0

    if self.phase == BRIDGING:
      if self.bridge is not None:
        with torch.inference_mode():
          bridged = self.bridge(self._bridge_obs())
        actions = actions.clone()
        actions[BRIDGED] = bridged[0]
      jump_actions = self.jump.act(
        obs, torch.ones_like(actions[:, 0], dtype=torch.bool)
      )
      actions[NAIVE] = jump_actions[NAIVE]
      if self.tick >= self.cfg.bridge_steps:
        # Hand over at the anchor decided at the switch, not at wherever the robot ended
        # up. Re-anchoring here would slide the clip onto the robot and hide the arrival
        # error, which is the one thing this demo is meant to show.
        self.jump_command.anchor_to_robot(
          self.solid_ids,
          start_frame=self.crouch,
          at_pos=self.handoff_pos,
          at_quat=self.handoff_quat,
        )
        self.phase, self.tick = JUMPING, 0

    elif self.phase == JUMPING:
      actions = self.jump.act(obs, torch.ones_like(actions[:, 0], dtype=torch.bool))

    self.env.step(actions)
    self.tick += 1

    # A fall ends that environment's episode, and mjlab resets it inside `step`: the
    # robot reappears at its spawn with no announcement. Left alone the two environments
    # drift out of step and the comparison quietly stops being one, so the run is ended
    # here and what happened is said out loud. A ghost that falls is the result, not a
    # glitch: it is what a hand-off with nothing in between did to the robot.
    fell = self.env.termination_manager.terminated
    if bool(fell.any()):
      who = " and ".join(
        name for i, name in ((BRIDGED, "bridged"), (NAIVE, "ghost")) if bool(fell[i])
      )
      self.last_outcome = f"{who} fell at {('walk', 'bridge', 'jump')[self.phase]}"
      print(f"  {self.last_outcome}")
      self.restart()
      return

    if self.phase == JUMPING and self.tick >= self.cfg.jump_steps:
      self.restart()

  def _qpos(self, env_id: int, shift: torch.Tensor | None = None) -> np.ndarray:
    data = self.robot.data
    pos = data.root_link_pos_w[env_id].clone()
    if shift is not None:
      pos = pos - shift
    return np.concatenate(
      [
        pos.cpu().numpy(),
        data.root_link_quat_w[env_id].cpu().numpy(),
        data.joint_pos[env_id].cpu().numpy(),
      ]
    )

  def _pose(self, ghost: Ghost, qpos: np.ndarray) -> None:
    d = self.g1.data
    d.qpos[0:3], d.qpos[3:7], d.qpos[7:] = qpos[0:3], qpos[3:7], qpos[7:]
    mujoco.mj_kinematics(self.g1.model, d)
    ghost.pose(d)

  def _draw(self) -> None:
    # The ghost is moved onto the solid one's lane, so the two overlap and the
    # difference between them is the only thing left to see.
    offset = self.origins[NAIVE] - self.origins[BRIDGED]
    with self.server.atomic():
      self._pose(self.solid, self._qpos(BRIDGED))
      if self.cb_ghost.value:
        self._pose(self.ghost, self._qpos(NAIVE, shift=offset))
      self.ghost.visible = self.cb_ghost.value
    self._info()

  def _info(self) -> None:
    name = ("walking", "BRIDGE", "jumping")[self.phase]
    height = float(self.robot.data.root_link_pos_w[BRIDGED, 2])
    ghost_height = float(self.robot.data.root_link_pos_w[NAIVE, 2])
    outcome = f"<br/><b>Last run:</b> {self.last_outcome}" if self.last_outcome else ""
    self.html.content = (
      '<div style="font-size:0.85em;line-height:1.4;padding:0 1em 0.5em 1em;">'
      f"<b>Phase:</b> {name} (step {self.tick})"
      f"<br/><b>Jump entry frame:</b> {self.crouch} (crouch) vs 0 (ghost)"
      f"<br/><b>Root height:</b> {height:.3f} m solid, {ghost_height:.3f} m ghost"
      f"{outcome}"
      "</div>"
    )

  def step(self) -> None:
    if self.cb_play.value:
      self.advance()
    self._draw()


def main(cfg: Config) -> None:
  import viser

  from mjlab.tasks.skills.architectures.arch_4.bridge.evaluate import find_checkpoint
  from mjlab.tasks.skills.experiments.parkour import build_pool
  from mjlab.tasks.skills.experiments.parkour.arena import parkour_arena_env_cfg

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = parkour_arena_env_cfg()
  env_cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  pool = build_pool(env, device)

  bridge = None
  try:
    path = find_checkpoint(cfg.bridge_checkpoint, "bridge_in_betweener")
    print(f"bridge: {path}")
    print(
      "  NOTE: if this predates the action-scale fix it will drive the robot wrongly; "
      "retrain with `uv run train Mjlab-Bridge-G1`."
    )
    bridge = load_bridge_actor(path, device)
  except SystemExit as err:
    print(f"no bridge ({err}); the solid robot will hand over with nothing in between")

  server = viser.ViserServer(port=cfg.port, label="walk -> bridge -> jump")
  demo = TransitionDemo(server, env, pool, bridge, cfg)
  print(f"\nViewer at http://localhost:{cfg.port} -- Ctrl-C to quit.")

  last = time.time()
  try:
    while True:
      now = time.time()
      if now - last >= env.step_dt / max(demo.sl_speed.value, 0.1):
        demo.step()
        last = now
      time.sleep(1.0 / 240.0)
  except KeyboardInterrupt:
    print("\nShutting down.")
  finally:
    env.close()


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
