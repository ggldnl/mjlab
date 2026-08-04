"""Watch one hand-over: the skill being left, then the skill being entered.

Pick a current skill and a next skill. The viewer shows, on a loop:

- the current skill in its last steps before the hand-over (its closing window), cut at
  a point drawn from the same range training draws from;
- then the next skill opening from its own start (its opening window).

That pair is exactly what a bridge is trained on, and the gap between the end of the
first half and the start of the second is the bridge's job. On the parkour experiment
the pair worth looking at is run into jump: the left half is a robot at speed, the
right half is a clip that opens from a stand.

Nothing here reads a stored dataset. The couple is rolled live, the same way the
collectors in windows.py roll theirs, and then recorded (the frames are kept and
replayed). That split matters for what you see: a closing window is the end of a longer
run, so producing it means stepping the simulator up to a couple of hundred times, and
doing that inside the playback loop is a visible stall every time the loop comes round.
Recording once and replaying costs a pause when a selection changes and nothing after.

Controls:

- `Current skill` is the one being handed away from, `Next skill` the one being handed
  to. Changing either records a fresh couple.
- The viewer's own reset button re-records the same pair with a new hand-over point, so
  it doubles as "show me another one".
- Everything else (pause, single step, speed, camera) is the play viewer's, unchanged.

An experiment wires this up in three lines; see `experiments/*/inspect.py`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from typing_extensions import override

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.tasks.skills.skill import SkillPool
from mjlab.tasks.skills.windows import WindowPlan, start_skill
from mjlab.viewer.base import EnvProtocol, ViewerAction
from mjlab.viewer.viser import ViserPlayViewer

# Builds the experiment's skills against an env. Matches every experiment's
# `build_pool(env, device, ...)`.
PoolFactory = Callable[[ManagerBasedRlEnv, str], SkillPool]

# One entity's pose at one instant: its root state (13), and its joint positions and
# velocities. Either is None when the entity does not have that kind of state -- fixed
# to the world, or jointless. Enough to redraw the frame and nothing more, since a
# recorded couple is only ever looked at.
EntityFrame = tuple[Optional[torch.Tensor], Optional[tuple[torch.Tensor, torch.Tensor]]]

# How many times to re-roll a couple whose source skill broke before reaching its cut.
_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class InspectConfig:
  """Options every experiment's inspect entry point shares."""

  source: int = 0
  """The skill handed away from, by id. Selectable in the viewer; this is the opening
  choice."""

  target: int = 1
  """The skill handed to, by id. Selectable in the viewer; this is the opening choice."""

  device: str | None = None


@dataclass
class Couple:
  """One recorded hand-over: frames to replay, and what each of them is."""

  frames: list[dict[str, EntityFrame]] = field(default_factory=list)
  labels: list[str] = field(default_factory=list)
  cut: int = 0
  note: str = ""
  """Set when the roll did not go as planned, e.g. the source skill fell over first."""

  def __len__(self) -> int:
    return len(self.frames)


class WindowViewer(ViserPlayViewer):
  """Replays a recorded couple, and re-records it when the pair changes.

  The base viewer's "advance one step" hook is the only thing that differs from the play
  viewer: instead of asking a policy for an action, it writes the next recorded frame
  into the simulator. Nothing is being simulated during playback, which is the point.
  """

  def __init__(
    self,
    env: EnvProtocol,
    pool: SkillPool,
    source: int,
    target: int,
    plan: WindowPlan,
  ) -> None:
    super().__init__(env, policy=lambda obs: obs, info_provider=self._active_label)
    plan.check(pool)
    self.pool = pool
    self.plan = plan
    self.source = source
    self.target = target
    self._couple = Couple()
    self._frame = 0
    self._active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  ##
  # Recording.
  ##

  def _capture(self) -> dict[str, EntityFrame]:
    """Every entity's current pose, cloned out of the simulator.

    Both halves are optional and for opposite reasons: an entity fixed to the world has
    no root state to record, and one with no joints (a prop, a marker) has no joint
    state. A scene usually holds some of each.
    """
    frame: dict[str, EntityFrame] = {}
    for name, entity in self.env.unwrapped.scene.entities.items():
      root = None
      if not entity.is_fixed_base:
        root = torch.cat(
          [entity.data.root_link_pose_w, entity.data.root_link_vel_w], dim=-1
        ).clone()
      joints = None
      if entity.is_articulated:
        joints = (entity.data.joint_pos.clone(), entity.data.joint_vel.clone())
      frame[name] = (root, joints)
    return frame

  def _apply(self, frame: dict[str, EntityFrame]) -> None:
    """Put a recorded frame back in the simulator so the viewer draws it."""
    env = self.env.unwrapped
    for name, (root, joints) in frame.items():
      entity = env.scene[name]
      if root is not None:
        entity.write_root_state_to_sim(root)
      if joints is not None:
        entity.write_joint_state_to_sim(*joints)
    env.scene.write_data_to_sim()
    env.sim.forward()

  @torch.no_grad()
  def _roll(self) -> Couple:
    """Roll the pair once and keep the frames, the way the collectors roll theirs."""
    env = self.env.unwrapped
    source, target = self.pool[self.source], self.pool[self.target]
    source_spec, target_spec = self.plan[source], self.plan[target]
    couple = Couple(cut=int(source_spec.sample_cuts(1, env.device).item()))

    # The half being left. A closing window ends at the cut, so the run up to it is
    # rolled and only the last `closing` frames are kept.
    obs, _ = env.reset()
    obs = start_skill(env, source_spec, obs)
    self.pool.reset(self._active)
    history: deque[dict[str, EntityFrame]] = deque(maxlen=source_spec.closing)
    for step in range(couple.cut):
      obs, _, terminated, time_out, _ = env.step(source.act(obs, self._active))
      if bool((terminated | time_out).any()):
        # The env has auto-reset, so everything after this belongs to a fresh episode.
        couple.note = f"'{source.name}' broke at step {step + 1} of {couple.cut}"
        break
      history.append(self._capture())
    couple.frames += list(history)
    couple.labels += [f"leaving '{source.name}'"] * len(history)

    # What the source would have gone on to do, for the experiments that ask to see it.
    for _ in range(source_spec.overrun):
      obs, _, terminated, time_out, _ = env.step(source.act(obs, self._active))
      if bool((terminated | time_out).any()):
        break
      couple.frames.append(self._capture())
      couple.labels.append(f"'{source.name}' carrying on past the hand-over")

    # The half being entered, from its own start rather than from where the first half
    # ended: that is what the skill was trained to do, and the distance between the two
    # is what a bridge exists to close.
    obs, _ = env.reset()
    obs = start_skill(env, target_spec, obs)
    target.reset(self._active)
    for _ in range(target_spec.opening):
      obs, _, terminated, time_out, _ = env.step(target.act(obs, self._active))
      couple.frames.append(self._capture())
      couple.labels.append(f"entering '{target.name}'")
      if bool((terminated | time_out).any()):
        break

    return couple

  def _record(self) -> None:
    """Record a couple, re-rolling a few times if the source keeps breaking early."""
    couple = Couple()
    with self._sim_lock:
      for _ in range(_MAX_ATTEMPTS):
        couple = self._roll()
        if not couple.note:
          break
      self._couple = couple
      self._frame = 0
      if couple.frames:
        self._apply(couple.frames[0])
    if couple.note:
      self.log(f"[inspect] {couple.note}; showing what was reached.")
    self._step_count = 0

  ##
  # The viewer.
  ##

  def _active_label(self, env_idx: int) -> str:
    del env_idx
    if not self._couple.labels:
      return "nothing recorded"
    return self._couple.labels[self._frame]

  @override
  def setup(self) -> None:
    # Built before the base viewer's tab group so it sits at the top of the panel.
    names = [skill.name for skill in self.pool.skills]
    with self._server.gui.add_folder("Hand-over"):
      current = self._server.gui.add_dropdown(
        "Current skill", options=names, initial_value=names[self.source]
      )

      @current.on_update
      def _(_) -> None:
        chosen = names.index(current.value)
        self.request_action("CUSTOM", {"type": "couple", "source": chosen})

      following = self._server.gui.add_dropdown(
        "Next skill", options=names, initial_value=names[self.target]
      )

      @following.on_update
      def _(_) -> None:
        chosen = names.index(following.value)
        self.request_action("CUSTOM", {"type": "couple", "target": chosen})

      self._info = self._server.gui.add_html("")

    super().setup()
    self._record()

  @override
  def _handle_custom_action(self, action: ViewerAction, payload: Optional[Any]) -> bool:
    if isinstance(payload, dict) and payload.get("type") == "couple":
      if "source" in payload:
        self.source = payload["source"]
      if "target" in payload:
        self.target = payload["target"]
      self._record()
      return True
    return super()._handle_custom_action(action, payload)

  @override
  def _execute_step(self) -> bool:
    """Draw the next recorded frame, looping back to the start at the end."""
    if not self._couple.frames:
      return True
    self._frame = (self._frame + 1) % len(self._couple.frames)
    with self._sim_lock:
      self._apply(self._couple.frames[self._frame])
    self._step_count += 1
    self._stats_steps += 1
    return True

  @override
  def reset_environment(self) -> None:
    # Re-record rather than reset: the pair is unchanged, but the hand-over point is
    # drawn again, so this is "show me another one from the same pair".
    self._record()

  @override
  def sync_env_to_viewer(self) -> None:
    super().sync_env_to_viewer()
    source_spec = self.plan[self.pool[self.source]]
    low, high = source_spec.interrupt_range
    self._info.content = (
      '<div style="font-size:0.85em;line-height:1.35;padding:0 1em 0.5em 1em;">'
      f"<strong>Hand-over at:</strong> step {self._couple.cut} of "
      f"'{self.pool[self.source].name}' (range {low} to {high})<br/>"
      f"<strong>Showing:</strong> {self._active_label(0)}<br/>"
      f"<strong>Frame:</strong> {self._frame + 1} / {len(self._couple)}"
      + (
        f"<br/><strong>Note:</strong> {self._couple.note}" if self._couple.note else ""
      )
      + "</div>"
    )


def run_inspect(
  cfg: InspectConfig,
  task_id: str,
  build_pool: PoolFactory,
  plan: WindowPlan,
  env_cfg: ManagerBasedRlEnvCfg | None = None,
) -> None:
  """Open the viewer for one experiment.

  `build_pool` builds that experiment's skills in the arena, and `plan` is the same
  `WindowPlan` its training uses.

  `task_id` names a registered task; its play-mode env is the arena when `env_cfg` is
  not given, and its action clipping is used either way. Pass `env_cfg` when the arena
  is not a registered task in its own right, which is the case as soon as an
  experiment's skills come from different environments: the pool then reads observation
  groups that only the composed arena carries, and building the env from any one skill's
  task gives a pool whose policies cannot find their own observations.

  Whatever env arrives, this wants it bare. A hand-over is a thing that happens to the
  robot, and terrain the couple was not rolled against is scenery in the way of it.
  """
  import mjlab.tasks  # noqa: F401  (populates the task registry)
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  if env_cfg is None:
    env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  pool = build_pool(env, device)

  for role, skill_id in (("--source", cfg.source), ("--target", cfg.target)):
    if not 0 <= skill_id < len(pool):
      raise ValueError(
        f"{role} {skill_id} is not a skill id; the pool has "
        f"{len(pool)}: {[s.name for s in pool.skills]}."
      )

  viewer_env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task_id).clip_actions)
  WindowViewer(viewer_env, pool, cfg.source, cfg.target, plan).run()
  viewer_env.close()
