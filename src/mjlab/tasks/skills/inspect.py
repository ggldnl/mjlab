"""Watch the windows a bridge is trained on, on whatever robot the experiment uses.

The dataset consists of windows over skill rollouts:
- Roll the source skill, cut it somewhere in a range, keep the last N steps
- Roll the target skill from its own reset, keep the first N steps

This viewer follows the same logic rather than replaying stored frames.
Effectively, it's like dynamically building a dataset and looking inside it:
the rollouts won't be the same as the ones used for training, but should
capture what we will end up using.

Each skill carries its own `SkillWindowSpec`, so the closing window and the cut range
come from the skill being left and the opening window from the skill being entered.
Overrun windows show up here too when a skill declares one: after the cut, the source
skill keeps driving, so you see what would have happened had control not been taken
away.

Controls:

- `Show` picks which window runs. `Closing` and `Overrun` belong to the source skill,
  `Opening` to the target.
- `Couple` walks the cut point across the source skill's range, so Previous and Next
  sweep from the earliest hand-over its spec allows to the latest. Each couple is
  seeded, so going back shows the same episode again.
- Everything else (pause, single step, speed, camera) is the play viewer's, unchanged.

An experiment wires this up in three lines; see `experiments/*/inspect.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import torch
from typing_extensions import override

from mjlab.envs import ManagerBasedRlEnv, VecEnvObs
from mjlab.tasks.skills.skill import SkillPool
from mjlab.tasks.skills.windows import (
  CLOSING,
  OPENING,
  OVERRUN,
  WindowPlan,
  start_skill,
)
from mjlab.viewer.base import EnvProtocol, ViewerAction
from mjlab.viewer.viser import ViserPlayViewer

# Builds the experiment's skills against an env. Matches every experiment's
# `build_pool(env, device, ...)`.
PoolFactory = Callable[[ManagerBasedRlEnv, str], SkillPool]

# What each role is called in the GUI, in the order a hand-over happens.
ROLE_LABELS = {CLOSING: "Closing", OVERRUN: "Overrun", OPENING: "Opening"}


@dataclass(frozen=True)
class InspectConfig:
  """Options every experiment's inspect entry point shares."""

  target: int = 1
  """The skill the bridge is aimed at: it supplies the opening window, and the closing
  and overrun windows come from one of the others."""

  device: str | None = None


class WindowViewer(ViserPlayViewer):
  """Runs the skills live, following the same window recipe training follows.

  Only the "advance one step" hook differs from the play viewer: instead of asking a
  policy, it drives whichever skill the current window belongs to and restarts the
  episode when the window ends.
  """

  def __init__(
    self,
    env: EnvProtocol,
    pool: SkillPool,
    target: int,
    plan: WindowPlan,
  ) -> None:
    super().__init__(env, policy=lambda obs: obs, info_provider=self._active_label)
    plan.check(pool)
    self.pool = pool
    self.target = target
    self.plan = plan
    self.others = [i for i in range(len(pool)) if i != target]
    if not self.others:
      raise ValueError("A bridge needs at least one other skill to come from.")
    self.source = self.others[0]
    self.role = self._roles()[0]
    self.index = 0
    self._frame = 0
    self._obs: VecEnvObs | None = None
    self._active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    # Captured the first time a couple is shown and replayed on every loop after, so the
    # window repeats exactly instead of re-rolling a new episode each cycle.
    self._rng: tuple[torch.Tensor, list[torch.Tensor]] | None = None

  # Which roles have any steps in them, given the two skills involved. A skill that
  # sizes a role at zero is not offering it.
  def _roles(self) -> list[str]:
    source_spec = self.plan[self.pool[self.source]]
    target_spec = self.plan[self.pool[self.target]]
    lengths = {
      CLOSING: source_spec.closing,
      OVERRUN: source_spec.overrun,
      OPENING: target_spec.opening,
    }
    return [role for role, steps in lengths.items() if steps > 0]

  @property
  def skill_id(self) -> int:
    return self.target if self.role == OPENING else self.source

  @property
  def spec(self):
    return self.plan[self.pool[self.skill_id]]

  @property
  def steps(self) -> int:
    return self.spec.length(self.role)

  @property
  def cut(self) -> int:
    """Where the hand-over falls, walked by the couple index across the source range."""
    source_spec = self.plan[self.pool[self.source]]
    low, _ = source_spec.interrupt_range
    return low + self.index % source_spec.cut_choices

  def _active_label(self, env_idx: int) -> str:
    del env_idx
    name = self.pool[self.skill_id].name
    if self.role == OPENING:
      return f"opening window of '{name}', from its own reset"
    if self.role == CLOSING:
      return f"closing window of '{name}', cut {self.cut} steps in"
    return f"overrun of '{name}', still driving past the cut at {self.cut}"

  @override
  def setup(self) -> None:
    # Built before the base viewer's tab group so it sits at the top of the panel.
    with self._server.gui.add_folder("Window"):
      labels = [ROLE_LABELS[role] for role in self._roles()]
      which = self._server.gui.add_button_group("Show", options=labels)

      @which.on_click
      def _(event) -> None:
        role = next(r for r, lbl in ROLE_LABELS.items() if lbl == event.target.value)
        self.request_action("CUSTOM", {"type": "window", "role": role})

      nav = self._server.gui.add_button_group("Couple", options=["Previous", "Next"])

      @nav.on_click
      def _(event) -> None:
        step = -1 if event.target.value == "Previous" else 1
        self.request_action("CUSTOM", {"type": "window", "step": step})

      if len(self.others) > 1:
        names = [self.pool[i].name for i in self.others]
        picker = self._server.gui.add_dropdown("Coming from", options=names)

        @picker.on_update
        def _(_) -> None:
          chosen = self.others[names.index(picker.value)]
          self.request_action("CUSTOM", {"type": "window", "source": chosen})

      self._info = self._server.gui.add_html("")

    super().setup()
    self._restart()

  @override
  def _handle_custom_action(self, action: ViewerAction, payload: Optional[Any]) -> bool:
    if isinstance(payload, dict) and payload.get("type") == "window":
      if "role" in payload:
        self.role = payload["role"]
      if "source" in payload:
        self.source = payload["source"]
        if self.role not in self._roles():
          self.role = self._roles()[0]
      if "step" in payload:
        choices = self.plan[self.pool[self.source]].cut_choices
        self.index = (self.index + payload["step"]) % choices
      self._rng = None  # A different couple, so draw a fresh episode for it
      self._restart()
      return True
    return super()._handle_custom_action(action, payload)

  def _run_up(self) -> int:
    """Steps to take before the window starts, which are run through in silence.

    A closing window is the *end* of a longer run, so everything before it is taken here
    rather than shown. An overrun window starts at the cut, so the whole run up to the
    cut is skipped. An opening window starts at the reset and skips nothing.
    """
    if self.role == CLOSING:
      return self.cut - self.steps
    if self.role == OVERRUN:
      return self.cut
    return 0

  def _restart(self) -> None:
    """Begin the window again: reset, run up to it in silence, then start showing it."""
    env = self.env.unwrapped
    with self._sim_lock:
      if self._rng is None:
        env.seed(self.index)
        self._rng = (torch.random.get_rng_state(), torch.cuda.get_rng_state_all())
      else:
        torch.random.set_rng_state(self._rng[0])
        torch.cuda.set_rng_state_all(self._rng[1])
      obs, _ = env.reset()
      # The same start state the collectors use, so what is watched here is what is
      # trained on rather than whatever the shared arena's reset happens to produce.
      obs = start_skill(env, self.spec, obs)
      self.pool.reset(self._active)
      skill = self.pool[self.skill_id]
      for _ in range(self._run_up()):
        obs, _, _, _, _ = env.step(skill.act(obs, self._active))
      self._obs = obs
    self._frame = 0
    self._step_count = 0

  @override
  def _execute_step(self) -> bool:
    """Drive the window's skill for one step, restarting when the window is over."""
    if self._frame >= self.steps:
      self._restart()
      return True
    env = self.env.unwrapped
    skill = self.pool[self.skill_id]
    with self._sim_lock, torch.no_grad():
      assert self._obs is not None
      self._obs, _, _, _, _ = env.step(skill.act(self._obs, self._active))
    self._frame += 1
    self._step_count += 1
    self._stats_steps += 1
    return True

  @override
  def reset_environment(self) -> None:
    self._restart()

  @override
  def sync_env_to_viewer(self) -> None:
    super().sync_env_to_viewer()
    source_spec = self.plan[self.pool[self.source]]
    low, high = source_spec.interrupt_range
    name = self.pool[self.skill_id].name
    if self.role == OPENING:
      where = "<strong>From:</strong> its own reset"
    elif self.role == CLOSING:
      where = f"<strong>Hand-over at:</strong> step {self.cut} of its run"
    else:
      where = f"<strong>Would have kept going from:</strong> step {self.cut}"
    self._info.content = (
      '<div style="font-size:0.85em;line-height:1.35;padding:0 1em 0.5em 1em;">'
      f"<strong>Couple:</strong> {self.index % source_spec.cut_choices + 1}"
      f" / {source_spec.cut_choices} (cuts {low} to {high})<br/>"
      f"<strong>Skill:</strong> '{name}', {self.role}<br/>"
      f"{where}<br/>"
      f"<strong>Frame:</strong> {min(self._frame + 1, self.steps)} / {self.steps}"
      "</div>"
    )


def run_inspect(
  cfg: InspectConfig,
  task_id: str,
  build_pool: PoolFactory,
  plan: WindowPlan,
) -> None:
  """Open the viewer for one experiment.

  `task_id` names the registered task whose (play-mode) env is the arena, `build_pool`
  builds that experiment's skills in it, and `plan` is the same `WindowPlan` its
  training uses.
  """
  import mjlab.tasks  # noqa: F401  (populates the task registry)
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  pool = build_pool(env, device)

  if not 0 <= cfg.target < len(pool):
    raise ValueError(
      f"--target {cfg.target} is not a skill id; the pool has "
      f"{len(pool)}: {[s.name for s in pool.skills]}."
    )

  viewer_env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task_id).clip_actions)
  WindowViewer(viewer_env, pool, cfg.target, plan).run()
  viewer_env.close()
