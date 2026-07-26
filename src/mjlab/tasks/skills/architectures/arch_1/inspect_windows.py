"""Look at what arch_1 is actually training on, before believing any of its metrics.

Collects the two window sets phase 1 learns from and shows them side by side: on the
left the terminal window of a source skill (the last steps before the bridge is
dropped in, ending at a harvested interrupt state), on the right the initiation window
of the target skill (what the discriminator is told "real" looks like). Previous/next
buttons walk either buffer, so the pairing can be swept rather than sampled once.

It also checks the interrupt states round-trip: restore one into the env, recompute the
observation, and compare it against the observation recorded at capture time. If those
disagree, the states being trained from are not the states that were harvested and
nothing downstream means what it says.

    uv run python -m mjlab.tasks.skills.architectures.arch_1.inspect_windows \\
        --experiment diffdrive --target 1

Collect once, look many times (collection needs the simulator, viewing does not):

    uv run python -m mjlab.tasks.skills.architectures.arch_1.inspect_windows \\
        --experiment diffdrive --target 1 --save windows/
    uv run python -m mjlab.tasks.skills.architectures.arch_1.inspect_windows \\
        --load windows/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.tasks.skills.architectures.arch_1.windows import (
  ManagerState,
  WindowSet,
  collect_interrupts,
  collect_target_windows,
  verify_restore,
)

# The experiments this can be pointed at, and the pieces it needs from each: the task
# whose env is the arena, the scene entity states are harvested from, and the pool
# factory. Adding an experiment is one line.
EXPERIMENTS: dict[str, tuple[str, str, str]] = {
  "diffdrive": (
    "mjlab.tasks.skills.experiments.diffdrive",
    "DRIVE_TASK_ID",
    "ENTITY_NAME",
  ),
  "cartpole": (
    "mjlab.tasks.skills.experiments.cartpole",
    "SPINUP_TASK_ID",
    "ENTITY_NAME",
  ),
}


@dataclass(frozen=True)
class InspectConfig:
  # Which experiment to collect from. Ignored when --load is given
  experiment: str = "diffdrive"

  # The skill the bridge is aimed at: its initiation window is the right-hand panel,
  # and interrupt states are harvested from every *other* skill
  target: int = 1

  # How many steps a window spans, on both sides
  window_steps: int = 64

  # How many source-skill steps a bridge may be dropped in after
  interrupt_max_steps: int = 64

  # How many windows to keep on each side
  num_target_windows: int = 256
  interrupt_capacity: int = 512

  num_envs: int = 128
  device: str | None = None

  # Write the collected windows here as .npz, so they can be viewed again without
  # rebuilding the simulator
  save: str | None = None

  # Read previously saved windows from here instead of collecting
  load: str | None = None

  # Collect (and verify) without opening a window
  show: bool = True


def _collect(cfg: InspectConfig) -> tuple[WindowSet, WindowSet]:
  import importlib

  import mjlab.tasks  # noqa: F401  (populates the task registry)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  if cfg.experiment not in EXPERIMENTS:
    raise ValueError(
      f"Unknown experiment '{cfg.experiment}'; known: {sorted(EXPERIMENTS)}."
    )
  module_name, task_attr, entity_attr = EXPERIMENTS[cfg.experiment]
  module = importlib.import_module(module_name)
  task_id = getattr(module, task_attr)
  entity_name = getattr(module, entity_attr)

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  pool = module.build_pool(env, device)
  skill_names = tuple(s.name for s in pool.skills)

  print(f"[collect] target skill '{skill_names[cfg.target]}' initiation window...")
  target_windows = collect_target_windows(
    env,
    pool[cfg.target],
    cfg.target,
    skill_names,
    cfg.window_steps,
    cfg.num_target_windows,
  )
  print(f"[collect] {len(target_windows)} target windows")

  print(f"[collect] {len(pool) - 1} source skill(s), terminal window + state...")
  interrupts = collect_interrupts(
    env,
    pool,
    entity_name,
    exclude=cfg.target,
    max_steps=cfg.interrupt_max_steps,
    num_rollouts=max(1, cfg.interrupt_capacity // cfg.num_envs),
    capacity=cfg.interrupt_capacity,
    window_steps=cfg.window_steps,
  )
  print(f"[collect] {len(interrupts)} interrupt states")

  # The one check that says whether any of this data means what it claims to.
  manager_state = ManagerState(env)
  max_error, within = verify_restore(
    env, env.scene[entity_name], interrupts, manager_state
  )
  verdict = "OK" if within > 0.99 else "MISMATCH"
  print(
    f"[verify] restore round-trip: max obs error {max_error:.3e}, "
    f"{within:.1%} of states exact -> {verdict}"
  )
  if within <= 0.99:
    print(
      "[verify] the harvested state does not fully determine the env: something "
      "per-env is not being carried across the restore."
    )

  env.close()
  return interrupts.windows, target_windows


def _save(source: WindowSet, target: WindowSet, path: Path) -> None:
  path.mkdir(parents=True, exist_ok=True)
  source.to_npz(path / "source_windows.npz")
  target.to_npz(path / "target_windows.npz")
  print(f"[save] wrote windows to {path}")


def _load(path: Path) -> tuple[WindowSet, WindowSet]:
  return (
    WindowSet.from_npz(path / "source_windows.npz"),
    WindowSet.from_npz(path / "target_windows.npz"),
  )


class WindowViewer:
  """Two stacked panels of channels, one buffer per column, with prev/next per side.

  Rows are the observation terms (labelled from the env's observation manager) plus
  the action. Both columns share a row's y limits, so a channel that looks different
  on the two sides really is different.
  """

  def __init__(self, source: WindowSet, target: WindowSet) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    self.plt = plt
    self.sets = (source, target)
    self.index = [0, 0]
    self.rows = [*source.layout.slices(), ("action", None)]

    n = len(self.rows)
    self.fig, self.axes = plt.subplots(
      n, 2, figsize=(13, 1.6 * n + 1.6), sharex="col", squeeze=False
    )
    self.fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.10, hspace=0.25)

    self.buttons = []
    for column, x0 in enumerate((0.13, 0.60)):
      for label, delta in (("< prev", -1), ("next >", 1)):
        left = x0 + (0.11 if delta > 0 else 0.0)
        axis = self.fig.add_axes((left, 0.015, 0.10, 0.04))
        button = Button(axis, label)
        button.on_clicked(self._stepper(column, delta))
        self.buttons.append(button)

    self.fig.canvas.mpl_connect("key_press_event", self._on_key)
    self._draw()

  def _stepper(self, column: int, delta: int):
    def handler(_event) -> None:
      self._advance(column, delta)

    return handler

  def _on_key(self, event) -> None:
    # Left/right walk the source buffer, up/down the target one
    moves = {"left": (0, -1), "right": (0, 1), "down": (1, -1), "up": (1, 1)}
    if event.key in moves:
      self._advance(*moves[event.key])

  def _advance(self, column: int, delta: int) -> None:
    size = len(self.sets[column])
    self.index[column] = (self.index[column] + delta) % size
    self._draw()

  def _draw(self) -> None:
    for axes_row in self.axes:
      for axis in axes_row:
        axis.clear()

    for column, windows in enumerate(self.sets):
      index = self.index[column]
      obs = windows.obs[index].cpu().numpy()
      action = windows.action[index].cpu().numpy()
      valid = windows.valid[index].cpu().numpy()
      steps = len(valid)
      # The hand-off is at step 0: the source window is the run-up to it (negative
      # steps), the target window is the skill's own first steps from its reset.
      x = np.arange(-steps, 0) if column == 0 else np.arange(steps)

      for row, (name, term_slice) in enumerate(self.rows):
        axis = self.axes[row][column]
        series = action if term_slice is None else obs[:, term_slice]
        for channel in range(series.shape[1]):
          axis.plot(x, series[:, channel], linewidth=1.2, label=f"[{channel}]")
        if not valid.all():
          # Left-padding: the source skill had not been running this long yet
          first = int(np.argmax(valid)) if valid.any() else steps
          axis.axvspan(x[0], x[max(first - 1, 0)], color="0.85", zorder=0)
        axis.axhline(0.0, color="0.7", linewidth=0.6, zorder=0)
        axis.set_ylabel(name, fontsize=8)
        axis.tick_params(labelsize=7)
        if series.shape[1] > 1:
          axis.legend(fontsize=6, ncol=series.shape[1], loc="upper left", frameon=False)

      self.axes[-1][column].set_xlabel("step relative to hand-off", fontsize=8)
      self.axes[0][column].set_title(self._title(windows, index), fontsize=9)

    # One y scale per row across both columns, so the two sides are comparable
    for axes_row in self.axes:
      low = min(axis.get_ylim()[0] for axis in axes_row)
      high = max(axis.get_ylim()[1] for axis in axes_row)
      for axis in axes_row:
        axis.set_ylim(low, high)

    self.fig.suptitle(
      f"arch_1 windows   |   source {self.index[0] + 1}/{len(self.sets[0])}"
      f"   target {self.index[1] + 1}/{len(self.sets[1])}"
      "   |   left/right: source, up/down: target",
      fontsize=10,
    )
    self.fig.canvas.draw_idle()

  @staticmethod
  def _title(windows: WindowSet, index: int) -> str:
    name = windows.name_of(index)
    offset = int(windows.offset[index])
    if windows.kind == "source terminal":
      return f"{windows.kind}: '{name}', dropped in after {offset} steps"
    return f"{windows.kind}: '{name}', from its own reset"

  def run(self) -> None:
    self.plt.show()


def run_inspect(cfg: InspectConfig) -> None:
  if cfg.load is not None:
    source, target = _load(Path(cfg.load))
    print(f"[load] {len(source)} source windows, {len(target)} target windows")
  else:
    source, target = _collect(cfg)
    if cfg.save is not None:
      _save(source, target, Path(cfg.save))

  if cfg.show:
    WindowViewer(source, target).run()


if __name__ == "__main__":
  run_inspect(tyro.cli(InspectConfig))
