"""Run a trained transition headlessly or watch it with Viser."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.tasks.bridging.baseline.evaluate import load_for_scene, trial
from mjlab.tasks.bridging.baseline.model import TransitionPolicy
from mjlab.tasks.bridging.baseline.scene import SceneCfg, TransitionScene


@dataclass(frozen=True)
class PlayCfg:
  checkpoint: Path
  scene: SceneCfg = field(default_factory=SceneCfg)
  distance: float = 0.4
  duration_s: float = 0.8
  seed: int = 10000
  viewer: Literal["viser", "none"] = "viser"
  automatic: bool = True


class Controller:
  """Request a transition at the next control tick, then hand over on time."""

  def __init__(
    self, scene: TransitionScene, policy: TransitionPolicy, cfg: PlayCfg
  ) -> None:
    self.scene, self.policy, self.cfg = scene, policy, cfg
    self.automatic = cfg.automatic
    self.distance, self.duration_s = cfg.distance, cfg.duration_s
    self.pending_entry: int | None = None
    self.reset()

  def reset(self) -> None:
    if self.pending_entry is not None:
      self.scene.select_entry(self.pending_entry)
      self.pending_entry = None
    self.scene.reset(self.cfg.seed)
    self.phase, self.ticks, self.budget = "source", 0, 0
    self.fire = False

  @property
  def status(self) -> str:
    active = (
      self.scene.source
      if self.phase == "source"
      else self.scene.target_skill
      if self.phase == "incoming skill"
      else self.phase
    )
    return (
      f"{active} | target {self.scene.run.distance:.2f} m | "
      f"{self.ticks * self.scene.env.step_dt:.2f} s in phase"
    )

  @torch.no_grad()
  def __call__(self, obs) -> torch.Tensor:
    del obs
    scene = self.scene
    if self.pending_entry is not None:
      self.reset()
    scene.failed |= scene.fallen()
    if self.phase == "incoming skill":
      scene.failed |= scene.target_failures.compute()
    if bool(scene.failed.any()):
      if self.phase != "failed":
        print(f"execution failed during {self.phase}")
      self.phase = "failed"
    if self.phase == "failed":
      return scene.env.action_manager.action.clone()

    if self.phase == "source":
      scene.hold_target()
      ready = scene.run.distance <= self.distance
      if scene.source != "walk":
        ready = scene.run._landed() and ready
      if self.fire or (self.automatic and ready):
        self.fire = False
        self.phase, self.ticks = "transition", 0
        self.budget = max(1, round(self.duration_s / scene.env.step_dt))
        scene.open_bridge(self.duration_s)
        print(
          f"transition requested: seed={self.cfg.seed}, entry={scene.cfg.entry}, "
          f"distance={scene.run.distance:.3f} m, budget={self.budget} ticks"
        )
      else:
        self.ticks += 1
        if (
          self.automatic
          and self.ticks * scene.env.step_dt >= scene.cfg.source_timeout_s
        ):
          self.phase = "failed"
          print("source trigger timed out")
        return scene.skill_action(scene.source)

    if self.phase == "transition":
      scene.hold_target()
      if self.ticks < self.budget:
        self.ticks += 1
        return self.policy(scene.observe())
      errors, within = scene.endpoint()
      print(
        f"handoff to {scene.target_skill}: endpoint errors={errors[0].tolist()}, "
        f"within box={bool(within[0])}"
      )
      scene.run.command.stop()
      self.phase, self.ticks = "incoming skill", 0

    self.ticks += 1
    return scene.skill_action(scene.target_skill)


def panel(server, controller: Controller, entries: tuple[int, ...]) -> None:
  with server.gui.add_folder("Transition policy"):
    server.gui.add_markdown(
      "Press **Start transition** while walking. Use **Reset** to repeat. "
      "The ghost marks the selected incoming entry."
    )
    button = server.gui.add_button("Start transition")
    button.on_click(lambda _: setattr(controller, "fire", True))
    automatic = server.gui.add_checkbox("Automatic", initial_value=controller.automatic)
    automatic.on_update(
      lambda _: setattr(controller, "automatic", bool(automatic.value))
    )
    distance = server.gui.add_slider(
      "Trigger distance, m",
      min=0.05,
      max=max(2.0, controller.distance),
      step=0.05,
      initial_value=controller.distance,
    )
    distance.on_update(lambda _: setattr(controller, "distance", float(distance.value)))
    low, high = controller.scene.run.command.cfg.duration_s_range
    duration = server.gui.add_slider(
      "Transition duration, s",
      min=low,
      max=high,
      step=0.05,
      initial_value=controller.duration_s,
    )
    duration.on_update(
      lambda _: setattr(controller, "duration_s", float(duration.value))
    )
    entry = server.gui.add_dropdown(
      "Incoming entry (restarts)",
      options=tuple(str(i) for i in entries),
      initial_value=str(controller.scene.cfg.entry),
    )
    entry.on_update(lambda _: setattr(controller, "pending_entry", int(entry.value)))


def play(cfg: PlayCfg) -> None:
  if any(
    not math.isfinite(value) or value <= 0 for value in (cfg.distance, cfg.duration_s)
  ):
    raise ValueError("distance and duration_s must be finite and positive")
  if cfg.viewer == "none" and not cfg.automatic:
    raise ValueError("Headless playback requires --automatic True")
  policy, scene_cfg, metadata = load_for_scene(cfg.checkpoint, cfg.scene)
  scene = TransitionScene(scene_cfg)
  server = None
  try:
    low, high = scene.run.command.cfg.duration_s_range
    if not low <= cfg.duration_s <= high:
      raise ValueError(f"duration_s must be between {low:g} and {high:g}")
    if cfg.viewer == "none":
      print(
        json.dumps(
          trial(scene, "transition", cfg.seed, cfg.distance, cfg.duration_s, policy)
        )
      )
      return
    import viser

    from mjlab.viewer import ViserPlayViewer

    controller = Controller(scene, policy, cfg)
    scene.env.manager_visualizers["transition_target"] = scene.run
    entries = tuple(metadata["config"].get("entries", [scene_cfg.entry]))
    server = viser.ViserServer(label=f"{scene_cfg.pair} transition policy")
    panel(server, controller, entries)
    viewer = ViserPlayViewer(
      scene.wrapped,
      controller,
      viser_server=server,
      info_provider=lambda _: controller.status,
      record_name=f"baseline_{scene_cfg.pair}",
    )
    if not cfg.automatic:
      viewer.pause()
    viewer.run()
  finally:
    if server is not None:
      server.stop()
    scene.close()


def main() -> None:
  play(tyro.cli(PlayCfg, config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
  main()
