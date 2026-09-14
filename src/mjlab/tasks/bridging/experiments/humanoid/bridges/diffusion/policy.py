"""Driving the robot with the model: a plan is drawn, a few of its actions are executed,
another plan is drawn.

The model produces a window, not an action, so something has to decide how much of one
window to believe before drawing the next. Executing the whole horizon open loop wastes the
only correction available, and redrawing every tick costs a full denoising ladder per
control step. `ControlCfg.replan_every` is that trade, and it is the number to move first
when a crossing looks jittery or the loop looks slow.

Nothing here reads the observation. The bridge needs the robot's state, the target and the
ticks left, and the first is available in full from the simulator while the last two live on
the command term. A policy that had to reconstruct a target from a flattened observation
vector would be doing work the environment already did.

Deploying through the runner

Every script in this repo loads a policy the same way: `load_runner_cls(task)` for the
class, then `load` and `get_inference_policy`. `DiffusionRunner` answers that interface
without being a reinforcement learning runner at all, so `uv run play`,
`bridges/evaluate.py` and the transition scripts drive this architecture with no change to
any of them. What it cannot do is `learn`: the model is trained offline by train.py, and
`uv run train` on this task says so rather than starting something that would not work.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  ROBOT,
  body_pos_b,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.data import (
  Layout,
  Normalizer,
  encode,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.diffusion import (
  Process,
  ProcessCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.guidance import aim
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.model import (
  Denoiser,
  ModelCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp.commands import (
  BridgeCommand,
)

COMMAND = "bridge"


@dataclass
class ControlCfg:
  """How the trained model is turned into a controller. All of it is inference only."""

  replan_every: int = 5
  """Control ticks executed from one plan, 0.1 s at 50 Hz. One is a fresh plan per tick
  and the horizon's own length is fully open loop."""

  sample_steps: int = 12
  """Denoising steps per plan. Overrides what the checkpoint trained with, since this is
  the knob that trades arrival against how long a control step takes."""

  strength: float = 1.0
  """How hard the deadline column is pulled onto the target. One is inpainting."""

  hold: float = 0.5
  """How hard the columns past the deadline are held there."""


DEFAULT_CONTROL = ControlCfg()
"""What DiffusionRunner uses. A script wanting different sampling replaces this before the
runner is built, since the runner interface every loader uses has no room to pass it."""


@dataclass
class Bundle:
  """A trained model, ready to sample. What a checkpoint holds."""

  process: Process
  layout: Layout
  normalizer: Normalizer
  history: int
  horizon: int

  @property
  def columns(self) -> int:
    return self.history + self.horizon


def load_bundle(path: Path, device: str | torch.device) -> Bundle:
  """Read a checkpoint written by train.py."""
  blob = torch.load(path, map_location=device, weights_only=False)
  layout = Layout(**blob["layout"])
  model_cfg = ModelCfg(**blob["model"])
  process_cfg = ProcessCfg(**blob["process"])
  columns = blob["history"] + blob["horizon"]

  denoiser = Denoiser(layout.width, columns, model_cfg)
  process = Process(denoiser, process_cfg)
  # The averaged weights, not the last ones. A diffusion model's running average is
  # reliably better than the iterate it came from, and sampling from the raw iterate is a
  # standard way to conclude a model did not train
  process.denoiser.load_state_dict(blob["ema"])
  process.to(device)
  process.eval()
  process.requires_grad_(False)

  return Bundle(
    process=process,
    layout=layout,
    normalizer=Normalizer(**blob["normalizer"]).to(device),
    history=int(blob["history"]),
    horizon=int(blob["horizon"]),
  )


class Crossing:
  """One trained model driving one vectorized environment across its windows.

  Holds the recent past, because the model is conditioned on a short history rather than on
  a single frame: one instantaneous state does not say which phase of a stride the robot is
  in or which foot just landed, and two situations needing opposite strategies look
  identical without it.
  """

  def __init__(
    self, env: ManagerBasedRlEnv, bundle: Bundle, cfg: ControlCfg | None = None
  ) -> None:
    self.env = env
    self.bundle = bundle
    self.cfg = cfg or DEFAULT_CONTROL
    self.bundle.process.cfg = replace(
      bundle.process.cfg, sample_steps=self.cfg.sample_steps
    )

    term = env.command_manager.get_term(COMMAND)
    assert isinstance(term, BridgeCommand), (
      f"The diffusion bridge drives a '{COMMAND}' window and this environment has "
      f"{type(term).__name__}."
    )
    self.command = term
    self.robot = env.scene[ROBOT]

    count = env.num_envs
    device = env.device
    joints = bundle.layout.num_joints
    width = 13 + 2 * joints

    # Preallocated and only ever written through, never reassigned. The evaluation harness
    # calls a policy inside inference mode, and a buffer replaced in there becomes an
    # inference tensor that the next call outside it cannot write to
    self.states = torch.zeros(count, bundle.history, width, device=device)
    self.actions = torch.zeros(count, bundle.history, joints, device=device)
    self.bodies = (
      torch.zeros(count, bundle.history, bundle.layout.num_bodies, 3, device=device)
      if bundle.layout.num_bodies
      else None
    )
    self.plan = torch.zeros(count, max(self.cfg.replan_every, 1), joints, device=device)
    self.cursor = self.plan.shape[1]
    """How far into the current plan. Starting at its end forces one to be drawn."""

  def observe(self) -> None:
    """Push this control tick onto the history.

    The action recorded beside a state is the one that produced it, which is the convention
    the corpus uses and the reason the action manager is read before the environment steps:
    what it holds now is what was applied to reach the state being recorded now.
    """
    state = self.command.state_now()
    bodies = None if self.bodies is None else body_pos_b(self.robot)
    action = self.env.action_manager.action

    fresh = self.command.step == 0
    if bool(fresh.any()):
      rows = fresh.nonzero().flatten()
      self.states[rows] = state[rows].unsqueeze(1)
      self.actions[rows] = action[rows].unsqueeze(1)
      if self.bodies is not None and bodies is not None:
        self.bodies[rows] = bodies[rows].unsqueeze(1)
      # A window that just opened has no past to plan from, so the plan in hand belongs to
      # the previous one and has to go
      self.cursor = self.plan.shape[1]

    keep = ~fresh
    if bool(keep.any()):
      rows = keep.nonzero().flatten()
      self.states[rows, :-1] = self.states[rows, 1:].clone()
      self.states[rows, -1] = state[rows]
      self.actions[rows, :-1] = self.actions[rows, 1:].clone()
      self.actions[rows, -1] = action[rows]
      if self.bodies is not None and bodies is not None:
        self.bodies[rows, :-1] = self.bodies[rows, 1:].clone()
        self.bodies[rows, -1] = bodies[rows]

  def draw(self) -> None:
    """Sample one window per environment and keep the actions this plan will execute."""
    bundle = self.bundle
    anchor = self.states[:, -1]
    history = bundle.normalizer(
      encode(bundle.layout, self.states, anchor, self.actions, self.bodies)
    )

    deadline = (self.command.window_steps - self.command.step).clamp(min=0)
    cost = aim(
      layout=bundle.layout,
      normalizer=bundle.normalizer,
      target=self.command.target,
      anchor=anchor,
      deadline=deadline,
      columns=bundle.columns,
      history=bundle.history,
      strength=self.cfg.strength,
      hold=self.cfg.hold,
    )

    window = bundle.process.sample(
      (self.states.shape[0], bundle.columns, bundle.layout.width),
      history,
      cost,
      self.env.device,
    )
    plan = bundle.normalizer.invert(window)
    # Column history is the action that drives the anchor tick to the next one, so the
    # plan starts there and not at the anchor's own column
    start = bundle.history
    self.plan[:] = plan[:, start : start + self.plan.shape[1], bundle.layout.action]

  def act(self) -> torch.Tensor:
    """The joint targets for this control step. (N, J)."""
    self.observe()
    if self.cursor >= self.plan.shape[1]:
      self.draw()
      self.cursor = 0
    action = self.plan[:, self.cursor].clone()
    self.cursor += 1
    return action


class DiffusionRunner:
  """The loader interface, over a model that was never trained by reinforcement learning.

  Constructed and called exactly like an rsl_rl runner, which is what makes every script
  that drives a bridge work with this architecture unchanged.
  """

  def __init__(
    self,
    env: Any,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    del train_cfg, log_dir
    self.env = env
    self.device = device
    self.crossing: Crossing | None = None

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    del load_cfg, strict
    bundle = load_bundle(Path(path), map_location or self.device)
    self.crossing = Crossing(self.env.unwrapped, bundle, DEFAULT_CONTROL)
    return {}

  def get_inference_policy(self, device: str | None = None):
    del device
    if self.crossing is None:
      raise ValueError("No checkpoint loaded. Call load first.")
    crossing = self.crossing

    def policy(obs):
      del obs
      return crossing.act()

    return policy

  def add_git_repo_to_log(self, path: str) -> None:
    """No-op. Kept so `uv run train` reaches its own refusal rather than an attribute
    error two lines earlier."""
    del path

  def learn(self, *args, **kwargs) -> None:
    del args, kwargs
    raise SystemExit(
      "The diffusion bridge is trained offline on the corpus, not in the simulator. Run "
      "`uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion"
      ".train` instead."
    )
