"""
Height command for legged robots. Implements a uniform-random target body height
command that integrates with the mjlab command manager.

Usage in envs:

    from mjlab.tasks.velocity.mdp import UniformHeightCommandCfg

    cfg.commands["target_height"] = UniformHeightCommandCfg(
        entity_name="robot",
        resampling_time_range=(4.0, 8.0),
        ranges=UniformHeightCommandCfg.Ranges(height=(0.02, 0.04)),
    )

The command tensor has shape (num_envs, 1) and contains the target height
in metres. Reward functions and observations can read it via:

    command = env.command_manager.get_command("target_height")  # (N, 1)
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mjlab.managers.command_manager import CommandTermCfg, CommandTerm

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class UniformHeightCommand(CommandTerm):
    """Samples a scalar target body height uniformly from a range.

    The command is held fixed for ``resampling_time_range`` seconds, then
    resampled — matching the lifecycle of UniformVelocityCommand so that
    both commands resample on the same rhythm when ranges are equal.
    """

    cfg: UniformHeightCommandCfg

    def __init__(self, cfg: UniformHeightCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # command buffer: (num_envs, 1) — target height in metres
        self._command = torch.zeros(env.num_envs, 1, device=env.device)

    @property
    def command(self) -> torch.Tensor:
        """Target height tensor, shape (num_envs, 1)."""
        return self._command

    def reset(self, env_ids: torch.Tensor | None = None) -> dict:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        self._resample(env_ids)
        return {}

    def compute(self, dt: float):
        """Resample commands for environments whose timer has expired."""
        self.time_left -= dt
        expired = (self.time_left <= 0.0).nonzero(as_tuple=False).squeeze(-1)
        if expired.numel() > 0:
            self._resample(expired)

    def _resample(self, env_ids: torch.Tensor):
        lo, hi = self.cfg.ranges.height
        n = env_ids.numel()
        self._command[env_ids, 0] = (
                torch.rand(n, device=self._env.device) * (hi - lo) + lo
        )
        # reset the countdown timer for these envs
        t_lo, t_hi = self.cfg.resampling_time_range
        self.time_left[env_ids] = (
                torch.rand(n, device=self._env.device) * (t_hi - t_lo) + t_lo
        )

    # Logging / visualisation

    def _set_debug_vis_impl(self, debug_vis: bool):
        # No geometric visualization needed for a scalar command.
        pass

    def _debug_vis_callback(self, event):
        pass


@dataclass
class UniformHeightCommandCfg(CommandTermCfg):
    """Configuration for :class:`UniformHeightCommand`.

    Parameters
    ----------
    entity_name:
        Name of the robot entity in the scene (e.g. ``"robot"``).
    resampling_time_range:
        ``(min_s, max_s)`` — how long each sampled height is held before
        a new one is drawn.
    ranges:
        The uniform interval ``[height_min, height_max]`` in metres.
    """

    class_type: type = UniformHeightCommand

    @dataclass
    class Ranges:
        height: tuple[float, float] = (0.02, 0.04)

    entity_name: str = "robot"
    resampling_time_range: tuple[float, float] = (4.0, 8.0)
    ranges: Ranges = field(default_factory=Ranges)
    debug_vis: bool = False

    def build(self, env: "ManagerBasedRlEnv") -> UniformHeightCommand:
        return self.class_type(cfg=self, env=env)