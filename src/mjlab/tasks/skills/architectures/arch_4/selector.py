"""Where to aim: which moment of the skill being entered the bridge should arrive at.

This is the second half of arch_4 and eventually it should be learned. It is hardcoded
here, and that is a deliberate first step rather than a placeholder: with two skills in
the pool the good entry moments can be found by looking, and having them written down by
hand is what lets the bridge be tested at all. What replaces this later is a module that
finds them, not a different idea of what they are.

##
# What an entry moment is
##

Not a pose. A moment carries three things:

    frame     where in the skill's own rollout it sits, so the skill can be resumed there
    strip     the short stretch of motion from that point on, which is what the bridge is
              given as its target
    label     what it is, for the log

A skill offers as many of these as it has ways of being joined, and the number matters
more than anything else here. A skill with one entry moment has one entry, its descriptor
is a constant, and no amount of bridging will beat delivering the robot to that one state.

##
# The moments, and why these
##

**Jump.** Two. The first is the clip's own opening, a stand: this is what a naive
hand-off gets, and it is the baseline. The second is the bottom of the crouch, found from
the clip rather than typed in, as the lowest the root gets before the feet leave the
ground. Entering at the crouch is the interesting one, because a robot arriving at
walking speed can plant and sink directly instead of stopping, standing up, settling, and
only then crouching.

There is an honest limit to how much this buys, and it is worth being clear about it. Every
ASAP jump clip opens with the subject standing still, so the crouch in them is a
*stationary* crouch. Arriving there at 1.5 m/s hands the jump policy a reference that says
"launch vertically from a standstill", and the momentum has nowhere to go. What entering at
the crouch actually saves is the stand-up-and-settle, which is real and measurable and is
not the same thing as jumping while running. Getting that would need a jump skill whose
clip opens with a run-up.

**Walk.** Walking is a feedback policy tracking a commanded velocity, not a clip, so it
has no privileged frames at all: it will pick up from anywhere in its basin. Its entry
moments are therefore phases of its own gait, and the useful ones for landing are the
frames where a foot is about to be planted, since that is what a body coming down out of
a jump needs to do next.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Moment:
  """One place a skill can be joined."""

  skill: str
  label: str

  frame: int
  """Index into the skill's own rollout. What the skill is told, so it resumes here."""

  strip: torch.Tensor
  """The motion from `frame` onward, (steps, 13 + 2J), in the rollout's own coordinates.
  Rebased onto the robot when it is handed to the bridge, exactly as training rebased a
  corpus window onto the environment origin."""

  def __len__(self) -> int:
    return int(self.strip.shape[0])


def crouch_frame(root_height: np.ndarray, airborne: np.ndarray) -> int:
  """The bottom of the crouch: the lowest the root gets before the feet leave the ground.

  Measured off the clip rather than written down, so adding a jump clip does not mean
  finding its crouch by hand, and so the number cannot drift away from the motion it is
  supposed to describe.
  """
  flight = np.flatnonzero(airborne)
  if flight.size == 0:
    raise ValueError("This clip never leaves the ground; it is not a jump.")
  takeoff = int(flight[0])
  if takeoff < 2:
    raise ValueError("This clip is airborne from the start; there is no crouch in it.")
  return int(np.argmin(root_height[:takeoff]))


def moments_for_jump(
  states: torch.Tensor, root_height: np.ndarray, airborne: np.ndarray, steps: int
) -> list[Moment]:
  """The two ways into a standing jump: from its own start, and from its crouch."""
  bottom = crouch_frame(root_height, airborne)
  return [
    Moment("jump", "stand", 0, states[0 : 0 + steps].clone()),
    Moment("jump", "crouch", bottom, states[bottom : bottom + steps].clone()),
  ]


def moments_for_walk(
  states: torch.Tensor, foot_height: np.ndarray, steps: int, count: int = 4
) -> list[Moment]:
  """Phases of the gait where a foot is about to land.

  A walk has no beginning, so these are not entry points in the sense the jump's are;
  they are the phases worth arriving *in*. A body coming down out of a jump has to put a
  foot on the ground, and handing it a target strip that is about to do exactly that is
  the difference between landing into a stride and landing into a stumble.
  """
  # A foot is about to land where its height is falling and close to the floor. Local
  # minima of the lower foot are the plants; the frames just before them are the moments.
  usable = len(foot_height) - steps
  if usable <= 1:
    raise ValueError("The walk rollout is too short to cut a target strip out of.")
  inner = foot_height[1 : usable - 1]
  plants = np.flatnonzero(
    (inner <= foot_height[0 : usable - 2]) & (inner <= foot_height[2:usable])
  )
  if plants.size == 0:
    # A rollout with no clear plant is a rollout of something that is not walking, but a
    # spread of phases is still a usable answer and a visible one in the log.
    chosen = np.linspace(0, usable - 1, count).astype(int)
  else:
    chosen = plants[np.linspace(0, plants.size - 1, count).astype(int)] + 1
  return [
    Moment("walk", f"plant_{i}", int(f), states[int(f) : int(f) + steps].clone())
    for i, f in enumerate(dict.fromkeys(chosen.tolist()))
  ]


class Selector:
  """Which moment of the skill being entered the bridge should aim at.

  Hardcoded, and narrow on purpose: it is asked only about the skill being entered, and
  it answers with the moment that skill most wants to be joined at. It does not yet look
  at the robot, which is the whole thing that makes it a stub. A real selector would
  compare what the body is currently doing against every moment on offer and pick the one
  it can actually reach in the time available; this one already knows the answer because
  there are two skills and someone looked.
  """

  def __init__(self, moments: dict[str, list[Moment]], preferred: dict[str, str]):
    self.moments = moments
    self.preferred = preferred
    for skill, label in preferred.items():
      if not any(m.label == label for m in moments.get(skill, [])):
        raise ValueError(
          f"'{label}' is not a moment {skill} offers; it has "
          f"{[m.label for m in moments.get(skill, [])]}."
        )

  def choose(self, skill: str) -> Moment:
    """The moment to aim at when entering `skill`."""
    if skill not in self.moments:
      raise KeyError(
        f"No entry moments recorded for '{skill}'. A skill with none can only be entered "
        f"at its own beginning, which is what arch_0 already does."
      )
    label = self.preferred.get(skill)
    if label is None:
      return self.moments[skill][0]
    return next(m for m in self.moments[skill] if m.label == label)

  def describe(self) -> str:
    lines = []
    for skill, moments in sorted(self.moments.items()):
      chosen = self.preferred.get(skill)
      for m in moments:
        mark = " <- aimed at" if m.label == chosen else ""
        lines.append(f"  {skill:6s} {m.label:10s} frame {m.frame:4d}{mark}")
    return "\n".join(lines)
