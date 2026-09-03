"""Put a state on a robot and measure it against the floor.

Answers one question: how far off the ground is the lowest part of the robot in this
state. Grounded states read about zero, airborne ones read tens of centimetres, and a
state written into the floor reads negative.

That is what separates a spot the bridge could deliver the robot to from one it could
not. A mid-flight state is a real place the skill goes through, and no bridge can put
the robot there: getting a body onto a specific ballistic arc is not something a
position controller does. Nothing about which skill it is enters into this.

No physics. Forward kinematics on a compiled model, then geometry. The state was
recorded under physics already, and stepping the solver would integrate it a second
time.

Used by build.py to filter and by view.py to draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.dataset import (
  ROOT_STATE_DIM,
)


@dataclass
class Slot:
  """Where one copy's numbers live in a shared qpos."""

  free: int
  """qpos address of its free joint. Position is [free, free + 3), orientation is
  [free + 3, free + 7)."""
  joints: np.ndarray
  """(J,) qpos addresses, in model joint order, which is the order the dataset's joint
  block was recorded in."""


def build(count: int = 1, floor: bool = True) -> tuple[mujoco.MjModel, list[Slot]]:
  """A model holding `count` copies of the G1, and where to write each pose."""
  world = mujoco.MjSpec()
  if floor:
    world.worldbody.add_geom(
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      size=[20.0, 20.0, 0.1],
      rgba=[0.3, 0.3, 0.32, 1.0],
    )
  prefixes = [f"e{i}/" for i in range(count)]
  for prefix in prefixes:
    # A fresh spec per copy. Attaching one twice asks MuJoCo to adopt the same bodies
    # into two places in the tree
    world.attach(get_spec(), prefix=prefix, frame=world.worldbody.add_frame())
  model = world.compile()
  return model, [slot(model, prefix) for prefix in prefixes]


def slot(model: mujoco.MjModel, prefix: str) -> Slot:
  """Resolve one copy's qpos addresses, in model joint order.

  Order and not name, because a dataset does not record the joint names it was written
  against. Every copy is the same spec attached again, so every copy gives the same
  order, and it is the order the rows were recorded in.
  """
  free: int | None = None
  joints: list[int] = []
  for joint in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
    if not name.startswith(prefix):
      continue
    address = int(model.jnt_qposadr[joint])
    if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
      free = address
    else:
      joints.append(address)
  if free is None:
    raise SystemExit(f"No free joint under '{prefix}'.")
  return Slot(free=free, joints=np.asarray(joints, dtype=np.int64))


def show(qpos: np.ndarray, where: Slot, state: np.ndarray, shift: np.ndarray) -> None:
  """Write one state into a shared qpos, moved by `shift`."""
  qpos[where.free : where.free + 3] = state[0:3] + shift
  qpos[where.free + 3 : where.free + 7] = state[3:7]
  qpos[where.joints] = state[ROOT_STATE_DIM : ROOT_STATE_DIM + where.joints.size]


def collision_geoms(model: mujoco.MjModel, prefix: str = "") -> np.ndarray:
  """Geoms the solver actually collides with, for one copy.

  Every geom, not a list of feet. A push leans on its hands and a fall lands on a knee,
  so the lowest part of the robot is not always a foot, and asking which body it is
  would be a per-skill question.
  """
  found = []
  for geom in range(model.ngeom):
    if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
      continue
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom])
    if prefix and not (body or "").startswith(prefix):
      continue
    found.append(geom)
  return np.asarray(found, dtype=np.int64)


def _drop(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> float:
  """How far one geom reaches below its own centre, in world z.

  Exact for the primitives the G1 collides with. A mesh falls back to its bounding
  sphere, which reads low rather than high, so a mesh can only make a state look more
  grounded than it is and never less.
  """
  size = model.geom_size[geom]
  rotation = data.geom_xmat[geom].reshape(3, 3)
  down = np.abs(rotation[2])  # world z component of each local axis
  kind = model.geom_type[geom]
  if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
    return float(size[0])
  if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
    return float(size[0] + down[2] * size[1])
  if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
    return float(down[2] * size[1] + np.sqrt(max(1.0 - down[2] ** 2, 0.0)) * size[0])
  if kind == mujoco.mjtGeom.mjGEOM_BOX:
    return float(down @ size)
  if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
    return float(np.linalg.norm(down * size))
  return float(model.geom_rbound[geom])


class Ground:
  """Measures states against the floor. Compiles one robot, then reuse it."""

  def __init__(self) -> None:
    self.model, slots = build(count=1, floor=False)
    self.slot = slots[0]
    self.data = mujoco.MjData(self.model)
    self.geoms = collision_geoms(self.model)

  @property
  def num_joints(self) -> int:
    return int(self.slot.joints.size)

  def clearance(self, state: np.ndarray) -> float:
    """Lowest point of the robot in this state, in metres above the floor.

    About zero standing, positive in the air, negative through the floor.
    """
    show(self.data.qpos, self.slot, state, np.zeros(3))
    mujoco.mj_kinematics(self.model, self.data)
    return min(
      float(self.data.geom_xpos[geom][2]) - _drop(self.model, self.data, geom)
      for geom in self.geoms
    )
