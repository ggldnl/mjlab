"""A coloured, poseable copy of the robot, for the two viewers in this architecture.

Both the corpus viewer and the evaluation viewer draw the same thing: a robot standing at
a pose that came from an array rather than from physics, in a colour that means something.
This is that, once.

Colours are baked into the meshes when they are uploaded, in viser as in mjviser, so
there is no way to recolour a ghost after the fact. What there is instead is one whole
ghost per colour, with only the ones that apply visible. Showing and hiding is instant;
recolouring would mean re-uploading every mesh.
"""

from __future__ import annotations

import mujoco
import numpy as np
import trimesh
import viser

# What each colour means, wherever a ghost is drawn.
CONTEXT_COLOR = (60, 200, 90)
"""Motion the model is given."""

MASKED_COLOR = (225, 55, 45)
"""Motion it has to invent: the hole, as the body actually crossed it."""

MODEL_COLOR = (70, 150, 255)
"""What the model produced there."""

BASELINE_COLOR = (170, 170, 175)
"""What straight interpolation produced there, which is the thing to beat."""

GHOST_OPACITY = 0.9


def quat_from_mat(mat: np.ndarray) -> np.ndarray:
  """MuJoCo's row-major 3x3 to a wxyz quaternion."""
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=np.float64).reshape(9))
  return quat


def visual_meshes(model: mujoco.MjModel) -> list[tuple[int, trimesh.Trimesh]]:
  """The geoms worth drawing, with their meshes, built once and shared by every ghost.

  Collision geoms are skipped: they are the crude convex stand-ins the solver uses, and
  drawing them shows a robot made of boxes.
  """
  from mjlab.viewer.viser import create_primitive_mesh, mujoco_mesh_to_trimesh

  out: list[tuple[int, trimesh.Trimesh]] = []
  for gid in range(model.ngeom):
    if model.geom_bodyid[gid] == 0:
      continue  # World, i.e. the floor
    if model.geom_contype[gid] or model.geom_conaffinity[gid]:
      continue
    if model.geom_rgba[gid, 3] == 0.0:
      continue
    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
      out.append((gid, mujoco_mesh_to_trimesh(model, gid)))
    else:
      out.append((gid, create_primitive_mesh(model, gid)))
  return out


class Ghost:
  """One robot in one colour, posed from an MjData."""

  def __init__(
    self,
    server: viser.ViserServer,
    meshes: list[tuple[int, trimesh.Trimesh]],
    name: str,
    color: tuple[int, int, int],
    visible: bool = True,
    opacity: float = GHOST_OPACITY,
  ) -> None:
    self.server = server
    self.name = name
    self.geom_ids = [gid for gid, _ in meshes]
    self._visible = visible
    self.handles = [
      server.scene.add_mesh_simple(
        f"/ghost/{name}/geom_{gid}",
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.uint32),
        color=color,
        opacity=opacity,
        visible=visible,
      )
      for gid, mesh in meshes
    ]

  @property
  def visible(self) -> bool:
    return self._visible

  @visible.setter
  def visible(self, value: bool) -> None:
    if value == self._visible:
      return
    for handle in self.handles:
      handle.visible = value
    self._visible = value

  def pose(self, data: mujoco.MjData) -> None:
    """Move to wherever `data` currently has the robot.

    Call this before making a hidden ghost visible, never after. A hidden ghost still
    carries the pose it had when it was last shown, so revealing it first puts it
    somewhere stale for a frame, which the eye reads as a teleport.
    """
    for handle, gid in zip(self.handles, self.geom_ids, strict=True):
      handle.position = data.geom_xpos[gid]
      handle.wxyz = quat_from_mat(data.geom_xmat[gid])
