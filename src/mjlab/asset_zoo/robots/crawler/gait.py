"""Open-loop gait for the crawler quadruped (no reinforcement learning).

This module has two consumers:

* ``GaitController`` -- the algorithm. A diagonal-trot foot-trajectory central
  pattern generator plus per-leg damped-least-squares inverse kinematics. It
  turns a commanded body twist ``(vx, vy, wz)`` and a target base height into
  joint-position targets, with no feedback (pure feed-forward). The legs are
  mounted at 45 degrees in an X-layout, so a given coxa rotation moves each foot
  in a different world direction; driving in Cartesian foot space + IK avoids
  having to hand-pick per-leg joint signs. This is the *teacher* used by the
  behavioral-cloning distillation script (``scripts/distill_crawler_gait.py``).

* ``CrawlerGait`` -- a headless diagnostic that rolls the controller out in a
  standalone CPU sim and reports achieved-vs-commanded base velocity, tip angle,
  foot contacts and actuator saturation::

      uv run python -m mjlab.asset_zoo.robots.crawler.gait --sweep
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.crawler.actuators import (
  JOINT_NAMES,
  LEG_PHASE_OFFSETS,
  NOMINAL_BASE_HEIGHT,
)
from mjlab.asset_zoo.robots.crawler.collisions import FOOT_SITE_NAMES
from mjlab.asset_zoo.robots.crawler.crawler_constants import get_crawler_robot_cfg
from mjlab.entity.entity import Entity

# Per-leg joint names, ordered coxa -> femur -> tibia.
_LEG_JOINTS = [
  (JOINT_NAMES[3 * i], JOINT_NAMES[3 * i + 1], JOINT_NAMES[3 * i + 2]) for i in range(4)
]


@dataclass
class GaitParams:
  """Tunable parameters of the open-loop gait."""

  frequency: float = 2.5
  """Stride frequency (Hz). One full swing+stance cycle per 1/frequency seconds."""
  duty: float = 0.6
  """Fraction of the cycle a foot spends in stance (>=0.5 keeps >=2 feet down)."""
  swing_height: float = 0.012
  """Peak foot clearance during swing (m)."""
  base_height: float = NOMINAL_BASE_HEIGHT
  """Target base height (m). Feet are pushed further below the base to raise it."""
  phase_offsets: tuple[float, float, float, float] = tuple(LEG_PHASE_OFFSETS.tolist())  # type: ignore[assignment]
  """Per-leg phase (rad). Default [0, pi, 0, pi] is a diagonal trot."""
  ik_damping: float = 1e-3
  """Damped-least-squares regularizer for the per-leg IK (m^2)."""
  ik_iters: int = 6
  """Gauss-Newton iterations per control tick."""
  max_stride: float = 0.05
  """Clamp on the half-stride length (m) to keep feet inside the workspace."""


class GaitController:
  """Feed-forward trot controller: body twist -> joint-position targets.

  Owns a small CPU MuJoCo model used purely for kinematics/Jacobians in the IK.
  The base is held fixed at the neutral stance, so the joint targets it returns
  are a function of the gait phase and command only (no state feedback).
  """

  def __init__(self, params: GaitParams | None = None) -> None:
    self.params = params or GaitParams()

    entity = Entity(get_crawler_robot_cfg())
    spec = entity.spec
    # A flat floor so the standalone diagnostic sim has ground to push on. The
    # IK ignores it.
    spec.worldbody.add_geom(
      name="floor",
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      size=[0, 0, 0.01],
      rgba=[0.7, 0.7, 0.7, 1.0],
      contype=1,
      conaffinity=1,
    )
    spec.worldbody.add_light(name="top", pos=[0, 0, 1.0], dir=[0, 0, -1])
    self.model = spec.compile()
    self._scratch = mujoco.MjData(self.model)

    self.base_bid = self.model.body("base").id
    self.foot_sids = [self.model.site(n).id for n in FOOT_SITE_NAMES]
    self._ctrl_of_joint: dict[str, int] = {}
    for a in range(self.model.nu):
      jname = self.model.joint(self.model.actuator(a).trnid[0]).name
      self._ctrl_of_joint[jname] = a
    self._dof_of_joint = {n: int(self.model.joint(n).dofadr[0]) for n in JOINT_NAMES}
    self._qadr_of_joint = {n: int(self.model.joint(n).qposadr[0]) for n in JOINT_NAMES}

    # Fixed base pose + nominal foot positions, captured at the neutral stance.
    key = self.model.key("init_state")
    self._scratch.qpos[:] = key.qpos
    self._scratch.qvel[:] = 0.0
    mujoco.mj_forward(self.model, self._scratch)
    self._key_qpos = key.qpos.copy()
    self._base_pos = self._scratch.xpos[self.base_bid].copy()
    self._base_rot = self._scratch.xmat[self.base_bid].reshape(3, 3).copy()
    self._nominal_foot_base = np.empty((4, 3))
    for i, sid in enumerate(self.foot_sids):
      self._nominal_foot_base[i] = self._base_rot.T @ (
        self._scratch.site_xpos[sid] - self._base_pos
      )

    # Persistent IK seed (joint angles in JOINT_NAMES order) for smoothness.
    self._seed = np.array([self._key_qpos[self._qadr_of_joint[n]] for n in JOINT_NAMES])

  @property
  def ctrl_of_joint(self) -> dict[str, int]:
    return self._ctrl_of_joint

  @property
  def nominal_foot_base(self) -> np.ndarray:
    """Foot site positions in the base frame at the neutral stance, shape (4, 3)."""
    return self._nominal_foot_base.copy()

  def reset_seed(self) -> None:
    """Reset the IK warm-start seed to the neutral stance."""
    self._seed = np.array([self._key_qpos[self._qadr_of_joint[n]] for n in JOINT_NAMES])

  def phase_at(self, t: float) -> float:
    """Gait phase in [0, 1) at simulation time ``t``."""
    return (t * self.params.frequency) % 1.0

  def _foot_targets_base(
    self, phase: float, vx: float, vy: float, wz: float, base_height: float
  ) -> np.ndarray:
    """Reference foot positions in the base frame for the current phase."""
    p = self.params
    targets = self._nominal_foot_base.copy()
    # Raise the base by pushing every foot further below it.
    targets[:, 2] -= base_height - NOMINAL_BASE_HEIGHT
    for i in range(4):
      r = self._nominal_foot_base[i, :2]
      # Forward reach at touchdown: during stance the planted foot sweeps from
      # +half (front) to -half (back), pushing the body along +(v + wz x r).
      reach = np.array([vx - wz * r[1], vy + wz * r[0]])
      half = reach * (p.duty / (2.0 * p.frequency))
      norm = float(np.linalg.norm(half))
      if norm > p.max_stride:
        half *= p.max_stride / norm

      s = (phase + p.phase_offsets[i] / (2.0 * math.pi)) % 1.0
      if s < p.duty:  # Stance: planted, sweeping front to back.
        prog = s / p.duty
        targets[i, :2] += half * (1.0 - 2.0 * prog)
      else:  # Swing: lifted, returning back to front.
        prog = (s - p.duty) / (1.0 - p.duty)
        targets[i, :2] += half * (2.0 * prog - 1.0)
        targets[i, 2] += p.swing_height * math.sin(math.pi * prog)
    return targets

  def _solve_leg_ik(
    self, leg: int, target_base: np.ndarray, seed: np.ndarray
  ) -> np.ndarray:
    """DLS IK for one leg about the fixed base. Returns coxa/femur/tibia (rad)."""
    p = self.params
    names = _LEG_JOINTS[leg]
    dofs = [self._dof_of_joint[n] for n in names]
    qadrs = [self._qadr_of_joint[n] for n in names]
    sid = self.foot_sids[leg]

    scratch = self._scratch
    scratch.qpos[:] = self._key_qpos
    scratch.qvel[:] = 0.0
    target_world = self._base_pos + self._base_rot @ target_base

    jacp = np.zeros((3, self.model.nv))
    q = seed.copy()
    for _ in range(p.ik_iters):
      for a, qi in zip(qadrs, q, strict=True):
        scratch.qpos[a] = qi
      mujoco.mj_kinematics(self.model, scratch)
      mujoco.mj_comPos(self.model, scratch)
      err = target_world - scratch.site_xpos[sid]
      if float(np.linalg.norm(err)) < 1e-5:
        break
      mujoco.mj_jacSite(self.model, scratch, jacp, None, sid)
      j = jacp[:, dofs]  # 3x3
      jtj = j.T @ j + p.ik_damping * np.eye(3)
      q = q + np.linalg.solve(jtj, j.T @ err)
    for k, n in enumerate(names):
      lo, hi = self.model.joint(n).range
      q[k] = float(np.clip(q[k], lo, hi))
    return q

  def joint_targets(
    self,
    phase: float,
    vx: float,
    vy: float,
    wz: float,
    base_height: float | None = None,
  ) -> np.ndarray:
    """Joint-position targets (rad), in ``JOINT_NAMES`` order, for this phase."""
    if base_height is None:
      base_height = self.params.base_height
    targets = self._foot_targets_base(phase, vx, vy, wz, base_height)
    q = self._seed.copy()
    for leg in range(4):
      q[3 * leg : 3 * leg + 3] = self._solve_leg_ik(
        leg, targets[leg], q[3 * leg : 3 * leg + 3]
      )
    self._seed = q
    return q


class CrawlerGait:
  """Headless diagnostic: roll the gait out in a CPU sim and measure tracking."""

  def __init__(self, params: GaitParams | None = None) -> None:
    self.controller = GaitController(params)
    self.model = self.controller.model
    self.data = mujoco.MjData(self.model)
    self.reset()

  def reset(self) -> None:
    mujoco.mj_resetData(self.model, self.data)
    key = self.model.key("init_state")
    self.data.qpos[:] = key.qpos
    self.data.ctrl[:] = key.ctrl
    mujoco.mj_forward(self.model, self.data)
    self.controller._seed = np.array(
      [key.qpos[self.controller._qadr_of_joint[n]] for n in JOINT_NAMES]
    )

  def _base_rot(self) -> np.ndarray:
    return self.data.xmat[self.controller.base_bid].reshape(3, 3).copy()

  def _control(self, t: float, vx: float, vy: float, wz: float) -> None:
    q = self.controller.joint_targets(self.controller.phase_at(t), vx, vy, wz)
    for n, qi in zip(JOINT_NAMES, q, strict=True):
      self.data.ctrl[self.controller.ctrl_of_joint[n]] = qi

  def run(
    self,
    vx: float = 0.08,
    vy: float = 0.0,
    wz: float = 0.0,
    duration: float = 6.0,
    settle: float = 1.0,
    control_dt: float = 0.02,
    verbose: bool = True,
  ) -> dict[str, float]:
    """Roll out the gait headless and return achieved-vs-commanded metrics."""
    self.reset()
    dt = self.model.opt.timestep
    substeps = max(1, round(control_dt / dt))
    n_total = int(duration / dt)
    n_settle = int(settle / dt)
    forcerange = self.model.actuator_forcerange.copy()

    samples: list[np.ndarray] = []
    base_z: list[float] = []
    tilt: list[float] = []
    sat_frac: list[float] = []
    contacts: list[int] = []
    start_xy = self.data.qpos[:2].copy()

    for step in range(n_total):
      if step % substeps == 0:
        self._control(step * dt, vx, vy, wz)
      mujoco.mj_step(self.model, self.data)
      if step < n_settle:
        continue
      rot = self._base_rot()
      v_base = rot.T @ self.data.qvel[:3]
      samples.append(np.array([v_base[0], v_base[1], self.data.qvel[5]]))
      base_z.append(float(self.data.qpos[2]))
      tilt.append(math.degrees(math.acos(np.clip(rot[2, 2], -1.0, 1.0))))
      foot_force = np.abs(self.data.actuator_force)
      lim = np.maximum(forcerange[:, 1], 1e-9)
      sat_frac.append(float(np.mean(foot_force >= 0.98 * lim)))
      contacts.append(self._count_foot_contacts())

    mean_v = np.array(samples).mean(axis=0)
    disp = (self.data.qpos[:2].copy() - start_xy) / (duration - settle)
    tipped = bool(np.max(tilt) > 70.0 or np.min(base_z) < 0.5 * NOMINAL_BASE_HEIGHT)
    metrics = {
      "cmd_vx": vx,
      "cmd_vy": vy,
      "cmd_wz": wz,
      "mean_vx_base": float(mean_v[0]),
      "mean_vy_base": float(mean_v[1]),
      "mean_wz": float(mean_v[2]),
      "net_speed_world": float(np.linalg.norm(disp)),
      "track_ratio": float(mean_v[0] / vx) if abs(vx) > 1e-6 else float("nan"),
      "mean_base_z": float(np.mean(base_z)),
      "min_base_z": float(np.min(base_z)),
      "max_tilt_deg": float(np.max(tilt)),
      "mean_foot_contacts": float(np.mean(contacts)),
      "force_saturation_frac": float(np.mean(sat_frac)),
      "tipped": tipped,
    }
    if verbose:
      self._print_metrics(metrics)
    return metrics

  def _count_foot_contacts(self) -> int:
    floor_gid = self.model.geom("floor").id
    foot_gids = {self.model.geom(f"leg_{i}_foot_collision").id for i in range(1, 5)}
    n = 0
    for c in range(self.data.ncon):
      con = self.data.contact[c]
      pair = {con.geom1, con.geom2}
      if floor_gid in pair and pair & foot_gids:
        n += 1
    return n

  @staticmethod
  def _print_metrics(m: dict[str, float]) -> None:
    print(
      f"\ncmd=(vx={m['cmd_vx']:+.3f}, vy={m['cmd_vy']:+.3f}, wz={m['cmd_wz']:+.3f})"
    )
    print(
      f"  achieved base vel: vx={m['mean_vx_base']:+.4f} m/s "
      f"vy={m['mean_vy_base']:+.4f} m/s  wz={m['mean_wz']:+.4f} rad/s"
    )
    print(f"  net world speed:   {m['net_speed_world']:.4f} m/s")
    if not math.isnan(m["track_ratio"]):
      print(f"  vx track ratio:    {m['track_ratio']:.2f}  (1.0 = perfect)")
    print(
      f"  base height:       mean={m['mean_base_z'] * 1000:.1f} mm "
      f"min={m['min_base_z'] * 1000:.1f} mm  (nominal {NOMINAL_BASE_HEIGHT * 1000:.1f})"
    )
    print(f"  max tilt:          {m['max_tilt_deg']:.1f} deg")
    print(f"  mean foot contacts:{m['mean_foot_contacts']:.2f} / 4")
    print(f"  force saturation:  {m['force_saturation_frac'] * 100:.1f}% of joints")
    print(f"  tipped over:       {m['tipped']}")


def _main() -> None:
  import tyro

  @dataclass
  class Args:
    vx: float = 0.08
    vy: float = 0.0
    wz: float = 0.0
    frequency: float = 2.5
    duty: float = 0.6
    swing_height: float = 0.012
    duration: float = 6.0
    sweep: bool = False

  args = tyro.cli(Args)
  params = GaitParams(
    frequency=args.frequency, duty=args.duty, swing_height=args.swing_height
  )
  gait = CrawlerGait(params)

  if args.sweep:
    print("Speed sweep (forward command vs achieved):")
    for vx in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15):
      gait.run(vx=vx, duration=args.duration, verbose=True)
    return

  gait.run(vx=args.vx, vy=args.vy, wz=args.wz, duration=args.duration)


if __name__ == "__main__":
  _main()
