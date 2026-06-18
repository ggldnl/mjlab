"""The diff-drive skill primitives as closed-form controllers (state -> torques).

A "skill" here is a plain function ``state -> action``: the analytic counterpart of
a trained policy. mjlab never cares whether that function is a neural network or a
control law -- the env produces an observation, *something* maps it to an action,
and the action manager applies it. The learned part of the thesis is the *bridge*
between these primitives, not the primitives themselves.

Three primitives, deliberately chosen so their initiation sets conflict:

* ``drive_straight`` -- cruise forward at a high speed, holding the heading. Leaves
  the robot carrying real translational momentum.
* ``turn_left`` / ``turn_right`` -- turn in place (zero forward speed, fixed yaw
  rate). Valid only from *low* speed: from a fast state they cannot spin cleanly,
  they skid in a wide curve.

The incompatibility is asymmetric: ``turn -> drive`` is easy (just accelerate),
but ``drive -> turn`` requires first bleeding off speed -- "non fa in tempo a
girare". That deceleration is exactly the behavior the (future) bridge must learn.
Reaching a goal pose is the high-level controller's job (sequencing primitives),
not a skill.

Run ``uv run python -m mjlab.tasks.diffdrive.skills`` to watch a primitive live in
a viser viewer (dropdown to pick it; sliders for cruise speed / turn rate).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mjlab.tasks.skills.config.diffdrive.dynamics import OMEGA, DiffDrive, V

Controller = Callable[[np.ndarray], np.ndarray]  # state (..., 5) -> torque (..., 2)


def _inner_pd(
  dd: DiffDrive,
  s: np.ndarray,
  v_des: np.ndarray,
  omega_des: np.ndarray,
  kp_v: float,
  kp_w: float,
) -> np.ndarray:
  """Force/moment PD on ``(v, omega)`` errors, mapped to wheel torques."""
  f_des = dd.mass * kp_v * (v_des - s[..., V])
  m_des = dd.inertia * kp_w * (omega_des - s[..., OMEGA])
  return dd.body_to_wheel(f_des, m_des)


def drive_skill(
  dd: DiffDrive,
  v_target: float,
  kp_v: float = 6.0,
  kp_w: float = 8.0,
) -> Controller:
  """Cruise forward at ``v_target``, damping yaw so the heading stays straight.

  Position and heading are left unregulated: the steady state is ``v = v_target``,
  ``omega = 0``, and the robot travels in whatever direction it currently faces.
  Interrupting it hands the next skill a state carrying real momentum.
  """

  def control(s: np.ndarray) -> np.ndarray:
    v_des = np.full(np.shape(s[..., V]), v_target)
    omega_des = np.zeros_like(np.asarray(s[..., OMEGA], dtype=float))
    return _inner_pd(dd, s, v_des, omega_des, kp_v, kp_w)

  return control


def turn_skill(
  dd: DiffDrive,
  yaw_rate: float,
  kp_v: float = 6.0,
  kp_w: float = 8.0,
) -> Controller:
  """Turn in place at ``yaw_rate`` (rad/s), commanding zero forward speed.

  Its intended initiation set is *low speed*: from there it spins cleanly about the
  center. From a fast state the same command produces a wide skidding curve rather
  than an in-place turn -- which is exactly why a ``drive -> turn`` switch must wait
  until the bridge has bled the speed off. Positive ``yaw_rate`` turns left.
  """

  def control(s: np.ndarray) -> np.ndarray:
    v_des = np.zeros_like(np.asarray(s[..., V], dtype=float))
    omega_des = np.full(np.shape(s[..., OMEGA]), yaw_rate)
    return _inner_pd(dd, s, v_des, omega_des, kp_v, kp_w)

  return control


@dataclass(frozen=True)
class Skill:
  """A named controller plus its operating point (for metrics / plotting)."""

  name: str
  controller: Controller
  target: np.ndarray  # [x, y, theta, v, omega]; NaN where the skill leaves it free.


def build_skills(
  dd: DiffDrive,
  v_cruise: float = 2.5,
  omega_turn: float = 2.0,
) -> dict[str, Skill]:
  """The three primitives: ``drive_straight``, ``turn_left``, ``turn_right``."""
  nan = float("nan")
  return {
    "drive_straight": Skill(
      "drive_straight",
      drive_skill(dd, v_cruise),
      np.array([nan, nan, nan, v_cruise, 0.0]),
    ),
    "turn_left": Skill(
      "turn_left",
      turn_skill(dd, omega_turn),
      np.array([nan, nan, nan, 0.0, omega_turn]),
    ),
    "turn_right": Skill(
      "turn_right",
      turn_skill(dd, -omega_turn),
      np.array([nan, nan, nan, 0.0, -omega_turn]),
    ),
  }


# MuJoCo bridge: extract the reduced skill state from a free-joint diff-drive.


def mjdata_to_state(data) -> np.ndarray:
  """Read ``[x, y, theta, v, omega]`` from a MuJoCo ``MjData`` for diffdrive.xml.

  qpos = ``[x, y, z, qw, qx, qy, qz, wheel_l, wheel_r]``; for planar motion only
  the yaw of the orientation and the in-plane velocities matter.
  """
  x, y = float(data.qpos[0]), float(data.qpos[1])
  qw, qx, qy, qz = (float(v) for v in data.qpos[3:7])
  theta = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
  vx, vy = float(data.qvel[0]), float(data.qvel[1])
  v = vx * np.cos(theta) + vy * np.sin(theta)  # forward (body-x) speed
  omega = float(data.qvel[5])  # yaw rate
  return np.array([x, y, theta, v, omega])


def main() -> None:
  """Spin up a viser viewer to watch one primitive at a time on the MuJoCo model.

  A dropdown selects the primitive; sliders set the cruise speed / turn rate. This
  steps the real ``diffdrive.xml`` plant, calling the analytic skill exactly where
  a trained ``policy(obs)`` would be called. (The goal marker in the model is for
  the high-level controller stage and is parked out of view here.)
  """
  # Local imports: keep the viewer's heavy deps out of the module that the env and
  # offline analysis import.
  import time
  from pathlib import Path

  import mujoco
  import viser
  from mjviser import ViserMujocoScene

  model = mujoco.MjModel.from_xml_path(str(Path(__file__).parent / "diffdrive.xml"))
  data = mujoco.MjData(model)
  dd = DiffDrive()

  server = viser.ViserServer()
  scene = ViserMujocoScene(server, model, num_envs=1)
  scene.create_scene_gui()

  names = ("drive_straight", "turn_left", "turn_right")
  skill_dd = server.gui.add_dropdown("Skill", names, initial_value="drive_straight")
  reset_btn = server.gui.add_button("Reset robot")
  v_cruise = server.gui.add_slider("drive: speed", 0.5, 4.0, 0.1, 2.5)
  omega_turn = server.gui.add_slider("turn: rate", 0.5, 4.0, 0.1, 2.0)

  def reset_robot() -> None:
    mujoco.mj_resetData(model, data)
    data.mocap_pos[0] = (0.0, 0.0, -5.0)  # park the goal marker out of view
    mujoco.mj_forward(model, data)

  reset_btn.on_click(lambda _: reset_robot())
  skill_dd.on_update(lambda _: reset_robot())
  reset_robot()

  decimation = 4
  while True:
    name = skill_dd.value
    if name == "drive_straight":
      controller = drive_skill(dd, v_cruise.value)
    else:
      sign = 1.0 if name == "turn_left" else -1.0
      controller = turn_skill(dd, sign * omega_turn.value)

    for _ in range(decimation):
      u = np.asarray(controller(mjdata_to_state(data)), dtype=float)
      data.ctrl[0], data.ctrl[1] = u[0], u[1]
      mujoco.mj_step(model, data)

    scene.update_from_mjdata(data)
    time.sleep(model.opt.timestep * decimation)


if __name__ == "__main__":
  main()
