"""Skill-bridging: compose analytic skills with transition bridges, across systems.

LAYOUT
------
Shared skeleton (system-agnostic, this directory):
  bridge.py      shared vocabulary + ``Bridge`` interface + ``InstantBridge`` (basic).
  controller.py  ``Controller`` interface + ``FSMController`` (basic FSM).
  play.py        run any analytic ``policy(model, data) -> ctrl`` in a viser viewer.
Per system (``config/<system>/``):
  the MuJoCo model, its skills, and its dynamics. e.g. ``config/diffdrive/``.
Per experiment (``experiments/<system>/``):
  wire a robot + a controller + a bridge + a scenario, then run it. e.g.
  ``experiments/diffdrive/naive_switch.py``. Custom bridges/controllers live here.

ROLES
-----
Skill       ``state -> command``. One analytic policy (e.g. a target twist).
Bridge      what to command *between* skills: ``reset(from, to)`` then
            ``step(state) -> (command, done)``. ``InstantBridge`` = no transition.
Controller  owns the skills + one bridge; tracks the active skill and, on
            ``switch_to(name)``, runs the bridge until ``done`` then activates the
            target. ``FSMController`` is the basic one. It does NOT decide when or
            to which skill to switch -- that is the experiment's job.
Experiment  picks a system + controller + bridge + scenario (when to switch),
            builds a ``policy(model, data) -> ctrl``, and runs it via ``play.run``.

PER-STEP FLOW (one control tick)
--------------------------------
    state  = state_from_mjdata(data)         # read reduced state from MjData
    # the experiment's supervisor may call controller.switch_to(name) here
    action = controller.step(state)          # active skill, OR bridge mid-switch
    data.ctrl[:] = action ; mj_step          # a skill IS a policy: state -> action

SWITCHING, IN DETAIL (inside ``FSMController.step``)
---------------------------------------------------
    not switching:  return active_skill(state)
    switching:      command, done = bridge.step(state)
                    if done: active = target      # bridge delivered us into the
                                                  # next skill; hand control over
                    return command
"""
