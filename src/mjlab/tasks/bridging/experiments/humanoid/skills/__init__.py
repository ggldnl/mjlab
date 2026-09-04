"""The individual G1 skills, one sub-package each.

    walk         commanded velocity locomotion, mjlab's flat G1 task under another name
    run          the same task, curriculum retuned for forward speed
    jump         goal conditioned jump, motion tracking against retargeted clips
    passing      a standing shove of a football at a commanded launch velocity
    kick         WIP
    push         walk into a heavy crate and drive it at a commanded velocity
    front_kick   a kick out of a standstill, tracking a crop of a LAFAN1 fight performance
    punch_combo  the same recipe on a different crop, borrowing front_kick's code

Run

    uv run train Mjlab-G1-Walk --env.scene.num-envs 4096

Swap in Run, Jump, Pass, ... for the others.
Each sub-package docstring says what its task needs: a skill that touches an object in the
scene carries different observations from one that does not, so their checkpoints are not
interchangeable.
"""

# TODO make sure the clustering is egocentric: a dynamic state is the same from the point
#   of the robot regardless of the direction it is facing

from mjlab.tasks.bridging.experiments.humanoid.skills.front_kick import FRONT_KICK_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.jump import JUMP_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.passing import PASS_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.punch_combo import PUNCH_COMBO_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.push import PUSH_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.run import RUN_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.walk import WALK_TASK_ID


SKILLS: dict[str, str] = {
  "walk": WALK_TASK_ID,
  "run": RUN_TASK_ID,
  "jump": JUMP_TASK_ID,
  "front_kick": FRONT_KICK_TASK_ID,
  "pass": PASS_TASK_ID,
  "push": PUSH_TASK_ID,
  "punch_combo": PUNCH_COMBO_TASK_ID,
}
"""
Skill name to task id. Every one logs to g1_<name>, so nothing else has to be said.
Add a skill by adding a line. It needs a trained checkpoint under that log directory.
"""