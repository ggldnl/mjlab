"""The individual G1 skills, one sub-package each.

    walk                commanded velocity locomotion, mjlab's flat G1 task under another name
    run                 the same task, curriculum retuned for forward speed
    jump                one ASAP clip tracked end to end, a 1.54 m forward jump. The reference
                        stays at inference, so it is the same jump every time
    jump_continuous     goal conditioned jump. Tracks clips to learn it, reads only the goal to
                        run it. Both phases are one task.
    passing             a standing shove of a football at a commanded launch velocity
    kick                a proper kick to the ball. A human kick shapes the reward while it learns
                        and is annealed away, leaving a policy paid only for where the ball goes.
    push                walk into a heavy crate and drive it at a commanded velocity
    martial             one task per martial arts motion, each tracking a crop of a LAFAN1 fight
                        performance from a standstill.

Run

    uv run train Mjlab-G1-Walk --env.scene.num-envs 4096

Swap in Run, Jump, Pass, ... for the others.
Each sub-package docstring says what its task needs: a skill that touches an object in the
scene carries different observations from one that does not, so their checkpoints are not
interchangeable.
"""

from mjlab.tasks.bridging.experiments.humanoid.skills.jump import JUMP_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous import (
  JUMP_CONTINUOUS_TASK_ID,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.kick import KICK_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.martial import MARTIAL_TASK_IDS
from mjlab.tasks.bridging.experiments.humanoid.skills.passing import PASS_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.push import PUSH_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.run import RUN_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.walk import WALK_TASK_ID

SKILLS: dict[str, str] = {
  "walk": WALK_TASK_ID,
  "run": RUN_TASK_ID,
  "jump": JUMP_TASK_ID,
  # "jump_continuous": JUMP_CONTINUOUS_TASK_ID,
  "pass": PASS_TASK_ID,
  "kick": KICK_TASK_ID,
  "push": PUSH_TASK_ID,
  **MARTIAL_TASK_IDS,
}
"""
Skill name to task id. Every one logs to g1_<name>. Add a skill by adding a line. 
It needs a trained checkpoint under that log directory.

jump_continuous was just cool to have but we won't use it in the bridging projected.
We will use the plain trajectory tracking version.
"""
