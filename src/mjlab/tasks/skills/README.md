# Bridging policies — problem statement and study examples


## 1. The core problem

A robot has a set of **skills** — individual behaviours, each realized by a policy that maps the robot's state (and possibly extra, command-conditioned observations) to actions. Examples of skills: walking, jumping, crouching, running forward, kicking.

A **higher-level controller** runs one skill and, at an arbitrary moment, may **interrupt** it and command a **different** skill. The problem is that the first skill can leave the robot in a *dynamic* state that the second skill was never built to start from — formally, **outside the second skill's initiation set**. A switch attempted from there fails or behaves badly.

**Goal:** design a system that *bridges* two skills — that carries the robot from wherever the interrupt leaves it into a state where the next skill can take over cleanly.

## 2. Vocabulary for the problem

- **Skill / policy:** a behavior, `state -> action`. Whether written by hand (analytic) or learned makes no difference to the bridging question.
- **Tube:** running a skill from its family of starting conditions traces a bundle of similar trajectories through a chosen **state representation** — a "tube." Executing a skill is moving along its tube. A switch (with the helo of the bridge) is really the act of getting the robot **into the next skill's tube**, not necessarily from the initiation set.
- **Initiation set:** the set of states from which a skill can start *successfully and conveniently*.
- **Controller:** a high level controller is responsible for deciding when to switch and to what skill to switch. It can be implemented using FSM, neural networks, a planning algorithm, ... We will assume it as given. It runs a skill and whenever it decides to switch, it uses a bridging policy to instantly stop the current skill and drive the robot safely to execute the next skill.
- **The bridge:** a short-lived bridging policy, engaged only around a switch, whose job is to take the robot from the interrupt state into the next tube.

A deliberate choice (advised by the professor and kept throughout): work in an explicit **state representation** (positions, velocities, body pose, …), **not** a learned latent space — so tubes and initiation sets stay interpretable.

## 3. What we want the bridge to accomplish — two pillars

1. **Don't wait for completion.** We do not let a skill run to its natural end before starting the next one. The bridge transitions from *whatever* the current state is into a state at the **beginning of the next skill's tube**.
2. **Enter the tube at the best point in a window — and recover.** The robot need not join the next tube at its nominal start. There is a **window of early states** along the next tube, and the bridge should enter at the **most effective** one (what "effective" means is to be defined). In particular, when the interrupt comes too late for a clean entry, the bridge must still find a feasible, non-nominal entry — a **recovery**.

(Inspiration: a paper that performs switching by matching against a *window of early frames* of the next behaviour using a velocity-aware **state feature** — i.e. it picks where to join the next behaviour rather than always at frame zero. We want a bridge that does this kind of selection, but synthesizes the transition rather than retrieving it.)

## 4. Constraints and guidance shaping the study

- **Use a state representation**, not a latent space.
- **Start small:** validate the idea on a low-dimensional, interpretable system before the humanoid.
- A skill **need not terminate "correctly"** — it might end in a state from which the next skill is **impossible to start** (outside its initiation set).
- It is the **dynamic state (velocity), not just the configuration**, that decides initiation-set membership. A position that looks fine can be unusable because the robot is moving too fast.

## 5. Examples

### Cartpole

Two skills: **swing-up** (pumps the pole from hanging to upright — a *deliverer*; it cannot hold the pole) and **balance** (holds the pole upright — a *stabilizer*; it was only ever shown already-upright states, so it only knows a bounded region near the top).

The problem it exposes: switching from swing-up to balance is fine only if swing-up has delivered the pole into balance's initiation set. Switch while the pole is still mid-swing and balance **cannot catch it — the pole falls**. The failure is **intrinsic and unrecoverable**, and *when* the switch happens is decisive.

### Differential drive

```
[
    [0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 5, 5, 5],
    [0, 0, 0, 0, 0, 0, 4, 0, 0, 6],
    [0, 0, 0, 0, 0, 0, 4, 0, 0, 6],
    [0, 0, 0, 3, 3, 3, 4, 0, 0, 6],
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 6],
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 7],
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 7],
    [1, 1, 1, 2, 0, 0, 0, 0, 0, 7],
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 7],
]
```

We have a differential drive robot in a corridor world: a grid world with corridors, each leading to the next one. The idea is that the robot has a skill fo each corridor, that tells it how to run through them. We need to stress that: 
- we can't transition safely from one skill to the next one
- each individual skill knows how to run through the corridor if starting from the first cell, it does not know how to start from a different cell (exactly what happens when we transition from one corridor to the next one)
- one cool thing that I was thinking about was the fact the robot doesn't have a skill to turn (corridors are straight) so it would be a nice behavior that the bridge could pick up, but this is not mandatory, just an idea we could explore

#### How the bridge is built and trained

We take the skills and the grid as they are, with no representative junction. The
whole bridge is derived from rollouts of the skills, so the same recipe carries over to
other robots whose skills we are only given, not allowed to reshape. Two things come out
of those rollouts (see `bridge/rollouts.py`):

- the window of each skill's tube: the early states it actually passes through just after
  it starts, which are the candidate states the bridge may join.
- the interrupt states: where the previous skill leaves the robot at the junction corner,
  spread out by jittering that skill's start inside its initiation set.

A switch then becomes a goal-reaching problem. At the moment of the switch we pick the
goal as the window state closest to the interrupt state, where closeness is velocity
aware (position, heading, and speed all count), and the bridge has to drive there. The
bridge is a single policy trained with PPO across all the junctions at once
(`Mjlab-Bridge-Diffdrive`): each episode resets the robot to a harvested interrupt state,
the policy commands a body twist through a wheel servo with bounded torque, and it is
rewarded for sliding into the goal without leaving the corridors. Because the robot
arrives carrying momentum it cannot cancel at once, the policy has to shed speed and turn,
which is exactly the behavior no single skill provides.

```sh
uv run train Mjlab-Bridge-Diffdrive --env.scene.num-envs 4096
uv run python -m mjlab.tasks.skills.experiments.diffdrive.experiment \
  --bridge learned --checkpoint <logdir>/<name>.onnx
```

The first command trains the bridge and exports the policy to ONNX on every checkpoint.
The second runs it in the corridor world: the controller follows each corridor's skill and,
at every junction, hands over to the learned bridge (`bridge/policy.py`) to carry the robot
into the next skill's tube.