# Bridging policies — problem statement and study examples


## 1. The core problem

A robot has a set of **skills** — individual behaviours, each realized by a policy that maps the robot's state (and possibly extra, command-conditioned observations) to actions. Examples of skills: walking, jumping, crouching, running forward, kicking.

A **higher-level controller** runs one skill and, at an arbitrary moment, may **interrupt** it and command a **different** skill. The problem is that the first skill can leave the robot in a *dynamic* state that the second skill was never built to start from — formally, **outside the second skill's initiation set**. A switch attempted from there fails or behaves badly.

**Goal:** design a system that *bridges* two skills — that carries the robot from wherever the interrupt leaves it into a state where the next skill can take over cleanly.

The canonical motivating scenario (the eventual target, a humanoid): the robot is running forward and is suddenly commanded to run backward. It cannot switch instantly — it must first shed the forward momentum (small steps, tilt the body back) before the "run backward" skill can engage. No single skill contains that momentum-shedding behaviour; the bridge must provide it.

## 2. Vocabulary for the problem

- **Skill / policy:** a behaviour, `state → action`. Whether written by hand (analytic) or learned makes no difference to the bridging question.
- **Tube:** running a skill from its family of starting conditions traces a bundle of similar trajectories through a chosen **state representation** — a "tube." Executing a skill is moving along its tube.
- **Initiation set:** the set of states from which a skill can start *successfully and conveniently*. A switch is really the act of getting the robot **into the next skill's initiation set / tube**.
- **The bridge:** a short-lived controller, engaged only around a switch, whose job is to take the robot from the interrupt state into the next tube.

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

TODO: We need to ultimate this example and make it "make sense": formalize it to show the effects, the problem we want to solve. 

**World:** a differential drive robot in maze. We are expected to have a controller that guides the robot through the maze (we don't have to solve it). The controller has to send signals to the robot, the robot has to realize the signals in a brief time window. We have to show that by trivially stitching the control skills one after the other we are not able to navigate the maze as the robot crashes or accumulates errors with no means to recover.

**Skills:** we will need at least two control policies to have something to switch to.

**The problem it exposes:** we will have to properly define what we want to show.

**The recovery aspect:** we should somehow demonstrate the robot entering a window at the start of the next policy (e.g. if the next locomotion policy has something to do with velocity, and it starts with zero velocity ramping up, we should demonstrate the controller starting executing the next policy with nonzero velocity as if it was skipping the "initial states coming from it"). I don't know how to demonstrate this.

**A framing insight we reached:** for the previous point (selecting an entry in a window, and recovering) to be demonstrable, the **next tube must be spatially extended** — there must be many distinguishable early states to choose among. An in-place turn is essentially a *point* (it doesn't go anywhere), so the rich, selectable window is not on the turn. We should be careful about what skills to choose to make the bridging and the window mechanism worth exploring. 

### Ant

TODO
