# Bridging policies — problem statement and study examples


## 1. The core problem

A robot has a set of **skills** — individual behaviours, each realized by a policy that maps the robot's state (and possibly extra, command-conditioned observations) to actions. Examples of skills: walking, jumping, crouching, running forward, kicking.

A **higher-level controller** runs one skill and, at an arbitrary moment, may **interrupt** it and command a **different** skill. The problem is that the first skill can leave the robot in a *dynamic* state that the second skill was never built to start from — formally, **outside the second skill's initiation set**. A switch attempted from there fails or behaves badly.

**Goal:** design a system that *bridges* two skills — that carries the robot from wherever the interrupt leaves it into a state where the next skill can take over cleanly.

The canonical motivating scenario (the eventual target, a humanoid): the robot is running forward and is suddenly commanded to run backward. It cannot switch instantly — it must first shed the forward momentum (small steps, tilt the body back) before the "run backward" skill can engage. No single skill contains that momentum-shedding behaviour; the bridge must provide it.

## 2. Vocabulary for the problem

- **Skill / policy:** a behavior, `state -> action`. Whether written by hand (analytic) or learned makes no difference to the bridging question.
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

0. Purpose

This example must demonstrate, on a low-dimensional interpretable system, the two pillars of the bridging problem:

- Don't wait for completion — when a switch is commanded, the next skill starts from whatever dynamic state the previous skill left the robot in, not from a hand-tuned handoff state.
- Enter the next tube inside a window, and recover — the next skill is engaged at a non-nominal early state along its tube (with residual velocity carried in), and when a clean entry is impossible the bridge still finds a feasible one instead of failing.

---
1. The world

- A grid (your 10×10 array). Cell value 0 = wall (not free). Nonzero values 1..7 = the seven corridors. It can have a variable number of corridors.
- Constraint you set (point 4): 0 cells are solid. The robot lives only inside corridor cells; touching a 0 cell or the grid boundary is a collision = unrecoverable failure. This is what gives the example its hard edge — without it, overshoot is harmless and the problem goes soft.
- The array is a topological map (which regions are corridor vs. wall). The robot moves in a continuous space laid over it: each corridor is a straight channel of physical width W and finite length. W is a free parameter.
- Corridors are pairwise perpendicular and connected in some way (not necessarily at their start e.g. 1-2):

- C1 ->(left) C2 ->(right) C3 ->(left) C4 ->(right) C5 ->(right) C6

- east, north, east, north, east, south. Every junction is a 90° turn with a wall straight ahead if the robot fails to turn.

---
2. The robot model

For the problem to exist at all, the robot must have inertia — it cannot change its velocity instantly.

- State: s = (x, y, θ, v, ω) — position, heading, forward speed, turn rate. We will experiment later with this so keep it general for now.
- Dynamics: unicycle/differential-drive kinematics driven by bounded accelerations: v_dot in [−a_max, a_max], w_dot in [−α_max, α_max] (or equivalently bounded wheel torques). The bounds are the whole point: a robot that can stop and spin in place instantly has no momentum to shed and therefore nothing to bridge.
- Consequence: residual forward speed at a junction cannot be canceled instantly; cancelling it costs time and distance, and distance is bounded by the wall. This is the direct analog of "the pole's angular momentum cannot be canceled instantly."

---
3. Skills / policies — definition

A skill is a controller action = pi(s) that drives the robot along one corridor: hold the corridor's centerline, align heading to the corridor axis, and regulate forward speed to that corridor's target cruise speed. The policy can only handle the case the robot is placed exactly on the centerline as it can only make it cruise in a straight line.

Constraint you set (point 2): the seven (in this case) policies are genuinely distinct, not one parameterized family. Each policy's domain is a disjoint region of state space (different x, y coordinates): policy k is only defined where x, y lie in corridor k's extent. Different corridor = different slab of state space = different policy. They share a functional form (all are "follow-this-axis" controllers), but that is an implementation convenience, not an identity; their domains do not overlap, so they are different policies.

This framing does real work: it makes the skills narrow by construction. A policy is simply undefined outside its corridor's region and unreliable near the edges of it. That narrowness is mandatory — a globally robust controller would steer out any cross-axis velocity on its own and there would be nothing to bridge (same reason cartpole's balance must be narrow).

---
4. Tubes, initiation sets, windows — the precise definitions

- Tube of skill k: the bundle of trajectories skill k produces inside corridor k — concretely, states near the centerline, aligned with the axis, at or near cruise speed, spanning the corridor's length. Spatially extended (a corridor is a segment, not a point) — which is exactly what makes a window possible.
- Nominal initiation set of skill k (your point 3): the aligned approach cell — the cell on the same row/column as corridor k, immediately before its start, with heading along the axis and speed in range. Not the feeder corridor's cell. This is the state the policy was designed to begin from.
- Window of skill k (your point 3): an n-cell interval around/after the corridor start, along the corridor. These are early in-tube states. The bridge is allowed to hand control to skill k anywhere in this window — not only at the nominal start.
- The demonstrable claim, made precise. In a perpendicular-corridor maze the robot never arrives at the nominal initiation set — it almost always enters from the side (the feeder corridor), carrying cross-axis velocity. So the nominal initiation set is, in practice, almost never used (except for 6-7 in this case). Bridging is therefore not an edge case here; it is almost always the only way a corridor is entered. The bridge converts a side-entry (out-of-set, cross-axis, would-crash) into a window state on the centerline from which the policy can actually run. "We start a policy outside its initiation set" = we hand it a window state that is not its nominal approach cell.

---
5. The controller / switching mechanism (your point 1)

Purely positional, no planner:

- Continuously read the robot's cell.
- Map cell -> owning corridor -> that corridor's skill. This is the high-level controller, set implicitly by the map.
- On a change of owning corridor (robot crosses from corridor j's region into corridor k's region), the target skill changes to k. Because the robot is moving along j's axis (perpendicular to k), it is outside k's nominal initiation set — so the controller does not hand straight to k. It first invokes the bridge j->k.
- The bridge runs until its success condition holds, then merges into skill k's tube — control passes to k at a window state on the centerline.

The bridge operates in the junction region (which topologically belongs to k). The previous skill j is interrupted the instant the robot reaches the junction at full corridor speed pointed along j — i.e., j is not required to decelerate or align first. That interruption-from-a-bad-state is pillar 1.

Note this fixes the example's decisive variable: since the switch is triggered positionally (always at the junction), the free knob is residual speed — how fast skill j was driving — and robot heading, not interrupt timing.

---
6. The bridge — what it must produce

The bridge is the controller engaged only around a switch. It must synthesize behavior that no corridor skill contains:

- Heading change — rotate ~90° from axis j to axis k. 
- Cross-axis velocity cancellation — kill the component of velocity along j.
- Along-axis velocity establishment — build/regulate velocity along k so the robot enters k already moving (your "transitioning velocity"), rather than stopping and restarting.
- Centerline placement (your point 4): terminate with the robot on corridor k's centerline — not necessarily at k's start, but somewhere in the window.

Bridge success condition (handoff predicate): robot is inside corridor k's window, on/near centerline (within lateral tolerance), heading aligned with axis k (within angular tolerance), forward speed within k's acceptable starting range, and no wall contact occurred. When this holds, skill k takes over.

The combination 1–4 is the emergent behavior — the arc-turn-while-carrying-speed-onto-the-centerline that neither j nor k ever performs. We will forget about the bridging policy for now and deal with how to train it later.

---
7. The crux: width W vs. speed vs. turn radius, and failure modes

Carrying speed through a 90° turn requires an arc of radius r ≈ v / ω. The arc must fit inside the channel: too large and the robot clips the inside corner or hits the outside wall. So there is a maximum speed that can be carried through the turn, set by W, α_max, and a_max. This is the diff-drive analog of "is the pole spinning too fast to catch," and it is where the science lives.

- W is a sweet-spot parameter. Too narrow -> no arc fits, the bridge must stop and restart (kills the "carry velocity" goal — the trivial reset we want to beat). Too wide -> every speed works, no failures, nothing interesting. Choose W so that moderate residual speed is carryable by a good bridge but excess residual speed cannot be.

Failure / outcome modes:
- Crash (unrecoverable): wall contact during the turn — overshoot the junction or clip the inside corner. The hard failure that mirrors a dropped pole.
- Out-of-window: bridge fails to get aligned+centered before the n-cell window ends -> no clean handoff.
- Trivial reset (to be beaten): bridge sheds all speed, stops on the centerline, then k restarts from rest. Safe but defeats pillar 2.
- Recovery (the interesting non-nominal entry): residual speed too high for the ideal low-speed start, so the bridge lands the robot deeper in the window and/or at higher retained speed — on the centerline, aligned, no crash — i.e., it skips k's accelerate-from-rest prefix. This is the demonstrable "enter the tube past frame zero, carrying momentum."

For now, we will define the grid with some parameters that I can hand-tune (e.g. cell width, ...).