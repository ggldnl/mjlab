# Related work: what came after CCSLTP

Survey done for step 1 of `problem_definition.md`. Starting point is Lee, Sun,
Somasundaram, Hu, Lim, "Composing Complex Skills by Learning Transition Policies",
ICLR 2019 (referred to as CCSLTP below). Source for the citation sweep is the
Semantic Scholar citation graph, 102 citing papers, filtered by relevance to
skill transitions rather than to hierarchical RL in general.

Confidence is marked per entry: [read] means the paper text was read, [abstract]
means only the abstract, project page or a summary was consulted.

## 1. Direct descendants: keep the bridge, change the reward

These accept CCSLTP's framing (pre-trained frozen skills, a separate short-lived
transition policy) and attack the same weak point, which is that the transition
policy's natural reward signal is sparse.

**Byun and Perrault, "Training Transition Policies via Distribution Matching for
Complex Tasks", ICLR 2022.** [read] The most important follow-up. Same
environments, same primitives, direct comparison against CCSLTP.

Two changes:

1. The proximity predictor is replaced by AIRL. The transition policy is a GAN
   generator; the discriminator is trained to separate the transition policy's
   trajectories from trajectories the *next skill itself* produced during the
   transition interval. Reward for the transition policy is the usual AIRL
   discriminator score. So instead of learning a scalar "how close am I to the
   initiation set", it matches the next skill's state-action distribution
   directly.
2. The termination decision is pulled out of the transition policy. CCSLTP folds
   a two-way softmax termination head into the policy and into the PPO ratio.
   Byun instead trains a separate double-DQN with a binary action space
   (switch / stay), rewarded with the sparse success or failure of the next skill.
   Their argument: the sparse signal is tolerable when the decision space is one
   binary dimension, and it is not when it is the full continuous action space.

Results, average success counts over 50 sims, 3 seeds. Arm manipulation is a
wash: CCSLTP 4.84 / 4.97 / 0.92 versus theirs 4.77 / 4.76 / 1.00. Locomotion is
where they win clearly: Patrol 3.33 -> 3.97, Hurdle 3.14 -> 4.84, Obstacle Course
1.90 -> 3.72. Note that on Hurdle, CCSLTP (3.14) is worse than a single
end-to-end TRPO policy (4.13), so the locomotion gap is real.

Two things matter for us. First, they explicitly define a **transition interval**,
a window before the switch point, and collect the next skill's trajectories
*inside that window* as the matching target. That is our "tube window" under a
different name. Second, the DQN is choosing *when* to hand over, which is a
restricted form of choosing *where in the window* to enter.

**Xu et al., "KEPT: Key Experience Prioritized Transition Policies for Smooth
Skill Composition", 2026.** [title only] Appears in the citation graph but I
could not retrieve the abstract or venue (Semantic Scholar rate limits, not
indexed by the web search). Title suggests prioritized replay over
transition-critical experience. Worth chasing down manually.

**"Seamless skill transitions with hierarchical reward shaping and failure-driven
replay" (HAFER), Neurocomputing, 2026.** [abstract] Two mechanisms: a
hierarchical adaptive reward that mixes local subtask completion with global task
progress, and failure-driven replay. Their stated motivation is directly relevant
to us: as training progresses, failure signals at skill boundaries become rare
exactly when they are most needed to correct subtle transition errors. This is
the same pathology CCSLTP's failure buffer papers over.

## 2. The competing paradigm: fix the skills instead of bridging them

This is the branch that got more traction, and it is worth being explicit that
the thesis is *not* taking it.

**Lee, Lim, Anandkumar, Zhu, "Adversarial Skill Chaining for Long-Horizon Robot
Manipulation via Terminal State Regularization" (T-STAR), CoRL 2021.**
[abstract] Same first author as CCSLTP, and he abandons the transition policy.
Instead of inserting a bridge, he fine-tunes each skill so that its *terminal*
state distribution is regularized, adversarially, to stay inside the next skill's
initiation set. The argument is that widening the next skill's initial state
distribution does not scale, because the required coverage grows with chain
length. Solves IKEA furniture assembly, where prior skill chaining fails.

**Chen et al., "Sequential Dexterity", CoRL 2023.** [abstract] Learns a
transition feasibility function that both fine-tunes the sub-policies to improve
chaining success and enables autonomous policy switching, including recovery from
failure and skipping redundant stages. The feasibility function is close in
spirit to a proximity predictor, but it is used to *modify the skills* rather
than to reward a bridge.

**Wang et al., "SCaR: Refining Skill Chaining for Long-Horizon Robotic
Manipulation via Dual Regularization", NeurIPS 2024.** [abstract] Extends
T-STAR. Dual regularization: intra-skill (adaptive equilibrium scheduling
balancing an RL and an IL objective during pre-training) and inter-skill
(bi-directional adversarial learning during fine-tuning). Evaluated on IKEA
assembly and kitchen organization with real-world validation.

The trade-off is clean and worth stating in the thesis: this branch buys
robustness by giving up the independence of the experts. Every skill has to be
retrained knowing its neighbours, which is the property `problem_definition.md`
says we want to keep.

## 3. The "when and where to hand over" thread

This is the thread that speaks to pillar 2 (enter the next tube at the best point
in a window), which the problem definition defers but which the literature has
already started on.

**Tidd, Hudson, Cosgun, Leitner, "Learning Setup Policies: Reliable Transition
Between Locomotion Behaviours", RA-L 2022.** [read] Closest existing work to
what we want, and it is on a legged robot (3D biped over jump terrain). A "setup
policy" bridges from a default walking policy to a target terrain policy, and it
outputs both joint torques and its own switch condition.

The interesting part is the reward. They do not train a proximity predictor.
They reuse the **value function of the target policy** as the measure of how good
a state is for the next skill, which is free if the skills were trained with an
actor-critic. The problem is that value functions are over-optimistic off their
training distribution, exactly where the bridge operates. Their fix is the
Advantage Weighted Target Value:

    r_t = (1 - min(alpha * A_hat^2, 1)) * beta * V^target(s_t)

where `A_hat = r_t + gamma * V(s_{t+1}) - V(s_t)` is the TD error under the
target policy's own reward. Where the target value function is trustworthy the TD
error is near zero and the bridge maximizes V. Where it is not, the weight
collapses and the reward is suppressed. This is a genuinely cheap substitute for
CCSLTP's whole proximity machinery, and it needs no success/failure buffers.

They also handle the credit assignment gap after the switch with an "extended
reward": the reward earned by the target policy *after* handover is added back
onto the last reward entry in the setup policy's buffer.

Results: single difficult jump terrain 51.3% -> 82.2%; random sequence of
obstacles 1.9% -> 71.2%.

**Byun and Perrault's DQN** (above) is the other answer to the same question.

**Sui et al., "N2M: Bridging Navigation and Manipulation by Learning Pose
Preference from Rollout", 2025.** [abstract] From the same lab (CLVR). Learns,
from manipulation policy rollouts alone, which base poses the manipulation policy
prefers, then drives the robot there before handing over. This is an initiation
set learned empirically from rollouts, in an explicit state representation, which
is exactly our recipe applied to mobile manipulation. Ego-centric observation
only, no global state. 3% to 54% better than reachability-based baselines.

**Discovery of skill switching criteria for learning agile quadruped locomotion,
Frontiers in Robotics and AI, 2026 (arXiv 2502.06676).** [abstract] Skills
(trot, bound, gallop) trained separately from contact patterns, then switching
criteria discovered rather than hand-specified, driven by distance to goal. Real
hardware, with failure recovery.

## 4. Humanoid-specific, 2026

Newest and closest to the eventual target, but note the different assumptions.

**"RPG: Robust Policy Gating for Smooth Multi-Skill Transitions in Humanoid
Fighting" (arXiv 2604.21355).** [abstract] Two ideas. During expert training,
execution is stochastically interrupted at arbitrary timesteps to force each
expert to start and terminate from a broad state distribution. At runtime, a
learned gating network blends expert outputs (separate weights for upper and
lower body) rather than switching between them.

Both of these push against the thesis's choices: the interruption randomization
is the "widen the initiation set" strategy T-STAR argues does not scale, and the
gating is blending, not bridging, with the side-behavior risks that implies. It
is useful as a foil.

**"Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion
Matching" (arXiv 2602.15827).** [abstract, low confidence, the fetched summary
was vague] Retrieval-based: matches current state features against a window of
candidate frames in a motion database and executes the retrieved motion, rather
than synthesizing a transition with a policy. This looks like the paper the
README already cites as inspiration for pillar 2 ("matching against a window of
early frames using a velocity-aware state feature"). Should be read properly.

## 5. Things worth reading but not central

- Bagaria and Konidaris, "Option Discovery using Deep Skill Chaining", ICLR 2020.
  Learns initiation set *classifiers* by taking the last K states of successful
  trajectories and fitting a one-class classifier, then refining it into a
  two-class classifier from on-policy data. Not a bridging method (it chains
  backward from a goal), but it is a clean precedent for building an initiation
  set from rollouts, which is what CCSLTP's seeding step does.
- "Auxiliary Reward Generation with Transition Distance Representation Learning",
  2024. Learns a distance metric in transition space, which is the general form of
  the "notion of distance between states" the problem definition asks for.
- "Improving the Performance of Learned Controllers in Behavior Trees Using Value
  Function Estimates at Switching Boundaries", 2023. Same core trick as Tidd,
  applied to behaviour trees.
- BOSS, "Benchmark for Observation Space Shift in Long-Horizon Task", 2025.
  Benchmarks the chaining failure mode, but from visual observation shift, so it
  is off-target for an explicit state representation.

## 6. What this means for the plan

1. **CCSLTP is still a reasonable baseline to reimplement**, but it is not the
   state of the art on its own benchmark. Byun 2022 beats it on all three
   locomotion tasks, and locomotion is our eventual domain. Worth being upfront
   about that in the thesis rather than presenting CCSLTP as the frontier.
2. **Three candidate reward signals for the bridge**, all of which the
   reimplementation could support behind the same interface:
   - learned proximity predictor (CCSLTP)
   - AIRL discriminator against a window of the next skill's trajectories
     (Byun 2022)
   - advantage-weighted value function of the next skill (Tidd 2022), which is
     the cheapest since it needs no extra network and no buffers
   This is a natural axis for the modular decomposition: the bridge policy, the
   handover decision, and the "is this state good for the next skill" estimator
   should be three separate components with three separate files.
3. **Nobody in this list keeps the experts frozen, bridges, AND interrupts at an
   arbitrary moment.** CCSLTP and Byun both wait for the previous skill to reach
   a natural end or a scripted interval. Tidd switches at a fixed distance from
   the obstacle, detected by an oracle. RPG interrupts arbitrarily but widens the
   experts instead of bridging. The combination in `README.md` pillar 1 is
   genuinely unoccupied.
4. **Pillar 2 is less unoccupied than assumed.** Byun's DQN and Tidd's learned
   switch condition both pick the handover moment. What neither does is pick
   *which point of the next skill's tube* to aim for; they pick when to stop
   bridging. The distinction is worth making precise early, because it is the
   part of the contribution that survives this literature.
