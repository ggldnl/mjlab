# Goal CVAE bridge

The student sees the live state, target dynamic state, deadline, measured foot
history, and optionally five known frames after the target. It does not see the
bridge's demonstrated future. A route predictor chooses touchdown count and
support-foot sequence; the prior samples one latent per bridge window. The PPO
oracle sees the demonstrated dense route and labels the states visited by the
student during DAgger.

Use a trained oracle checkpoint that has passed `cvae.oracle.evaluate`. Then:

```sh
uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.collect --checkpoint <oracle.pt>
uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.branch --checkpoint <oracle.pt>
uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.merge
uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.evaluate_oracle --checkpoint <oracle.pt>
uv run train Mjlab-G1-Goal-CVAE-Bridge --agent.teacher-checkpoint <oracle.pt>
uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.evaluate --checkpoint <student.pt>
```

Inspect the branch collector's accepted starts and the oracle replay's per-channel
errors before starting a long student run. A physically valid A-to-B rollout
does not imply that the oracle can correct back onto it after a perturbation.
The branch collector keeps only simulated continuations from a shared start;
it never pairs endpoints from different rollouts. If it cannot retain distinct
branches, the dataset has not established causal endpoint conditioning.
All branches from one shared start remain in the same train/evaluation split.

For `walk2kick`, rebuild selector recordings so selected entries include foot
pose/velocity/contact and the short continuation after B:

```sh
uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.record
uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.build
uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2kick --bridge goal-cvae
```

The post-B context is optional when an entering skill has no known continuation.
The CVAE loss supervises actions, route labels and demonstrated foot trajectories
through shared decoder features;
it does not backpropagate terminal physical errors through MuJoCo. Always judge
handoff accuracy from held-out root, leg, foot and action-jump metrics.
