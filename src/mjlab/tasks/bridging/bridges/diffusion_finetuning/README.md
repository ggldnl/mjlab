# Diffusion planner fine tuning

Fine tunes the kinematic diffusion planner of `bridges/diffusion` with feedback
from the frozen universal tracker, including the plans it fails to follow.

`diffusion.train` already executes planner outputs with the frozen tracker and
replays the successful executions. Failed plans are dropped. This package keeps
that loop and adds a learned per frame critic of the tracker, so failures also
become a learning signal, located on the frames where tracking broke down.

The simulator is not differentiable. The critic is: it predicts, for every frame of
a planned window, the tracker's tracking score. The planner's own samples are scored
by the critic, and the gradient flows through it into the denoiser. Only the last
denoising rung keeps gradients, as in DRaFT.

## Package layout

```text
diffusion_finetuning/
├── critic.py   per frame feasibility critic, labels, replay
└── train.py    fine tuning task, extends the planner improvement runner
```

Nothing in `bridges/diffusion` is modified. Checkpoints keep its format.

## One cycle

| Step    | What happens |
|---------|--------------|
| collect | generate, gate and execute plans with the frozen tracker |
| label   | per frame tracking score, 0 from the first fallen frame on, 0 everywhere for gate rejected plans |
| critic  | train on every plan, successes, failures and gate rejects |
| planner | demonstration loss, physical replay share, feasibility loss, anchor |

Until the critic replay holds `--agent.feasibility-critic.min-samples` executed plans, cycles
only collect and train the critic. The planner is untouched.

## Run

Train a universal tracker and pretrain a planner as described in the diffusion
README, then:

```sh
uv run train Mjlab-G1-Diffusion-Planner-Finetune --agent.max-iterations 30 --agent.tracker-checkpoint logs/rsl_rl/g1_diffusion_universal_tracker/<run>/model_5999.pt --agent.planner-checkpoint logs/rsl_rl/g1_diffusion_kinematic_planner/<run>/model_30000.pt
```

Use `Mjlab-T1-Diffusion-Planner-Finetune` with T1 checkpoints. Output is written to
`logs/rsl_rl/g1_diffusion_planner_finetune/<run>/`.

Run walk to kick with the result:

```sh
uv run python -m mjlab.tasks.bridging.config.g1.tests.transitions.walk2kick --bridge diffusion --bridge-checkpoint logs/rsl_rl/g1_diffusion_planner_finetune/<run>/model_30.pt
```

## What to watch

| Metric | Meaning |
|--------|---------|
| `critic/fresh_plan_error` | critic error on plans it has not trained on, should fall |
| `finetune/predicted_score` | critic score of the planner's samples, rises as it learns |
| `plans/success_fraction` | measured by the real tracker, the number that matters |

If `finetune/predicted_score` rises while `plans/success_fraction` does not, the
planner is exploiting the critic. Watch the gate terms too: `gate/foot_placement`
and `gate/acceleration` jump first. Raise `--agent.finetune.anchor-weight` or lower
`--agent.finetune.feasibility-weight`.

Main options:

| Option | Default | |
|--------|---------|---|
| `--agent.finetune.feasibility-weight` | 0.1 | weight of the feasibility loss |
| `--agent.finetune.anchor-weight` | 1.0 | pull toward the planner as the run started |
| `--agent.finetune.feasibility-batch` | 64 | plans sampled per feasibility loss |
| `--agent.finetune.feasibility-every` | 4 | planner updates per feasibility loss |
| `--agent.finetune.gradient-rungs` | 1 | last denoising rungs with gradients |
| `--agent.planner.physical-fraction` | 0.10 | share of the replay of executed paths, 0 isolates the critic |
| `--agent.feasibility-critic.min-samples` | 2048 | executed plans before fine tuning starts |
