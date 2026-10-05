# Footstep diffusion bridge

The diffusion bridge of `bridges/diffusion`, conditioned on footsteps. A heuristic
planner decides where and when each foot is planted between A and B, the diffusion
model generates the whole body around those footsteps, and the universal tracker
executes the result.

The previous version of this package built feet, root and hand trajectories by hand
and solved whole body IK on them. Each trajectory was smooth on its own but they did
not agree with each other, and the motion came out jerky. Here only the planted
footsteps are fixed. Swing trajectories and everything else come from the learned
prior.

## Footstep channels

Every frame carries six channels per foot after the diffusion planner's condition
channels: a contact flag, the planted sole position and its yaw. Swing frames have
zero positions. The known mask tells the model which footsteps it is given:

```text
contact known, position known     planted here, at this spot
contact known, position unknown   planted here, anywhere
contact unknown                   no opinion
```

Training reads footsteps from the clip after time warping and mirroring, so they
always match the augmented window. Each window then picks what to show, in the
spirit of MaskedMimic:

```text
all footsteps           40%
random stance runs      40%, each run kept with probability 0.5
no footsteps            20%
timing only             10% on top, positions hidden
B root only             10% on top, B's joints hidden
root keyframes          10% on top, two interior frames
```

Shown footsteps get a 1.5 cm and 0.05 rad offset per stance run and one frame of
timing jitter in 30% of feet, so the model tolerates the heuristic planner. The
windows with no footsteps make the checkpoint usable without a planner too.

## Footstep planner

The planner reads speed from the boundaries, not only positions. A cubic Hermite
curve through the root position and velocity at A and B gives the root path, so a
fast A and a still B is a braking root. Ticks decelerating faster than 0.5 m/s^2
count as braking.

How many steps: a foot that moves more than 4 cm, or turns more than 1.5 rad,
covers its distance in corpus strides at the peak root speed, at least one step.
Feet that start and end on the same spot do not step, so a root swaying in place
plans no steps. Both thresholds come from the corpus: 97% of planted feet moving
less than 4 cm never step, and a foot that only turns pivots in 70% of windows up
to 1.5 rad.

When: landings alternate between the feet. The gap between landings and the swing
time are corpus medians at the root speed of that moment, in two rows, steady and
braking. Above 1 m/s braking steps come every 17 to 20 ticks against 24 when
steady, so a braking root gets quicker steps, which shrink as the root slows.
Steps that do not fit before B are compressed in time. A foot lifted at A lands
first, and a foot lifted at B swings through B.

Where: a foot lands where the root will be at touchdown, moved sideways by its
stance offset and ahead along the root velocity by the corpus lead at that speed.
The last landing of a foot planted at B is B's own plant.

All tables (stride, interval, swing, lead per speed bin) are fitted from the
corpus before training and stored in the checkpoint. On 1024 held out windows,
against the steps the clips actually take:

```text
                         landings   mismatch   interval
clip                        1.2                  23.3
evenly spaced planner       1.9       0.89       14.2
speed aware planner         1.1       0.23       24.1

braking windows only
clip                        1.6                  20.7
evenly spaced planner       2.4       1.04       14.2
speed aware planner         1.3       0.41       22.7
```

Mismatch is the mean absolute difference in landings per window. Contact at B is
judged from the sole speed as well as its height, stepping the B state back one
tick along its velocities. With height alone a foot gliding in just before
touchdown counted as planted and got a step it never takes.

## Planting the feet

The model learns to follow its footsteps but is not forced to. After every
denoising rung, a few damped least squares steps on the leg joints pull each
planted sole onto its footstep. Correcting planted frames alone made the joint
correction jump at every lift and landing. On a short run that gave joint
accelerations six times those of the recorded clips. So the correction is
interpolated through swings, smoothed, and tapered to zero at A and B.

## Train

```sh
uv run python -m mjlab.tasks.bridging.bridges.mixed.train --robot g1
```

Checkpoints are written to `logs/rsl_rl/g1_footstep_diffusion_planner/<run>/`.

## Inspect and evaluate

```sh
uv run python -m mjlab.tasks.bridging.bridges.mixed.view --checkpoint logs/rsl_rl/g1_footstep_diffusion_planner/<run>/model_30000.pt
uv run python -m mjlab.tasks.bridging.bridges.mixed.evaluate --checkpoint logs/rsl_rl/g1_footstep_diffusion_planner/<run>/model_30000.pt
```

The viewer switches footsteps between the planner, the clip and none. The evaluation
reports, per source, the sole distance to the given footsteps and to the clip's
footsteps, planted sole slip and joint acceleration. If the model ignored its
footsteps, the distance to clip footsteps would be the same with and without them.
A second table compares the planner's steps with the clip's, by speed at A and for
braking windows. `--generate False` prints only that table, without the model.

## Run walk to kick

The tracker is the diffusion universal tracker, unchanged:

```sh
uv run python -m mjlab.tasks.bridging.config.g1.tests.transitions.walk2kick --bridge mixed --tracker-checkpoint logs/rsl_rl/g1_diffusion_universal_tracker/<run>/model_5999.pt
```
