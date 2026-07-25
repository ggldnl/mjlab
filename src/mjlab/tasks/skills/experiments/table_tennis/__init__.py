"""
Robot arm playing table tennis: a KUKA iiwa14 (`mjlab.asset_zoo.robots.kuka_iiwa_14`)
with a racket welded to its wrist, and a 40 mm ball.

Four primitive skills, each trained on its own and later composed by a bridge:

- catch   : a ball falls out of the air; get the blade under it and take its energy
            out so it settles on the bat instead of bouncing off.
- balance : a nearly-stationary ball sits over the blade; keep it on the sweet spot,
            as close as possible to the racket's "face" site.
- toss    : a ball resting on the blade; launch it straight up so its apex reaches a
            commanded height, without drifting sideways.
- hit     : a ball drops in with real momentum; strike it so it reverses and comes to
            a standstill at the commanded point, arriving there with nothing left.

The ball may only touch the racket, in every task. Contact with the arm or the floor
is a failure, enforced by two contact sensors on the scene; `hit` ends at the ball's
apex so its ball never gets the chance to reach the floor.

Every spawn is anchored to the blade's own x/y and differs only in height above it, so
the ball is always somewhere the arm can actually play it whatever the ready stance is
tuned to.

All four share one observation and action space, and one command whose meaning is the
same everywhere: where the ball should end up. That is what lets a bridge act in the
skills' own space and hand control between them.

To train:

    uv run train Mjlab-TableTennis-Catch
    uv run train Mjlab-TableTennis-Balance
    uv run train Mjlab-TableTennis-Toss
    uv run train Mjlab-TableTennis-Hit

Look at where each task starts its ball, with its initial velocity drawn as an arrow
and its goal as a sphere (no checkpoints needed):

    uv run python -m mjlab.tasks.skills.experiments.table_tennis.demo --debug-init catch

Test a trained skill on its own, or watch the composition:

    uv run python -m mjlab.tasks.skills.experiments.table_tennis.demo --debug
    uv run python -m mjlab.tasks.skills.experiments.table_tennis.demo
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.skills.experiments.table_tennis.table_tennis_env_cfg import (
  balance_env_cfg,
  catch_env_cfg,
  hit_env_cfg,
  table_tennis_ppo_runner_cfg,
  toss_env_cfg,
)

# Per-task initial exploration std. Gaussian exploration is white noise -- an
# independent perturbation every control step -- and the contact skills cannot survive
# much of it: at std 1.0 the blade is commanded +-0.5 rad of jitter at 100 Hz, which
# flings the 2.7 g ball off the bat in ~15 steps. Smooth motion of the same amplitude
# keeps it on for ~350, so the limit is the noise, not the task. Balance is the most
# delicate (it has to hold a ball that is already resting); catch has to settle one, so
# it is sensitive too. Toss and hit want to move hard and keep the default.
_SKILLS = {
  "Catch": (catch_env_cfg, "table_tennis_catch", 0.4),
  "Balance": (balance_env_cfg, "table_tennis_balance", 0.25),
  "Toss": (toss_env_cfg, "table_tennis_toss", 1.0),
  "Hit": (hit_env_cfg, "table_tennis_hit", 1.0),
}

for _name, (_cfg_fn, _experiment, _init_std) in _SKILLS.items():
  register_mjlab_task(
    task_id=f"Mjlab-TableTennis-{_name}",
    env_cfg=_cfg_fn(),
    play_env_cfg=_cfg_fn(play=True),
    rl_cfg=table_tennis_ppo_runner_cfg(_experiment, init_std=_init_std),
  )
