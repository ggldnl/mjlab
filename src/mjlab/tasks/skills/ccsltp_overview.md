# CCSLTP in broad terms

Lee, Sun, Somasundaram, Hu, Lim, "Composing Complex Skills by Learning Transition
Policies", ICLR 2019. Paper and reference implementation are in `references/ccsltp/`. This is a
plain description of what the architecture is and what each part is for. No math.

## 0. Summary for dummies

We have a bunch of primitive policies, each with an associated transition policy 
that has to be learned. We have an implicit controller which tells what primitive 
has to run next, but not when: each primitive tells when it finishes. 

During inference, we start with a policy and then, when it finishes, 
the meta-policy (implicit controller) decides what has to start next, 
picks the next primitive, invokes its transition policy which in turn 
brings the robot into a state from which the primitive can run safely.

Training instead is a chicken-and-egg problem. To train a transition policy 
we need to know which states are good launching points. To find out which states 
are good launching points we need transition attempts to succeed or fail. At the 
start we have neither.

- Before any training, run each primitive on its own, a thousand times, from its
  normal starting conditions. Throw away the runs that failed. From each surviving
  run, take the states from the first stretch of it (first 10/20%). Those are
  states the primitive is demonstrably happy to be in near its start. This is 
  the "good" pile i.e. the initiation set (automatically defined).
- Once we have the good pile, the predictor has something to learn from. We have
  one predictor and one transition policy for each primitive. 
- Four phases (for each primitive, independently):
  - **Collect.** Run episodes normally. A primitive runs until it reports done.
    the meta-policy names the next one. The primitive bridge takes over and 
    drives, using the predictor's current opinion as its reward: at every step,
    it is paid for having made the score go up since the last step. 
  - **File the evidence.** Look back at the bridging attempts and check the 
    verdict of the primitive. 

## 1. The setting

You are given a set of primitives, which is their word for what we call experts: short
policies, each trained on its own for one behaviour, frozen from then on. You are also
given a task that needs several of them in sequence, like walking then jumping then
crawling.

Running them back to back does not work. Each primitive was only ever trained on
states its own training distribution produced, so when the previous one hands over,
the next one is looking at a state it has never seen and falls over. The set of states
a primitive can actually start from is its initiation set, and the problem is that the
previous primitive does not leave you inside it.

Their answer is to insert a third kind of policy between every pair, whose whole job
is to get from wherever the previous primitive stopped into the next one's initiation
set. They call it a transition policy. We call it a bridge.

The obvious way to train that bridge is to reward it when the next primitive succeeds
afterwards. That reward is far too sparse to learn from: one bit, arriving long after
the actions that caused it. Essentially the entire paper is about manufacturing a
dense reward to replace it.

## 2. The cast

**Primitives.** Frozen pre-trained policies. Beyond acting, each one must be able to
say, at every step, whether it is still going, has succeeded, or has failed. That
three-way signal is the only thing the rest of the system asks of them.

**Meta-policy.** Decides which primitive comes next. In the paper this is essentially
not a research contribution: it is a rule the environment provides, an oracle that
names the next primitive when the current one reports it is done. Our controller.

**Transition policy.** The bridge. One per primitive, indexed by the primitive being
transitioned *into*, and shared across every primitive it might be coming *from*. It
is told which primitive it is coming from as an extra input, so a single network
serves all the predecessors of a given target. It emits actions in the same action
space as the primitives, and it emits one more thing: its own decision to stop. It is
not run for a fixed number of steps; it decides when it has arrived, up to a cap of
100 steps.

**Proximity predictor.** The heart of the method. One per primitive. It is a small
network that looks at a state and answers one question: how close is this to a state
the next primitive could start from. Close to 1 means "you are essentially in the
initiation set", 0 means "hopeless". It is what turns the sparse success bit into a
dense per-step signal.

## 3. What the proximity predictor buys you

Once you have a function that scores any state by how good a launching point it is,
the bridge's reward writes itself: reward it, at every single step, for having
increased that score since the last step. Plus a bonus at the end for the score of the
state it actually stops in.

That combination is doing two different jobs, and both are needed. The per-step
increase is the dense part: it gives feedback immediately, on every action, so the
bridge can learn without waiting for the outcome. The terminal bonus is what stops the
bridge from gaming the dense part, since a policy paid only for improvement has an
incentive to wander somewhere bad first so it has room to improve, and to keep
improving forever rather than stop and hand over.

Note what this signal is not. It is not a distance in any geometric sense, and it is
not handed to us. It is a learned, per-primitive, asymmetric notion of "how good is
this state for *this specific skill*", which is exactly the notion of distance between
robot states that our problem definition says the bridging problem needs.

## 4. Where the predictor's own training signal comes from

The predictor is trained from two piles of remembered states, kept per primitive as
fixed-size queues that forget the oldest entries.

The success pile holds states that genuinely led into a working start. The failure
pile holds states from bridge attempts where the next primitive then failed. The
predictor is trained to score the first pile high and the second pile at 0.

States in the success pile are not all scored equally. The state where the primitive
actually took over scores 1, and states earlier in the run leading up to it are
discounted, multiplied by 0.95 for each step of distance. So the label encodes not
just "this worked" but "this was about this far away from working". That decay is what
makes the predictor produce a smooth gradient across the state space rather than a
cliff, and it is why the bridge can follow it.

The piles are filled by the bridge itself, as it trains. Every bridge attempt is run
to its conclusion: the bridge stops, the next primitive takes over, and eventually
reports success or failure. That verdict is then propagated backwards over the whole
attempt, and every state visited during it gets filed into the corresponding pile.

So the two networks bootstrap each other. The predictor tells the bridge where to go;
the bridge goes there and finds out whether it worked; the answer trains the
predictor; the predictor gets sharper about which states really are good. The training
loop just alternates: a batch of rollouts, some gradient steps on the predictor, a PPO
update on the bridge, repeat.

The adversarial reading is worth holding on to, because it explains why this works.
The bridge is a generator trying to reach states the predictor scores highly. The
predictor is a discriminator that keeps being shown the bridge's own failures and
learns to stop being fooled by them. Any state the bridge finds that scores high but
does not actually work gets filed as a failure and scored down. The predictor cannot
be exploited for long, because it is retrained on precisely the states its current
mistakes lead the bridge into.

## 5. The cold start

At the very beginning both piles are empty and the predictor knows nothing, so the
bridge has no signal at all. They seed it: run each primitive on its own for 1000
episodes, take the successful runs, and file the states from the first 10 to 20
percent of each one into that primitive's success pile.

That opening slice is an empirical stand-in for the initiation set. Nobody wrote down
what the initiation set is; it is simply the states the primitive is observed to pass
through just after it starts working. This is the same idea as our window of early
states along the tube, and it means the whole method only ever needs rollouts of the
primitives, never any privileged description of them.

## 6. One rollout, end to end

The meta-policy names the next primitive. The bridge for that primitive is engaged and
runs, at each step receiving the increase in proximity as its reward, until either it
declares itself done or it hits the 100 step cap. Control passes to the primitive,
which runs until it reports success or failure. That verdict labels every state the
bridge visited, which fills the piles. Then the meta-policy names the next one and it
repeats. At the end of the batch the predictor gets its gradient steps and the bridge
gets its PPO update.

Because several primitives get exercised in one episode, the flat rollout has to be
cut into per-primitive segments afterwards, since each target primitive has its own
bridge and its own predictor to update.

## 7. What they compared against

The ablations are the useful part, because they isolate exactly what the proximity
machinery contributes.

- Without-TP: no bridge, primitives run back to back. The baseline failure.
- TP-Task: a bridge trained on the sparse task reward. Tests whether a bridge helps at
  all without the dense signal.
- TP-Sparse: a bridge rewarded only by the proximity of the state it stops in, no
  per-step term. Tests whether the density specifically is what matters.
- TP-Dense: the full method.

Plus end to end RL with a hand-engineered dense reward, as the "why not just train one
policy" control.

## 8. What is load-bearing and what is incidental

Load-bearing:

- Frozen primitives that report success or failure.
- A learned, per-target, state-quality signal, kept honest by being retrained on the
  bridge's own failures.
- A dense reward from that signal, with a terminal term to stop the bridge gaming it.
- Initiation sets estimated from rollouts of the primitives, never specified by hand.
- The bridge choosing its own moment to hand over.

Incidental, or at least specific to their setup:

- Tiny networks and small batches, because it is 2019 single-environment MuJoCo with
  MPI workers. None of the sizes should be copied.
- The rule-based meta-policy, which is a placeholder for our controller.
- The sequential rollout generator with its nested loops. On a vectorized env this
  becomes per-env phase masks, and it is the single biggest structural difference we
  will have to deal with.
- One network per primitive. The bridge is already conditioned on which primitive it
  came from, so conditioning on the target as well and sharing one network is a small
  change to their design, and it is the variant we want to be able to try.

## 9. The seam: making the state-quality signal swappable

The proximity predictor answers one question, "how good is this state as a launching
point for skill X", and everything else in the architecture consumes only that answer.
That makes it the natural place to cut.

The alternative we want to try is to use the next skill's own value function instead.
If the skills were trained with an actor-critic, their critic already estimates how
well things will go from a given state, which is the same question asked a different
way, and it costs nothing extra to obtain. This is what Tidd et al 2022 do for legged
locomotion, and their results are strong.

Two things are worth knowing before we run that experiment.

First, the property that swap gives up is the arms race described in section 4. A
frozen critic cannot be corrected. Where it is wrong the bridge will find that out and
exploit it, driving towards states the critic overrates, and nothing pushes back.
Tidd's fix is to damp the signal wherever the critic is demonstrably unreliable, which
they detect by checking whether the critic's own prediction is consistent with what
actually happens next, and suppressing the reward where it is not. So the swap is not
quite free, and the damping is part of the method rather than a detail.

Second, a critic is trained on the skill's own state distribution, which is exactly
the region the bridge is *not* in. Off that distribution, value functions are
famously over-optimistic. So this substitution is strongest when the bridge is already
close, and weakest at the start of a transition, which is the opposite of the
proximity predictor's failure mode.

Practically, this means the interface should be small: something that takes a batch of
states and a target skill and returns a scalar per state, plus a way to be told about
outcomes for the implementations that learn online. The learned proximity predictor
uses the outcome channel and needs the two piles; the value-function version ignores
it and needs neither. Everything downstream, the reward shaping, the bridge, the
handover decision, should not be able to tell which one it is talking to.

This also has to hold across both bridge topologies, one shared bridge and one bridge
per expert, since the state-quality signal is indexed by the target skill in either
case and is independent of how many bridge networks exist.
