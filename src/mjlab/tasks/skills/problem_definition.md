# The problem

I need to work on my thesis. This is the idea. Today we obtain policies for humanoid robot control in multiple ways:

* end-to-end: train a single policy that does a lot of stuff;
* distillation: train a lot of smaller policies and fuse them together into a single one;
* keep policies small and modular;

Each of the above techniques have its downsides:

* end-to-end is often unreliable, slow to train, difficult to make it converge;
* with distillation we get a single policy that has a worst behavior than each individual expert and that introduces side behaviors that could even be dangerous;
* keeping policies small and modular should be the best approach as each expert is independent, interpretable, takes less time to train, but then we have another problem: how do we use multiple experts together?

This is where I want to move for my thesis: developing an approach to put together multiple experts. There are two ways of doing so, we will refer to them as "blending" and "bridging".

* Blending is when we execute multiple experts at once. This could be useful for example in locomanipulation tasks, where for example an expert could handle the locomotion and one the manipulation. This is significantly harder to implement: we will get the same downsides as distillation in principle (side behaviors, less interpretability, ...);
* Bridging is when we have a higher level controller (whatever: a FSM, a planning algorithm, a neural network, ...) that establishes what expert could be used at a given moment and commands a switch. We have an expert currently running, this switching signal arrives, a bridging policy starts immediately and its role is to "put the robot in a state from where the next expert (decided by the controller) could start safely". This is because it is not always granted that the robot could transition to an expert from the state it is in (e.g. it is heavily unbalanced and about to fall). 

I'm leaning toward the second option because it keeps experts independent and observable and introduces no side behaviors, but it also has some problems. How do we know when we are in a suitable state from which the next expert could start safely? This in particular requires having a notion of "distance" between robot states and this by itself is a whole new problem. Also, from where should the next expert start (where should the bridge make the next expert start)? For example, what if the robot currently has some momentum that the next expert could exploit at regime (after it starts) but the expert itself requires a "reset" before (for example due to how it is trained)? A trivial example could be the following. 

Imagine we have a humanoid robot that has two policies: running at a given speed and jumping from a standing position. If the signal to jump arrives while the robot is running, a naive direct hand-off will make the robot stop, loosing momentum, and then jump; our method should exploit the momentum to jump directly while running.

We will have to somehow know this in advance, to build a "policy descriptor" that predicts what the policy will do in the subsequent time window in order to establish where the hand-off should happen.

This (where the hand-off should happen) is very hard to implement, so we will first focus on a simpler version of the problem and treat this as a future expansion. The simpler version just makes the bridge answer the first question (how to make the next expert start safely?). 

# Attack plan

To implement the simplified version, we will heavily rely on two papers I found: 
- Composing Complex Skills by Learning Transition Policies
- Training Transition Policies via Distribution Matching for Complex Tasks

We will take the best of both worlds and experiment a lot on each individual part of the resulting architecture.
For this reason, we will have to use a modular approach. Each component has to be clearly separated by the others in its respective file, dependencies should be clearly stated and be sound. 
We will discuss the general plan and I will give you guidance on the overall implementation, leaving the details to you.

## Composing Complex Skills by Learning Transition Policies (CCSLTP)

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

The paper breaks the chicken-and-egg problem by having the two pieces teach each
other over many rounds, each round adding a bit more information:

1. The transition policy (still mostly untrained, so it moves somewhat randomly
   at first) drives the robot to some state and stops there.
2. The next primitive is told to just try starting from that state. Sometimes
   it works, sometimes it doesn't. Either way, we write down the state and the
   outcome (success or failure) in a growing logbook.
3. A second small network, the "scorer", is trained on this logbook to guess,
   given any state, how likely the primitive is to succeed if it starts from
   there. Early on the logbook is tiny and the scorer is a poor guesser, but it
   improves as the logbook grows.
4. The transition policy is then trained (or re-trained) to drive the robot
   toward states the scorer currently rates highly, using ordinary
   reward-driven learning: reaching a high-scoring state is treated as a reward.
5. With a better transition policy, the robot now lands in different, hopefully
   better, states. Go back to step 2, try the primitive again from these new
   states, add the results to the logbook, retrain the scorer, retrain the
   transition policy, and so on.

Repeated enough times, this loop bootstraps itself: the logbook fills up with
useful examples, the scorer becomes an accurate judge of "is this a safe place
to start the next skill", and the transition policy gets good at steering the
robot to such places. At the very start of training there is also a bit of
free, cheap data: the primitive's own early states (right where it usually
starts from at the beginning of its own training) are used as a first batch of
known "good" examples, so the logbook and the scorer aren't starting from
absolutely nothing.

## Training Transition Policies via Distribution Matching for Complex Tasks (TTPDMCT)

This paper keeps the same overall picture (frozen primitives, a transition
policy that bridges between them) but changes two things: how the transition
policy learns what a good landing state looks like, and how the switch to the
next primitive is decided.

**How it learns to land well.** Instead of building a scorer from a logbook of
successes and failures, this paper watches the next primitive itself. Just
before the next primitive would normally start (and for a short window after),
you can record what states and moves it naturally produces on its own. Treat
this recording as "the example to imitate": the transition policy is trained
to make its own path through this window look just like the next primitive's
own path through it. This is done with a copy-catch game: one network (a
"referee") is trained to tell apart the transition policy's path from the
primitive's own recorded path, and the transition policy is trained to fool
the referee, i.e. to make its path indistinguishable from the real thing. As
training goes on, the transition policy gets better at fooling the referee,
and the referee gets better at spotting the difference, and the two push each
other until the transition policy's landing behavior closely matches what the
next primitive would have done anyway. This sidesteps the logbook-and-scorer
loop entirely: there's no need to run the primitive many times from many
states and note down pass/fail, since the primitive's own natural behavior is
used directly as the target.

**How the switch is decided.** In the first paper, the same transition policy
that decides "how to move" also decides "when to stop and hand over", learned
together as one thing. Here, that decision is pulled out into its own,
separate and very simple decision maker: a small network that only ever
answers one yes/no question, "should we switch to the next primitive right
now, or keep going?". It's trained using the same kind of success/failure
feedback as before (did the next primitive succeed once handed control?), but
because the question it has to answer is just yes/no, rather than "which exact
move should I make", it's much easier to learn well from a signal that only
shows up rarely.

Put together: the transition policy learns to move like the next primitive
would, by playing a copy-catch game against a referee network, while a
separate, simple switch-decider learns purely when to pull the trigger and
hand control over. During inference the flow is the same as before (current
primitive runs, finishes, controller picks the next one, transition policy
bridges, next primitive takes over), except the bridging path now imitates the
next primitive's own natural behavior, and the hand-off moment is chosen by
the dedicated switch-decider instead of being baked into the transition policy
itself.

## Next steps

The distribution-matching paper is the stronger of the two, but it still has
real weak spots, and each one is a natural place for the thesis to push
further.

1. **It only decides *when* to switch, not *where* to land.** The switch-decider
   is a yes/no question: hand over now, or keep going. It never asks "which
   point, among all the states the next primitive could accept, should we aim
   for". That's exactly pillar 2 from the problem statement (the running vs.
   jumping example): a bridge that can only pick a moment, not a target state,
   can't be momentum-aware. This is the most direct opening for our own
   contribution.

2. **The example it imitates is narrow.** The "recording" of the next
   primitive's natural behavior usually comes from watching it run under its
   own normal, undisturbed conditions, which means the target is close to a
   single typical way of starting, not the full range of states the primitive
   could actually handle. A bridge trained to imitate one narrow example may
   end up ignoring perfectly good landing states just because they don't look
   like the recording. Widening this recording (by letting the next primitive
   run from many different starting conditions, not just its default one)
   should make the target more representative of everything the primitive can
   actually cope with.

3. **Fooling the referee isn't the same as succeeding.** The copy-catch game
   rewards the transition policy for looking like the next primitive's own
   behavior, but "looks similar" is a stand-in for "will actually work", not a
   guarantee of it. It would be worth also checking, directly, whether the
   next primitive succeeds once handed control, and folding that outcome back
   into the reward, rather than trusting the referee alone.

4. **The copy-catch training is known to be finicky.** Adversarial setups like
   this one (two networks pushing against each other) are notoriously touchy
   to get to converge well, and the paper's own results reflect that: it wins
   clearly on locomotion but is basically tied with the older paper on arm
   manipulation. A steadier, cheaper signal to fall back on, or to mix in,
   would be worth having. One candidate from later work: if the next primitive
   was trained with a value function (a "how good is this state" estimate it
   already produces as a side effect of its own training), that estimate can
   be reused directly as a reward for the bridge, for free, without training
   any extra network. It would need care though, since a value function tends
   to be overconfident on states it never saw during its own training, which
   is precisely where the bridge spends its time.

5. **It doesn't win everywhere.** The reported gains are locomotion-specific;
   on manipulation the two papers are roughly equivalent. Before leaning on
   distribution matching as the default, it's worth checking early, on our own
   simplest experiment (the differential-drive robot), whether it actually
   beats the first paper's logbook-and-scorer approach, rather than assuming
   the locomotion result carries over.

The things I want to test first are:
- using a single bridge instead of one per primitive;
- deciding where to land;
- using the value function as a reward for the bridge;

On the value function (ideas coming from the paper "Learning Setup Policies: Reliable Transition Between Locomotion 
Behaviors", a separate work that tries to answer the same questions from a parallel path):
- When a policy is trained with the actor-critic method, we don't just get the policy, 
  we also get, as a free byproduct, a second small network called the value function. 
  Its only job during training was to answer "starting from this state, roughly 
  how much reward am I going to rack up going forward, if I keep acting as this policy?" 
- If the next primitive already has a value function, just ask it "how good do you think 
  this candidate state is?" and use that number directly as the reward for the bridge. 
  No logbook, no separate scorer network, no adversarial referee game.
- The problem is that a value function is only trustworthy on states similar to what 
  the primitive saw during its own training. The bridge, by definition, operates on states 
  the primitive normally wouldn't be in (that's the whole point of bridging). Off that 
  trained distribution, value functions tend to be overconfident (they'll happily output 
  a great score for a state that's actually bad, because they were never taught not to). 
- So naively using it as a reward would push the bridge toward states that only look good 
  to a network that's guessing. Their fix is to also compute a second number, a prediction 
  error: "did this state's actual outcome match what the value function predicted it would?" 
  When that error is small, the value function is judged trustworthy and its score is used as-is. 
  When the error is large, the value function's opinion is suppressed (weighted down) instead 
  of trusted. So the reward automatically discounts itself exactly where the value function is 
  least reliable, which is exactly the region the bridge lives in.

# Experiments

We will then apply this new architecture to some concrete problems. The problems span various robots and environments, 
so modularity is of paramount importance. 

The experiments are the following:

## Differential drive robot

A differential drive robot that can physically tip if turned too fast. It has the
following skills:
- drive: commanded to go forward at a fixed high speed (a narrow, forward-only 
  command range). It never sees a turning command during its own training, so 
  it doesn't know how to turn safely.
- turn: commanded to execute a turn (a nonzero angular-velocity command) at a much
  lower linear speed. It never sees a fast approach during its own training, so it
  never has to cope with momentum it didn't build up itself.

A simple, scripted controller drives the demonstration: run drive for a fixed stretch,
signal a switch to turn, hold turn for a fixed duration (or until the heading
error is small), signal back to drive, and repeat.

Tipping over is a measurable failure: a termination condition on excessive chassis 
roll/pitch is something the turn skill's own low-speed training avoids and a 
naive high-speed hand-off would risk.

The robot cruises at high speed, the controller calls a turn, the bridge visibly 
decelerates the robot into the turn without tipping, turn executes; the bridge 
reaccelerates, cruising resumes.

## Robot arm playing table tennis

A robot arm playing table tennis. It has the following skills:
- tossing the ball: with the racket at rest and the ball on the racket,
    the robot thrusts the racket up to launch the ball to a given height;
- serve; the ball is descending, the robot swings to send the ball to a target;
- return; the ball is incoming (from the opposite side of the table, 
  instead of from above as the serve case) and the robot strokes it to a target;

This experiment was also proposed by the CCSLTP paper, we can take a look at
their implementation, even though it was slightly different and not impressive
(we must do better).

We might use a DLR-KUKA lightweight arm with a racket attached at the end effector 
and enable the collisions only between the ball and the racket. Try to set up 
the mjlab configuration for the arm, I'll attach the racket to it.

## Humanoid robot running and jumping

A humanoid robot on a straight corridor. It has the following skills:
- running at a given speed (goal conditioned); we can simplify the problem by
    assuming it just runs forward (otherwise it will fall from the corridor 
    and terminate the episode);
- jumping at a given height (goal conditioned); this policy is trained with
    the humanoid in a standing position, so the experiment should demonstrate
    that the architecture is able to make it jump while running, skipping the
    initial states of the jumping policy and taking advantage of the residual
    momentum of the robot;
- crouching (optional); this policy is also trained with the humanoid in a
    standing position and additionally with a lower speed; this should
    demonstrate that the bridge is able to both slow it down and make it
    crouch while running absorbing the residual momentum of the robot.
The experiment consists of the robot having to complete the corridor 
with obstacles that require it to jump or crouch (optional). Obstacles
could be relatively low to make jumping easier.