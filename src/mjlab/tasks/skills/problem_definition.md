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

I'm leaning toward the second option because it keeps experts indipendent and observable and introduces no side behaviors, but it also has some problems. How do we know when we are in a suitable state from which the next expert could start safely? This in particular requires having a notion of "distance" between robot states and this by itself is a whole new problem. Also, from where should the next expert start (where should the bridge make the next expert start)? For example, what if the robot currently has some momentum that the next expert could exploit at regime (after it starts) but the expert itself requires a "reset" before (for example due to how it is trained)? A trivial example could be the following. 

Imagine we have a humanoid robot that has two policies: running at a given speed and jumping from a standing position. If the signal to jump arrives while the robot is running, a naive direct hand-off will make the robot stop, loosing momentum, and then jump; our method should exploit the momentum to jump directly while running.

We will have to somehow know this in advance, to build a "policy descriptor" that predicts what the policy will do in the subsequent time window in order to establish where the hand-off should happen.

This (where the hand-off should happen) is very hard to implement, so we will first focus on a simpler version of the problem and treat this as a future expansion. The simpler version just makes the bridge answer the first question (how to make the next expert start safely?). 

To implement the simplified version, we will heavily rely on a paper I found: Composing Complex Skills by Learning Transition Policies.
We will have to:

1. Search for what comes next this paper: is there any other work that takes the ideas of this paper and expands on them to improve the technique?
2. Re-implement the architecture on mjlab to have a baseline. Details can be found in the paper and on the official repo. You can find them in the `ccsltp` folder. 
3. Experiment and extend that implementation. For this reason, while reimplementing the architecture, avoid at all cost implementing useless stuff and be extra careful of using a modular approach. Each component has to be clearly separated by the others in its respective file, dependencies should be clearly stated and be sound. We will discuss about the general plan and I will give you guidance on the overall implementation, leaving the details to you.

We will then apply this architecture to some concrete problems. The problems span various robots and environments, so modularity is of paramount importance.
Some things they do in the paper have some friction with mjlab. For this reason, we will talk about what to implement step by step, one thing at a time.
