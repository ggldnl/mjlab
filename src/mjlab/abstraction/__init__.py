"""Abstraction mechanism: template-model reference signals to guide RL.

See :mod:`mjlab.abstraction.abstraction` for the design rationale.

This package holds only the *general* mechanism (the base classes). Concrete
abstractions live in their task packages (e.g.
``mjlab.tasks.stepover.mdp.abstractions``) and are imported there, never here -
importing task code from this core package would create a circular import
(core -> tasks -> core).
"""

from mjlab.abstraction.abstraction import Abstraction, AbstractionCfg

__all__ = [
  "Abstraction",
  "AbstractionCfg",
]
