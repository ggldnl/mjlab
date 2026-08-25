"""The individual G1 skills, one sub-package each.

Every sub-package registers its task on import (see `mjlab/tasks/__init__.py`, which
walks this tree), exports the task id its consumers name it by, and keeps its
environment config and its MDP terms to itself. There is deliberately nothing shared
here: two skills that happen to want the same reward term should each say so, because
the moment one of them is retuned the sharing becomes a coupling nobody asked for.
"""
