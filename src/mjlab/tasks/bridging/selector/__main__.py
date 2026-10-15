"""Record and build the entry points of one robot, see the package docstring."""

import tyro

import mjlab
from mjlab.tasks.bridging.selector.run import SelectorCfg, main

if __name__ == "__main__":
  main(tyro.cli(SelectorCfg, config=mjlab.TYRO_FLAGS))
