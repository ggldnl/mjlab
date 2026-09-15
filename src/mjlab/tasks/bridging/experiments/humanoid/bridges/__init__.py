"""Bridge policies available to runtime tools."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BridgeSpec:
  name: str
  task: str


BRIDGES = {
  "docking": BridgeSpec("docking", "Mjlab-G1-Docking-Bridge"),
}
