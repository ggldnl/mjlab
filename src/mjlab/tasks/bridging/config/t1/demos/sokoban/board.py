"""Sokoban maps and validated plans, independent of simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Cell = tuple[int, int]
DIRECTIONS: dict[str, Cell] = {"E": (1, 0), "W": (-1, 0), "N": (0, 1), "S": (0, -1)}
DEFAULT_MAP = """#########
#       #
#       #
# @$  . #
#       #
#       #
#########"""


def advance(cell: Cell, direction: Cell, cells: int = 1) -> Cell:
  return cell[0] + direction[0] * cells, cell[1] + direction[1] * cells


@dataclass(frozen=True)
class Board:
  width: int
  height: int
  walls: frozenset[Cell]
  floor: frozenset[Cell]
  goals: frozenset[Cell]
  boxes: tuple[Cell, ...]
  robot: Cell

  @classmethod
  def parse(cls, text: str) -> Board:
    rows = text.strip("\r\n").splitlines()
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
      raise ValueError("Map must be a nonempty rectangle")
    walls, floor, goals, boxes, robots = set(), set(), set(), [], []
    for row_index, row in enumerate(rows):
      for x, symbol in enumerate(row):
        cell = x, len(rows) - 1 - row_index
        if symbol not in "# .$@*+":
          raise ValueError(f"Unknown map symbol {symbol!r}")
        if symbol == "#":
          walls.add(cell)
          continue
        if x in (0, len(row) - 1) or row_index in (0, len(rows) - 1):
          raise ValueError("Map boundary must consist of walls")
        floor.add(cell)
        if symbol in ".*+":
          goals.add(cell)
        if symbol in "$*":
          boxes.append(cell)
        if symbol in "@+":
          robots.append(cell)
    if len(robots) != 1 or not boxes or len(goals) != len(boxes):
      raise ValueError(
        "Map needs one robot and equal nonzero numbers of boxes and goals"
      )
    return cls(
      len(rows[0]),
      len(rows),
      frozenset(walls),
      frozenset(floor),
      frozenset(goals),
      tuple(boxes),
      robots[0],
    )

  @staticmethod
  def center(cell: Cell) -> tuple[float, float]:
    return cell[0] + 0.5, cell[1] + 0.5


@dataclass(frozen=True)
class Action:
  skill: Literal["walk", "push"]
  direction: str
  cells: int = 1

  def __post_init__(self) -> None:
    if self.skill not in ("walk", "push") or self.direction not in DIRECTIONS:
      raise ValueError("Action needs walk/push and an E/W/N/S direction")
    if type(self.cells) is not int or self.cells < 1:
      raise ValueError("Action cells must be a positive integer")


@dataclass(frozen=True)
class Instruction:
  action: Action
  start: Cell
  finish: Cell
  box_index: int | None = None
  box_start: Cell | None = None
  box_finish: Cell | None = None


def compile_plan(board: Board, actions: list[Action]) -> tuple[Instruction, ...]:
  """Check every crossed cell and merge consecutive pushes of the same box"""
  robot, boxes = board.robot, list(board.boxes)
  result: list[Instruction] = []
  for action in actions:
    direction = DIRECTIONS[action.direction]
    start = robot
    box_index = None
    box_start = None
    if action.skill == "push":
      front = advance(robot, direction)
      if front not in boxes:
        raise ValueError("Push has no box in front of the robot")
      box_index = boxes.index(front)
      box_start = front
    for _ in range(action.cells):
      next_robot = advance(robot, direction)
      if next_robot not in board.floor:
        raise ValueError("Plan crosses a wall")
      if box_index is None:
        if next_robot in boxes:
          raise ValueError("Walk action collides with a box")
      else:
        next_box = advance(boxes[box_index], direction)
        if next_box not in board.floor or next_box in boxes:
          raise ValueError("Push destination is blocked")
        boxes[box_index] = next_box
      robot = next_robot
    instruction = Instruction(
      action,
      start,
      robot,
      box_index,
      box_start,
      None if box_index is None else boxes[box_index],
    )
    if (
      result
      and action.skill == "push"
      and result[-1].action.skill == "push"
      and result[-1].box_index == box_index
      and result[-1].action.direction == action.direction
    ):
      assert box_index is not None
      previous = result.pop()
      instruction = Instruction(
        Action("push", action.direction, previous.action.cells + action.cells),
        previous.start,
        robot,
        box_index,
        previous.box_start,
        boxes[box_index],
      )
    result.append(instruction)
  if frozenset(boxes) != board.goals:
    raise ValueError("Plan does not put all boxes on goals")
  return tuple(result)
