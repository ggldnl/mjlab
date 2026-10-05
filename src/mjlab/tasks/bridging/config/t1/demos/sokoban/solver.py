"""Adapter for the Unified Planning encoding in axioms_usage.ipynb."""

from __future__ import annotations

import importlib
import re
from collections import deque
from typing import Any

from mjlab.tasks.bridging.config.t1.demos.sokoban.board import (
  DIRECTIONS,
  Action,
  Board,
  Cell,
  advance,
)


def _walk_path(
  board: Board, robot: Cell, target: Cell, boxes: set[Cell]
) -> list[Action]:
  queue = deque([robot])
  previous: dict[Cell, tuple[Cell, str] | None] = {robot: None}
  while queue and target not in previous:
    cell = queue.popleft()
    for name, direction in DIRECTIONS.items():
      neighbor = advance(cell, direction)
      if neighbor in board.floor and neighbor not in boxes and neighbor not in previous:
        previous[neighbor] = cell, name
        queue.append(neighbor)
  if target not in previous:
    raise ValueError("Solver push face is not reachable")
  result = []
  while target != robot:
    edge = previous[target]
    assert edge is not None
    target, direction = edge
    result.append(Action("walk", direction))
  return result[::-1]


def translate_plan(board: Board, text: str) -> list[Action]:
  """Accept printed move and push-box plans, including implicit axiom walks"""
  robot, boxes = board.robot, set(board.boxes)
  actions = []
  for line in text.splitlines():
    match = re.search(r"\b(move|push-box)\b", line)
    if match is None:
      if "loc-" in line:
        raise ValueError(f"Unknown solver action: {line}")
      continue
    locations = [
      (int(x), board.height - 1 - int(y))
      for x, y in re.findall(r"loc-(\d+)-(\d+)", line)
    ]
    if match[1] == "move":
      if len(locations) != 2 or locations[0] != robot:
        raise ValueError("Invalid solver move")
      start, finish = locations
    else:
      if len(locations) not in (3, 4):
        raise ValueError("Invalid solver push")
      if len(locations) == 4 and locations[0] != robot:
        raise ValueError("Axiom push starts from the wrong player location")
      start, box, finish = locations[-3:]
      if box not in boxes:
        raise ValueError("Solver push refers to a missing box")
      actions.extend(_walk_path(board, robot, start, boxes))
      direction = (box[0] - start[0], box[1] - start[1])
      if advance(box, direction) != finish:
        raise ValueError("Solver push is not straight")
      boxes.remove(box)
      boxes.add(finish)
      finish = box
    direction = (finish[0] - start[0], finish[1] - start[1])
    name = next(
      (name for name, vector in DIRECTIONS.items() if vector == direction), None
    )
    if name is None:
      raise ValueError("Solver action does not connect adjacent cells")
    actions.append(Action("walk" if match[1] == "move" else "push", name))
    robot = finish
  return actions


def create_problem(board: Board) -> Any:
  """The notebook's explicit move encoding, without its experimental axioms"""
  try:
    up = importlib.import_module("unified_planning.shortcuts")
  except ModuleNotFoundError as error:
    raise RuntimeError(
      "Install unified-planning and a planner engine, or supply an exported plan"
    ) from error
  problem = up.Problem("sokoban")
  location = up.UserType("location")
  player = up.Fluent("has_player", up.BoolType(), l=location)
  box = up.Fluent("has_box", up.BoolType(), l=location)
  adjacent = up.Fluent("adjacent", up.BoolType(), l1=location, l2=location)
  straight = up.Fluent("adjacent_2", up.BoolType(), l1=location, l2=location)
  for fluent in (player, box, adjacent, straight):
    problem.add_fluent(fluent, default_initial_value=False)
  move = up.InstantaneousAction("move", fr=location, to=location)
  fr, to = move.parameters
  for condition in (adjacent(fr, to), player(fr), up.Not(box(to))):
    move.add_precondition(condition)
  move.add_effect(player(fr), False)
  move.add_effect(player(to), True)
  push = up.InstantaneousAction("push-box", x=location, y=location, z=location)
  x, y, z = push.parameters
  for condition in (
    adjacent(x, y),
    adjacent(y, z),
    straight(x, z),
    player(x),
    box(y),
    up.Not(box(z)),
  ):
    push.add_precondition(condition)
  for fluent, value in (
    (player(x), False),
    (player(y), True),
    (box(y), False),
    (box(z), True),
  ):
    push.add_effect(fluent, value)
  problem.add_action(move)
  problem.add_action(push)
  problem.add_quality_metric(up.MinimizeActionCosts({move: 0, push: 1}))
  objects = {
    cell: up.Object(f"loc-{cell[0]}-{board.height - 1 - cell[1]}", location)
    for cell in sorted(board.floor)
  }
  problem.add_objects(list(objects.values()))
  problem.set_initial_value(player(objects[board.robot]), True)
  for cell in board.boxes:
    problem.set_initial_value(box(objects[cell]), True)
  for cell in board.goals:
    problem.add_goal(box(objects[cell]))
  for cell, obj in objects.items():
    for direction in DIRECTIONS.values():
      neighbor = advance(cell, direction)
      if neighbor in objects:
        problem.set_initial_value(adjacent(obj, objects[neighbor]), True)
      farther = advance(cell, direction, 2)
      if farther in objects:
        problem.set_initial_value(straight(obj, objects[farther]), True)
  return problem


def solve(board: Board) -> list[Action]:
  problem = create_problem(board)
  up = importlib.import_module("unified_planning.shortcuts")
  with up.OneshotPlanner(problem_kind=problem.kind) as planner:
    result = planner.solve(problem)
  if result.plan is None:
    raise RuntimeError(f"Solver did not produce a plan: {result.status}")
  return translate_plan(board, str(result.plan))
