"""Storage the bridging architectures share.

Nothing here knows about skills or bridges: it is a fixed-capacity, overwrite-oldest
buffer of named tensor fields, shared across many parts of the architectures.
"""

from __future__ import annotations

import torch


class RingBuffer:
  """Fixed-capacity, overwrite-oldest storage for named tensor fields.

  Fields may have any trailing shape, so one buffer holds flat rows (an observation),
  windows (`(steps, obs_dim)`) and scalars side by side.
  """

  def __init__(
    self,
    capacity: int,
    device: str,
    shapes: dict[str, tuple[int, ...]],
    dtypes: dict[str, torch.dtype] | None = None,
  ) -> None:
    self.capacity = capacity
    self.device = device
    self._fields = tuple(shapes)
    dtypes = dtypes or {}
    self._data = {
      name: torch.zeros(
        (capacity, *shape), device=device, dtype=dtypes.get(name, torch.float32)
      )
      for name, shape in shapes.items()
    }
    self._size = 0
    self._next = 0

  def __len__(self) -> int:
    return self._size

  @property
  def full(self) -> bool:
    return self._size >= self.capacity

  def add(self, **values: torch.Tensor) -> None:
    """Append a batch of rows, each tensor shaped (N, *field_shape)."""
    n = next(iter(values.values())).shape[0]
    if n == 0:
      return
    idx = (torch.arange(n, device=self.device) + self._next) % self.capacity
    for name in self._fields:
      self._data[name][idx] = values[name].to(self._data[name].dtype)
    self._next = (self._next + n) % self.capacity
    self._size = min(self._size + n, self.capacity)

  def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
    if self._size == 0:
      raise RuntimeError("Cannot sample from an empty buffer")
    idx = torch.randint(0, self._size, (batch_size,), device=self.device)
    return {name: self._data[name][idx] for name in self._fields}

  def all(self) -> dict[str, torch.Tensor]:
    """Every stored row, in insertion order for a buffer that never wrapped."""
    return {name: self._data[name][: self._size] for name in self._fields}
