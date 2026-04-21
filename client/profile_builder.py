"""
Builds a structured text descriptor for each client each round.
The descriptor is sent to the server and embedded by the LLM module.
No raw data is ever included — only statistics.
"""
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClientProfile:
    client_id: int
    num_classes: int
    history_len: int = 20

    # Rolling history (maintained locally, summary sent to server)
    _loss_history: deque = field(default_factory=lambda: deque(maxlen=20))
    _active_history: deque = field(default_factory=lambda: deque(maxlen=20))
    _grad_norm_history: deque = field(default_factory=lambda: deque(maxlen=20))

    # Latest round stats
    last_loss: float = 0.0
    last_grad_norm: float = 0.0
    last_latency: float = 0.0
    label_counts: Optional[np.ndarray] = None  # shape: (num_classes,)

    def update(self, loss: float, grad_norm: float, latency: float,
               label_counts: np.ndarray, was_active: bool):
        self.last_loss = loss
        self.last_grad_norm = grad_norm
        self.last_latency = latency
        self.label_counts = label_counts
        self._loss_history.append(loss)
        self._active_history.append(int(was_active))
        self._grad_norm_history.append(grad_norm)

    def to_descriptor(self) -> str:
        """Serialise profile to a natural-language string for LLM embedding."""
        active_frac = np.mean(self._active_history) if self._active_history else 0.0
        last_gap = _last_gap(self._active_history)
        loss_trend = _trend(self._loss_history)
        loss_var = float(np.var(list(self._loss_history))) if len(self._loss_history) > 1 else 0.0

        if self.label_counts is not None and self.label_counts.sum() > 0:
            dist = self.label_counts / self.label_counts.sum()
            dist_str = "[" + ", ".join(f"{v:.2f}" for v in dist) + "]"
            entropy = float(-np.sum(dist * np.log(dist + 1e-9)))
        else:
            dist_str = "unknown"
            entropy = 0.0

        return (
            f"Client {self.client_id}: "
            f"active {active_frac:.0%} of last {len(self._active_history)} rounds, "
            f"last_gap={last_gap}, "
            f"loss={self.last_loss:.4f} ({loss_trend} trend, var={loss_var:.4f}), "
            f"class_dist={dist_str}, "
            f"data_entropy={entropy:.3f}, "
            f"grad_norm={self.last_grad_norm:.3f}, "
            f"latency={self.last_latency:.2f}s"
        )

    def get_label_histogram(self) -> np.ndarray:
        if self.label_counts is None:
            return np.zeros(self.num_classes)
        total = self.label_counts.sum()
        return self.label_counts / total if total > 0 else self.label_counts


def _last_gap(active_history) -> int:
    """Number of consecutive inactive rounds at the tail of the history."""
    gap = 0
    for v in reversed(active_history):
        if v == 0:
            gap += 1
        else:
            break
    return gap


def _trend(loss_history) -> str:
    if len(loss_history) < 3:
        return "unknown"
    vals = list(loss_history)
    slope = np.polyfit(range(len(vals)), vals, 1)[0]
    if slope < -0.001:
        return "↓"
    elif slope > 0.001:
        return "↑"
    return "→"