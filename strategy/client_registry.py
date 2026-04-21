"""
Server-side store for client profiles received after each fit() round.
Keeps rolling history per client; never stores raw data.
"""
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ServerClientRecord:
    client_id: str
    num_classes: int
    max_history: int = 20

    loss_history: deque = field(default_factory=lambda: deque(maxlen=20))
    active_history: deque = field(default_factory=lambda: deque(maxlen=20))
    grad_norm_history: deque = field(default_factory=lambda: deque(maxlen=20))
    last_descriptor: str = ""
    last_label_dist: Optional[np.ndarray] = None
    rounds_participated: int = 0
    last_round: int = -1

    def update_from_metrics(self, metrics: dict, current_round: int):
        self.loss_history.append(metrics.get("train_loss", 0.0))
        self.grad_norm_history.append(metrics.get("grad_norm", 0.0))
        self.active_history.append(1)
        self.last_descriptor = metrics.get("descriptor", "")
        self.rounds_participated += 1
        self.last_round = current_round

        # Reconstruct label distribution from per-label metrics
        label_keys = sorted([k for k in metrics if k.startswith("label_")],
                            key=lambda k: int(k.split("_")[1]))
        if label_keys:
            counts = np.array([metrics[k] for k in label_keys], dtype=np.float32)
            total = counts.sum()
            self.last_label_dist = counts / total if total > 0 else counts

    def mark_inactive(self, current_round: int):
        self.active_history.append(0)

    @property
    def last_loss(self) -> float:
        return self.loss_history[-1] if self.loss_history else 0.0

    @property
    def active_fraction(self) -> float:
        return float(np.mean(self.active_history)) if self.active_history else 0.0


class ClientRegistry:
    def __init__(self):
        self._records: Dict[str, ServerClientRecord] = {}

    def update(self, client_id: str, metrics: dict, current_round: int, num_classes: int):
        if client_id not in self._records:
            self._records[client_id] = ServerClientRecord(
                client_id=client_id, num_classes=num_classes
            )
        self._records[client_id].update_from_metrics(metrics, current_round)

    def mark_inactive(self, client_id: str, current_round: int, num_classes: int):
        if client_id not in self._records:
            self._records[client_id] = ServerClientRecord(
                client_id=client_id, num_classes=num_classes
            )
        self._records[client_id].mark_inactive(current_round)

    def get(self, client_id: str) -> Optional[ServerClientRecord]:
        return self._records.get(client_id)

    def all_descriptors(self) -> Dict[str, str]:
        return {cid: r.last_descriptor for cid, r in self._records.items()
                if r.last_descriptor}

    def all_losses(self) -> Dict[str, float]:
        return {cid: r.last_loss for cid, r in self._records.items()}

    def known_clients(self) -> List[str]:
        return list(self._records.keys())