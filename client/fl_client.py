import time
import copy
import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from flwr.common import NDArrays, Scalar
from typing import Dict, List, Optional, Tuple

from client.profile_builder import ClientProfile
from client.device_utils import get_device, clear_device_cache


class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, model: nn.Module, train_loader, test_loader,
                 num_classes: int, config: dict):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.device = get_device()
        self.model.to(self.device)
        self.profile = ClientProfile(client_id=client_id, num_classes=num_classes)
        self._round = 0
        self._label_counts: Optional[np.ndarray] = None  # cached; recomputed once

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------

    def get_parameters(self, config: Dict) -> NDArrays:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: NDArrays):
        state_dict = self.model.state_dict()
        for key, val in zip(state_dict.keys(), parameters):
            # Create on CPU first then move — torch.tensor(device='mps') is unreliable
            t = torch.as_tensor(val).to(dtype=state_dict[key].dtype).to(self.device)
            state_dict[key] = t
        self.model.load_state_dict(state_dict, strict=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
        self.set_parameters(parameters)
        self._round = int(config.get("server_round", self._round + 1))

        lr = float(config.get("lr", 0.01))
        epochs = int(config.get("local_epochs", 1))
        mu = float(config.get("mu", 0.0))  # FedProx proximal term

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # Keep a frozen copy of global params for FedProx proximal term
        global_params: Optional[List[torch.Tensor]] = None
        if mu > 0:
            global_params = [p.data.clone().detach() for p in self.model.parameters()]

        self.model.train()
        total_loss, num_samples = 0.0, 0
        t0 = time.time()

        for _ in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(x), y)

                # FedProx proximal term: (mu/2) * ||w - w_global||^2
                if mu > 0 and global_params is not None:
                    prox = sum(
                        ((p - g) ** 2).sum()
                        for p, g in zip(self.model.parameters(), global_params)
                    )
                    loss = loss + (mu / 2.0) * prox

                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y)
                num_samples += len(y)

        latency = time.time() - t0
        avg_loss = total_loss / max(num_samples, 1)

        grad_norm = float(sum(
            p.grad.norm().item() ** 2
            for p in self.model.parameters() if p.grad is not None
        ) ** 0.5)

        # Compute label counts once and cache them
        if self._label_counts is None:
            self._label_counts = self._compute_label_counts()

        self.profile.update(avg_loss, grad_norm, latency, self._label_counts, was_active=True)

        clear_device_cache(self.device)

        metrics = {
            "train_loss": avg_loss,
            "grad_norm": grad_norm,
            "latency": latency,
            "descriptor": self.profile.to_descriptor(),
            **{f"label_{i}": float(v) for i, v in enumerate(self._label_counts)},
        }
        return self.get_parameters(config={}), num_samples, metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        total_loss, correct, num_samples = 0.0, 0, 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                total_loss += criterion(logits, y).item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                num_samples += len(y)

        clear_device_cache(self.device)
        accuracy = correct / max(num_samples, 1)
        return total_loss / max(num_samples, 1), num_samples, {"accuracy": accuracy}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_label_counts(self) -> np.ndarray:
        counts = np.zeros(self.num_classes, dtype=np.float32)
        for _, y in self.train_loader:
            for label in y.numpy().flatten():
                if 0 <= int(label) < self.num_classes:
                    counts[int(label)] += 1
        return counts
