import time
import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from flwr.common import NDArrays, Scalar
from typing import Dict, Tuple
from client.profile_builder import ClientProfile


class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, model: nn.Module, train_loader, test_loader,
                 num_classes: int, config: dict):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.profile = ClientProfile(client_id=client_id, num_classes=num_classes)
        self._round = 0

    def get_parameters(self, config: Dict) -> NDArrays:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: NDArrays):
        state_dict = self.model.state_dict()
        for key, val in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(val)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
        self.set_parameters(parameters)
        self._round = int(config.get("server_round", self._round + 1))

        lr = float(config.get("lr", 0.01))
        epochs = int(config.get("local_epochs", 1))
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        total_loss, num_samples = 0.0, 0
        grad_norm = 0.0
        t0 = time.time()

        for _ in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y)
                num_samples += len(y)

        latency = time.time() - t0
        avg_loss = total_loss / max(num_samples, 1)

        # Compute gradient norm from last backward pass
        grad_norm = float(sum(
            p.grad.norm().item() ** 2 for p in self.model.parameters() if p.grad is not None
        ) ** 0.5)

        label_counts = self._compute_label_counts()
        self.profile.update(avg_loss, grad_norm, latency, label_counts, was_active=True)

        metrics = {
            "train_loss": avg_loss,
            "grad_norm": grad_norm,
            "latency": latency,
            "descriptor": self.profile.to_descriptor(),
            **{f"label_{i}": float(v) for i, v in enumerate(label_counts)},
        }
        return self.get_parameters(config={}), num_samples, metrics

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

        accuracy = correct / max(num_samples, 1)
        avg_loss = total_loss / max(num_samples, 1)
        return avg_loss, num_samples, {"accuracy": accuracy}

    def _compute_label_counts(self) -> np.ndarray:
        counts = np.zeros(self.num_classes, dtype=np.float32)
        for _, y in self.train_loader:
            for label in y.numpy().flatten():
                if 0 <= int(label) < self.num_classes:
                    counts[int(label)] += 1
        return counts