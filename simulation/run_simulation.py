"""
Entry point for Flower simulation.
Usage:
    python simulation/run_simulation.py --config configs/cifar10_crossdevice.yaml
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import flwr as fl
from flwr.common import ndarrays_to_parameters

from datasets.dataset_factory import get_dataset
from client.fl_client import FLClient
from client.models import get_model
from strategy.llm_fed_strategy import LLMFedStrategy


def make_client_fn(cfg: dict, num_classes: int):
    """Returns a closure so cfg is captured — avoids passing through Context."""
    def client_fn(context):
        cid = int(context.node_id)
        train_loader, test_loader, _ = get_dataset(
            name=cfg["dataset"],
            client_id=cid % cfg["num_clients"],
            num_clients=cfg["num_clients"],
            alpha=cfg["alpha"],
            config=cfg,
        )
        model = get_model(cfg["dataset"], num_classes)
        return FLClient(cid, model, train_loader, test_loader, num_classes, cfg).to_client()
    return client_fn


def build_strategy(cfg: dict, num_classes: int, init_params) -> LLMFedStrategy:
    return LLMFedStrategy(
        num_clients_per_round=cfg["clients_per_round"],
        min_fit_clients=cfg["clients_per_round"],
        min_evaluate_clients=max(2, cfg["clients_per_round"] // 5),
        min_available_clients=cfg["clients_per_round"],
        local_epochs=cfg.get("local_epochs", 1),
        lr=cfg.get("lr", 0.01),
        num_classes=num_classes,
        initial_parameters=init_params,
    )


def run(cfg: dict) -> fl.server.History:
    _, _, num_classes = get_dataset(
        name=cfg["dataset"], client_id=0, num_clients=cfg["num_clients"],
        alpha=cfg["alpha"], config=cfg,
    )
    init_model = get_model(cfg["dataset"], num_classes)
    init_params = ndarrays_to_parameters(
        [val.cpu().numpy() for val in init_model.state_dict().values()]
    )

    strategy = build_strategy(cfg, num_classes, init_params)

    history = fl.simulation.start_simulation(
        client_fn=make_client_fn(cfg, num_classes),
        num_clients=cfg["num_clients"],
        config=fl.server.ServerConfig(num_rounds=cfg["num_rounds"]),
        strategy=strategy,
        client_resources={
            "num_cpus": cfg.get("cpus_per_client", 1),
            "num_gpus": cfg.get("gpus_per_client", 0.0),
        },
    )
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(cfg)


if __name__ == "__main__":
    main()
