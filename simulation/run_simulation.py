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


def client_fn(context):
    cid = int(context.node_config["partition-id"])
    cfg = context.run_config

    train_loader, test_loader, num_classes = get_dataset(
        name=cfg["dataset"],
        client_id=cid,
        num_clients=cfg["num_clients"],
        alpha=cfg["alpha"],
        config=cfg,
    )
    model = get_model(cfg["dataset"], num_classes)
    return FLClient(cid, model, train_loader, test_loader, num_classes, cfg).to_client()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Build initial parameters from a fresh model
    _, _, num_classes = get_dataset(
        name=cfg["dataset"], client_id=0, num_clients=cfg["num_clients"],
        alpha=cfg["alpha"], config=cfg,
    )
    init_model = get_model(cfg["dataset"], num_classes)
    init_params = ndarrays_to_parameters(
        [val.cpu().numpy() for val in init_model.state_dict().values()]
    )

    strategy = LLMFedStrategy(
        num_clients_per_round=cfg["clients_per_round"],
        min_fit_clients=cfg["clients_per_round"],
        min_evaluate_clients=max(2, cfg["clients_per_round"] // 5),
        min_available_clients=cfg["clients_per_round"],
        local_epochs=cfg.get("local_epochs", 1),
        lr=cfg.get("lr", 0.01),
        num_classes=num_classes,
        initial_parameters=init_params,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg["num_clients"],
        config=fl.server.ServerConfig(num_rounds=cfg["num_rounds"]),
        strategy=strategy,
        client_resources={"num_cpus": cfg.get("cpus_per_client", 1),
                          "num_gpus": cfg.get("gpus_per_client", 0.0)},
        actor_kwargs={"on_actor_init_fn": lambda: None},
        run_config=cfg,
    )


if __name__ == "__main__":
    main()
