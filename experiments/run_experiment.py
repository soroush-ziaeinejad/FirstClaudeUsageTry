"""
Run a full experiment sweep across configs and methods.
Usage:
    python experiments/run_experiment.py --config configs/cifar10_crossdevice.yaml --method fedavg
    python experiments/run_experiment.py --config configs/cifar10_crossdevice.yaml --method llmfed
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import flwr as fl
from flwr.common import ndarrays_to_parameters

from fl_datasets.dataset_factory import get_dataset
from client.models import get_model
from simulation.run_simulation import make_client_fn

AVAILABLE_METHODS = ["fedavg", "fedprox", "poc", "oort", "llmfed"]


def get_strategy(method: str, cfg: dict, init_params, num_classes: int, logger=None):
    base_kwargs = dict(
        num_clients_per_round=cfg["clients_per_round"],
        min_fit_clients=cfg["clients_per_round"],
        min_evaluate_clients=max(2, cfg["clients_per_round"] // 5),
        min_available_clients=cfg["clients_per_round"],
        local_epochs=cfg.get("local_epochs", 1),
        lr=cfg.get("lr", 0.01),
        num_classes=num_classes,
        initial_parameters=init_params,
        logger=logger,
    )

    if method == "fedavg":
        from strategy.llm_fed_strategy import LLMFedStrategy
        return LLMFedStrategy(**base_kwargs)

    elif method == "fedprox":
        from strategy.fedprox_strategy import FedProxStrategy
        return FedProxStrategy(mu=cfg.get("mu", 0.01), **base_kwargs)

    elif method == "poc":
        from strategy.poc_strategy import PoCStrategy
        return PoCStrategy(oversample_factor=cfg.get("poc_oversample", 2.0), **base_kwargs)

    elif method == "oort":
        from strategy.oort_strategy import OORTStrategy
        return OORTStrategy(
            deadline=cfg.get("oort_deadline", 60.0),
            staleness_lambda=cfg.get("oort_lambda", 0.1),
            **base_kwargs,
        )

    elif method == "llmfed":
        from strategy.llm_cluster_strategy import LLMClusterStrategy
        return LLMClusterStrategy(**base_kwargs)

    else:
        raise ValueError(f"Unknown method '{method}'. Available: {AVAILABLE_METHODS}")


def run(cfg: dict, method: str = "fedavg", plot_path: str = None, logger=None) -> fl.server.History:
    _, _, num_classes = get_dataset(
        name=cfg["dataset"], client_id=0, num_clients=cfg["num_clients"],
        alpha=cfg["alpha"], config=cfg,
    )
    init_model = get_model(cfg["dataset"], num_classes)
    init_params = ndarrays_to_parameters(
        [val.cpu().numpy() for val in init_model.state_dict().values()]
    )

    strategy = get_strategy(method, cfg, init_params, num_classes, logger=logger)
    strategy._num_rounds = cfg["num_rounds"]

    history = fl.simulation.start_simulation(
        client_fn=make_client_fn(cfg, num_classes),
        num_clients=cfg["num_clients"],
        config=fl.server.ServerConfig(num_rounds=cfg["num_rounds"]),
        strategy=strategy,
        client_resources={
            "num_cpus": cfg.get("cpus_per_client", 1),
            "num_gpus": cfg.get("gpus_per_client", 0.0),
        },
        ray_init_args={
            "include_dashboard": False,
            "object_store_memory": cfg.get("ray_object_store_mb", 2048) * 1024 * 1024,
        },
    )
    return history, strategy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", default="fedavg", choices=AVAILABLE_METHODS)
    parser.add_argument("--plot", default=None, help="Save convergence plot to this path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from rich.console import Console
    from rich.rule import Rule
    console = Console()
    console.print(Rule(f"[bold blue]{args.method.upper()} — {cfg['dataset'].upper()}[/bold blue]"))

    history, strategy = run(cfg, method=args.method, plot_path=args.plot)

    if args.plot and strategy._history:
        from test_run import plot_results, print_summary_table
        print_summary_table(strategy._history, cfg)
        plot_results(strategy._history, cfg, save_path=args.plot)
        console.print(f"\nPlot saved → [underline]{os.path.abspath(args.plot)}[/underline]")


if __name__ == "__main__":
    main()
