"""
Run a full experiment sweep across configs and methods.
Usage:
    python experiments/run_experiment.py --config configs/cifar10_crossdevice.yaml --method fedavg
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

AVAILABLE_METHODS = ["fedavg", "fedprox", "poc", "oort", "llmfed"]


def get_strategy(method: str, cfg: dict, init_params, num_classes: int):
    from strategy.llm_fed_strategy import LLMFedStrategy

    base_kwargs = dict(
        num_clients_per_round=cfg["clients_per_round"],
        min_fit_clients=cfg["clients_per_round"],
        min_evaluate_clients=max(2, cfg["clients_per_round"] // 5),
        min_available_clients=cfg["clients_per_round"],
        local_epochs=cfg.get("local_epochs", 1),
        lr=cfg.get("lr", 0.01),
        num_classes=num_classes,
        initial_parameters=init_params,
    )

    if method == "fedavg":
        return LLMFedStrategy(**base_kwargs)
    elif method == "llmfed":
        # Phase 3: will import LLMClusterStrategy once implemented
        try:
            from strategy.llm_cluster_strategy import LLMClusterStrategy
            return LLMClusterStrategy(**base_kwargs)
        except ImportError:
            print("[WARN] LLMClusterStrategy not implemented yet — falling back to FedAvg")
            return LLMFedStrategy(**base_kwargs)
    else:
        raise NotImplementedError(f"Method '{method}' not yet implemented. "
                                  f"Available: {AVAILABLE_METHODS}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", default="fedavg", choices=AVAILABLE_METHODS)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print(f"  Dataset : {cfg['dataset']}")
    print(f"  Method  : {args.method}")
    print(f"  Clients : {cfg['num_clients']}  (per round: {cfg['clients_per_round']})")
    print(f"  Rounds  : {cfg['num_rounds']}   alpha={cfg['alpha']}")
    print(f"{'='*60}\n")

    import flwr as fl
    from flwr.common import ndarrays_to_parameters
    from datasets.dataset_factory import get_dataset
    from client.models import get_model
    from simulation.run_simulation import client_fn

    _, _, num_classes = get_dataset(
        name=cfg["dataset"], client_id=0, num_clients=cfg["num_clients"],
        alpha=cfg["alpha"], config=cfg,
    )
    init_model = get_model(cfg["dataset"], num_classes)
    init_params = ndarrays_to_parameters(
        [val.cpu().numpy() for val in init_model.state_dict().values()]
    )

    strategy = get_strategy(args.method, cfg, init_params, num_classes)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg["num_clients"],
        config=fl.server.ServerConfig(num_rounds=cfg["num_rounds"]),
        strategy=strategy,
        client_resources={"num_cpus": cfg.get("cpus_per_client", 1),
                          "num_gpus": cfg.get("gpus_per_client", 0.0)},
        run_config=cfg,
    )


if __name__ == "__main__":
    main()
