"""
End-to-end test runner with rich terminal output and matplotlib plots.
Usage:
    python test_run.py --config configs/cifar10_test.yaml
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.rule import Rule

console = Console()


def plot_results(history, cfg: dict, save_path: str = "results.png"):
    rounds = [h["round"] for h in history]
    losses = [h["loss"] for h in history]
    accs = [h["accuracy"] for h in history]

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(
        f"FL Training — {cfg['dataset'].upper()}  |  "
        f"{cfg['num_clients']} clients, {cfg['clients_per_round']}/round  |  "
        f"α={cfg['alpha']}",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # Loss curve
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(rounds, losses, "o-", color="#e74c3c", linewidth=2, markersize=5, label="Eval Loss")
    ax1.fill_between(rounds, losses, alpha=0.1, color="#e74c3c")
    ax1.set_xlabel("Round", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Convergence — Loss", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Accuracy curve
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(rounds, [a * 100 for a in accs], "s-", color="#2ecc71",
             linewidth=2, markersize=5, label="Eval Accuracy")
    ax2.fill_between(rounds, [a * 100 for a in accs], alpha=0.1, color="#2ecc71")
    ax2.set_xlabel("Round", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title("Convergence — Accuracy", fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def print_summary_table(history, cfg: dict):
    table = Table(title="Training Summary", box=box.ROUNDED, show_header=True,
                  header_style="bold magenta")
    table.add_column("Round", style="cyan", justify="right")
    table.add_column("Loss", style="yellow", justify="right")
    table.add_column("Accuracy", style="green", justify="right")
    table.add_column("Δ Accuracy", justify="right")

    for i, h in enumerate(history):
        delta = ""
        if i > 0:
            diff = h["accuracy"] - history[i - 1]["accuracy"]
            color = "green" if diff >= 0 else "red"
            delta = f"[{color}]{diff:+.4f}[/{color}]"
        table.add_row(
            str(h["round"]),
            f"{h['loss']:.4f}",
            f"{h['accuracy']:.4f}  ({h['accuracy']*100:.1f}%)",
            delta,
        )
    console.print(table)


def print_config_panel(cfg: dict):
    lines = [
        f"[bold]Dataset[/]        {cfg['dataset'].upper()}",
        f"[bold]Clients[/]        {cfg['num_clients']} total, {cfg['clients_per_round']} per round",
        f"[bold]Rounds[/]         {cfg['num_rounds']}",
        f"[bold]Non-IID α[/]      {cfg['alpha']}",
        f"[bold]Local epochs[/]   {cfg.get('local_epochs', 1)}",
        f"[bold]LR[/]             {cfg.get('lr', 0.01)}",
        f"[bold]Batch size[/]     {cfg.get('batch_size', 32)}",
    ]
    console.print(Panel("\n".join(lines), title="[bold cyan]Experiment Config[/]",
                        border_style="cyan", expand=False))


def print_device_info():
    import torch
    if torch.backends.mps.is_available():
        dev = "[bold green]MPS (Apple Silicon GPU)[/bold green]"
    elif torch.cuda.is_available():
        dev = f"[bold green]CUDA ({torch.cuda.get_device_name(0)})[/bold green]"
    else:
        dev = "[yellow]CPU[/yellow]"
    console.print(f"  Device: {dev}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_test.yaml")
    parser.add_argument("--method", default="fedavg",
                        choices=["fedavg", "fedprox", "poc", "oort", "llmfed"])
    parser.add_argument("--plot", default="results.png", help="Where to save the plot")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from utils.logger import get_logger
    log = get_logger(method=args.method, cfg=cfg)

    console.print(Rule(f"[bold blue]FL Experiment — {args.method.upper()}[/bold blue]"))
    print_config_panel(cfg)
    print_device_info()
    console.print(f"  [dim]Log → {log.log_path}[/dim]\n")
    console.print("[dim]Starting Flower simulation...[/dim]\n")

    from experiments.run_experiment import get_strategy
    from datasets.dataset_factory import get_dataset
    from client.models import get_model
    from simulation.run_simulation import make_client_fn
    from flwr.common import ndarrays_to_parameters
    import flwr as fl
    import time as _time

    _, _, num_classes = get_dataset(
        name=cfg["dataset"], client_id=0, num_clients=cfg["num_clients"],
        alpha=cfg["alpha"], config=cfg,
    )
    init_model = get_model(cfg["dataset"], num_classes)
    init_params = ndarrays_to_parameters(
        [val.cpu().numpy() for val in init_model.state_dict().values()]
    )

    strategy = get_strategy(args.method, cfg, init_params, num_classes, logger=log)
    strategy._num_rounds = cfg["num_rounds"]

    t_start = _time.time()
    fl.simulation.start_simulation(
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
    total_time = _time.time() - t_start

    console.print()
    console.print(Rule("[bold green]Results[/bold green]"))
    print_summary_table(strategy._history, cfg)

    if strategy._history:
        best = max(strategy._history, key=lambda h: h["accuracy"])
        log.experiment_end(best["round"], best["accuracy"], total_time)
        console.print(
            f"\n[bold]Best accuracy:[/bold] [green]{best['accuracy']*100:.2f}%[/green] "
            f"at round [cyan]{best['round']}[/cyan]\n"
        )
        plot_path = plot_results(strategy._history, cfg, save_path=args.plot)
        log.info(f"Plot saved → {os.path.abspath(plot_path)}")
        console.print(f"[bold]Plot saved →[/bold] [underline]{os.path.abspath(plot_path)}[/underline]")
        console.print(f"[bold]Log  saved →[/bold] [underline]{log.log_path}[/underline]\n")

    console.print(Rule())


if __name__ == "__main__":
    main()
