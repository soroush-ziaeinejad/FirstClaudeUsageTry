"""
Run all FL methods on the same config and produce a comparison table + figure.

Usage:
    python experiments/compare_methods.py --config configs/cifar10_compare.yaml
    python experiments/compare_methods.py --config configs/cifar10_compare.yaml --methods fedavg poc
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.rule import Rule
from rich.panel import Panel
from rich import box

console = Console()

ALL_METHODS = ["fedavg", "fedprox", "poc", "oort", "llmfed"]

METHOD_STYLES = {
    "fedavg":  {"color": "#7f8c8d", "marker": "o", "label": "FedAvg",  "ls": "--"},
    "fedprox": {"color": "#e67e22", "marker": "s", "label": "FedProx", "ls": "-."},
    "poc":     {"color": "#3498db", "marker": "^", "label": "PoC",     "ls": ":"},
    "oort":    {"color": "#9b59b6", "marker": "D", "label": "OORT",    "ls": (0,(3,1,1,1))},
    "llmfed":  {"color": "#2ecc71", "marker": "*", "label": "LLMFed (Ours)", "ls": "-"},
}


def run_one(method: str, cfg: dict) -> list:
    """Run a single method, return history [{round, loss, accuracy}]."""
    import ray
    import flwr as fl
    from flwr.common import ndarrays_to_parameters
    from datasets.dataset_factory import get_dataset
    from client.models import get_model
    from simulation.run_simulation import make_client_fn
    from experiments.run_experiment import get_strategy
    from utils.logger import get_logger

    log = get_logger(method=method, cfg=cfg)

    _, _, num_classes = get_dataset(
        name=cfg["dataset"], client_id=0, num_clients=cfg["num_clients"],
        alpha=cfg["alpha"], config=cfg,
    )
    init_model = get_model(cfg["dataset"], num_classes)
    init_params = ndarrays_to_parameters(
        [val.cpu().numpy() for val in init_model.state_dict().values()]
    )

    strategy = get_strategy(method, cfg, init_params, num_classes, logger=log)
    strategy._num_rounds = cfg["num_rounds"]

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

    # Shutdown Ray between runs to free memory
    if ray.is_initialized():
        ray.shutdown()

    if strategy._history:
        best = max(strategy._history, key=lambda h: h["accuracy"])
        log.experiment_end(best["round"], best["accuracy"],
                           sum(h.get("elapsed", 0) for h in strategy._history))

    return strategy._history


def plot_comparison(results: dict, cfg: dict, save_path: str):
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(
        f"FL Method Comparison — {cfg['dataset'].upper()}  |  "
        f"{cfg['num_clients']} clients, {cfg['clients_per_round']}/round  |  "
        f"α={cfg['alpha']}  |  {cfg['num_rounds']} rounds",
        fontsize=13, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax_loss = fig.add_subplot(gs[0])
    ax_acc  = fig.add_subplot(gs[1])

    for method, history in results.items():
        if not history:
            continue
        st = METHOD_STYLES[method]
        rounds = [h["round"] for h in history]
        losses = [h["loss"] for h in history]
        accs   = [h["accuracy"] * 100 for h in history]

        ax_loss.plot(rounds, losses, marker=st["marker"], color=st["color"],
                     label=st["label"], linewidth=2, markersize=5,
                     linestyle=st["ls"])
        ax_acc.plot(rounds, accs, marker=st["marker"], color=st["color"],
                    label=st["label"], linewidth=2, markersize=5,
                    linestyle=st["ls"])

    for ax, ylabel, title in [
        (ax_loss, "Loss",         "Convergence — Loss"),
        (ax_acc,  "Accuracy (%)", "Convergence — Accuracy"),
    ]:
        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, loc="best")

    ax_acc.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_comparison_table(results: dict, cfg: dict):
    num_rounds = cfg["num_rounds"]

    table = Table(
        title=f"Method Comparison — {cfg['dataset'].upper()} ({num_rounds} rounds)",
        box=box.ROUNDED, show_header=True, header_style="bold magenta",
    )
    table.add_column("Method",        style="bold",   justify="left")
    table.add_column("Best Acc",      style="green",  justify="right")
    table.add_column("Best Round",    style="cyan",   justify="right")
    table.add_column(f"Acc@R{num_rounds//2}", style="yellow", justify="right")
    table.add_column(f"Acc@R{num_rounds}",    style="green",  justify="right")
    table.add_column("Final Loss",    style="red",    justify="right")
    table.add_column("Convergence ▲", justify="right")

    rows = []
    for method, history in results.items():
        if not history:
            continue
        best = max(history, key=lambda h: h["accuracy"])
        mid  = next((h for h in history if h["round"] == num_rounds // 2), history[0])
        last = history[-1]
        gain = last["accuracy"] - history[0]["accuracy"]
        rows.append((method, best, mid, last, gain))

    # Sort by final accuracy descending
    rows.sort(key=lambda r: r[3]["accuracy"], reverse=True)

    for i, (method, best, mid, last, gain) in enumerate(rows):
        st = METHOD_STYLES[method]
        gain_str = f"[green]+{gain*100:.1f}%[/green]" if gain >= 0 else f"[red]{gain*100:.1f}%[/red]"
        rank_prefix = "🥇 " if i == 0 else ("🥈 " if i == 1 else ("🥉 " if i == 2 else "   "))
        table.add_row(
            f"{rank_prefix}{st['label']}",
            f"{best['accuracy']*100:.2f}%  (R{best['round']})",
            str(best["round"]),
            f"{mid['accuracy']*100:.2f}%",
            f"{last['accuracy']*100:.2f}%",
            f"{last['loss']:.4f}",
            gain_str,
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/cifar10_compare.yaml")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS)
    parser.add_argument("--plot",    default="comparison.png")
    parser.add_argument("--results", default="comparison_results.json",
                        help="Save raw results to JSON for later analysis")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    console.print(Rule("[bold blue]FL Method Comparison[/bold blue]"))
    console.print(Panel(
        f"[bold]Dataset[/]  {cfg['dataset'].upper()}\n"
        f"[bold]Clients[/]  {cfg['num_clients']} total, {cfg['clients_per_round']}/round\n"
        f"[bold]Rounds[/]   {cfg['num_rounds']}    α={cfg['alpha']}\n"
        f"[bold]Methods[/]  {', '.join(args.methods)}",
        title="[cyan]Experiment Setup[/cyan]", border_style="cyan", expand=False,
    ))
    console.print()

    results = {}
    total_start = time.time()

    for i, method in enumerate(args.methods):
        console.print(Rule(f"[bold]({i+1}/{len(args.methods)}) Running: {METHOD_STYLES[method]['label']}[/bold]"))
        t0 = time.time()
        try:
            history = run_one(method, cfg)
            elapsed = time.time() - t0
            results[method] = history
            if history:
                best_acc = max(h["accuracy"] for h in history)
                console.print(
                    f"  [green]✓[/green] Done in {elapsed:.0f}s — "
                    f"best acc: [green]{best_acc*100:.2f}%[/green]\n"
                )
        except Exception as e:
            console.print(f"  [red]✗ {method} failed: {e}[/red]\n")
            results[method] = []

    total_elapsed = time.time() - total_start

    # Save raw results
    with open(args.results, "w") as f:
        json.dump({"config": cfg, "results": results}, f, indent=2)

    # Print comparison table
    console.print()
    console.print(Rule("[bold green]Comparison Results[/bold green]"))
    print_comparison_table(results, cfg)

    # Save plot
    plot_comparison(results, cfg, save_path=args.plot)
    console.print(
        f"\n[bold]Plot saved →[/bold] [underline]{os.path.abspath(args.plot)}[/underline]"
    )
    console.print(
        f"[bold]JSON saved →[/bold] [underline]{os.path.abspath(args.results)}[/underline]"
    )
    console.print(f"\nTotal time: {total_elapsed/60:.1f} min\n")
    console.print(Rule())


if __name__ == "__main__":
    main()
