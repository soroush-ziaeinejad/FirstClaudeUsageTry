"""
Experiment logger — creates a dated log file under logs/ and mirrors
important messages to the rich console.

Usage:
    from utils.logger import get_logger
    log = get_logger()           # call once per experiment
    log.info("Round 1 started")
    log.round(1, loss=0.42, accuracy=0.75, elapsed=58.3)
    log.selection("llmfed", selected_cids=["3","7","12"], server_round=1)
"""
import logging
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from rich.logging import RichHandler
from rich.console import Console

_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_console = Console()
_logger_registry: Dict[str, "ExperimentLogger"] = {}


class ExperimentLogger:
    def __init__(self, name: str, log_path: Path):
        self.log_path = log_path
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        if not self._logger.handlers:
            # File handler — plain text, full detail
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            self._logger.addHandler(fh)

            # Console handler — rich, INFO+ only
            rh = RichHandler(console=_console, show_path=False,
                             rich_tracebacks=True, markup=True)
            rh.setLevel(logging.INFO)
            rh.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(rh)

        self._logger.info(f"Log file: {log_path}")

    # ------------------------------------------------------------------
    # Standard log levels
    # ------------------------------------------------------------------

    def info(self, msg: str):
        self._logger.info(msg)

    def debug(self, msg: str):
        self._logger.debug(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    # ------------------------------------------------------------------
    # Structured experiment events
    # ------------------------------------------------------------------

    def experiment_start(self, method: str, cfg: dict):
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = f"cuda:{torch.cuda.get_device_name(0)}"
        else:
            device = "cpu"

        self._logger.info(
            f"=== EXPERIMENT START | method={method} dataset={cfg.get('dataset')} "
            f"clients={cfg.get('num_clients')} rounds={cfg.get('num_rounds')} "
            f"alpha={cfg.get('alpha')} lr={cfg.get('lr')} device={device} ==="
        )
        self._logger.debug(f"Full config: {json.dumps(cfg, indent=2)}")

    def experiment_end(self, best_round: int, best_accuracy: float, total_time: float):
        self._logger.info(
            f"=== EXPERIMENT END | best_acc={best_accuracy:.4f} ({best_accuracy*100:.2f}%) "
            f"at round {best_round} | total_time={total_time:.1f}s ==="
        )

    def round_start(self, server_round: int, num_available: int):
        self._logger.debug(
            f"[Round {server_round}] START — {num_available} clients available"
        )

    def round_end(self, server_round: int, loss: float, accuracy: float,
                  elapsed: float, num_trained: int, failures: int = 0):
        self._logger.info(
            f"[Round {server_round:>3}] loss={loss:.4f}  acc={accuracy:.4f} "
            f"({accuracy*100:.2f}%)  trained={num_trained}  failures={failures}  "
            f"time={elapsed:.1f}s"
        )

    def selection(self, method: str, selected_cids: List[str],
                  server_round: int, strategy_info: Optional[str] = None):
        cid_str = ", ".join(str(c) for c in selected_cids[:10])
        extra = f"  [{strategy_info}]" if strategy_info else ""
        self._logger.debug(
            f"[Round {server_round}] SELECT ({method}) → [{cid_str}]{extra} "
            f"({len(selected_cids)} clients)"
        )

    def client_fit(self, cid: str, loss: float, grad_norm: float,
                   latency: float, num_samples: int):
        self._logger.debug(
            f"  client={cid}  loss={loss:.4f}  grad_norm={grad_norm:.3f}  "
            f"latency={latency:.2f}s  samples={num_samples}"
        )

    def embedding_info(self, num_embedded: int, elapsed: float):
        self._logger.debug(
            f"[Embedding] Embedded {num_embedded} clients in {elapsed:.2f}s"
        )

    def clustering_info(self, num_clusters: int, cluster_sizes: List[int]):
        self._logger.debug(
            f"[Clustering] k={num_clusters}  sizes={cluster_sizes}"
        )


def get_logger(method: str = "experiment", cfg: Optional[dict] = None) -> ExperimentLogger:
    """
    Get (or create) the logger for this experiment run.
    Log file is named  logs/YYYY-MM-DD_HH-MM-SS_<method>.log
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    key = f"{timestamp}_{method}"

    if key not in _logger_registry:
        log_path = _LOG_DIR / f"{key}.log"
        _logger_registry[key] = ExperimentLogger(name=key, log_path=log_path)
        if cfg:
            _logger_registry[key].experiment_start(method=method, cfg=cfg)

    return _logger_registry[key]
