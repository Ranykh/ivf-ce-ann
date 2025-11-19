"""Run parameter ablations for IVF-CE."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from config import load_config
from data.dataset_loader import load_dataset
from evaluation import Evaluator
from experiments.utils import format_results, select_queries
from src.index import IVFCEIndex
from src.search import IVFCEExplorer
from src.utils import setup_logger


DEFAULT_ABLATIONS = {
    "k1": [10, 20, 30, 50],
    "m_max": [4, 8, 16, 32],
    "n1": [1, 3, 5, 10],
    "n2": [5, 10, 15, 20],
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the ablation script."""
    parser = argparse.ArgumentParser(description="Run IVF-CE ablation study.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default_config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--parameter",
        choices=DEFAULT_ABLATIONS.keys(),
        help="Single parameter to sweep. If omitted, sweeps all.",
    )
    return parser.parse_args()


def build_index(cfg: Dict[str, Any], database: np.ndarray) -> IVFCEIndex:
    """Construct an IVF-CE index from configuration and data."""
    clustering_cfg = cfg["clustering"]
    indexing_cfg = cfg["indexing"]
    index = IVFCEIndex(
        dimension=database.shape[1],
        n_clusters=clustering_cfg["n_clusters"],
        k1=indexing_cfg["k1"],
        m_max=indexing_cfg["m_max"],
        p_index=indexing_cfg["p_index"],
        n_init=clustering_cfg.get("n_init", 10),
        max_iter=clustering_cfg.get("max_iter", 100),
        seed=clustering_cfg.get("seed"),
    )
    index.build(database)
    return index


def evaluate_configuration(
    cfg: Dict[str, Any],
    database: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k_values: Iterable[int],
) -> str:
    """Evaluate a specific configuration sweep value and format metrics."""
    ivf_ce_index = build_index(cfg, database)
    query_cfg = cfg["query"]
    searcher = IVFCEExplorer(
        ivf_ce_index,
        n1=query_cfg["n1"],
        n2=query_cfg["n2"],
        k2=query_cfg["k2"],
    )
    evaluator = Evaluator(ground_truth)
    result = evaluator.evaluate(searcher, queries, k_values)
    label = (
        f"k1={cfg['indexing']['k1']}, m_max={cfg['indexing']['m_max']}, "
        f"n1={query_cfg['n1']}, n2={query_cfg['n2']}"
    )
    return format_results(label, result)


def run_ablation(cfg: Dict[str, Any], parameter: str | None) -> None:
    """Sweep one or more parameters and log the resulting metrics."""
    logger = setup_logger("experiments.ablation")
    dataset_cfg = cfg["dataset"]
    evaluation_cfg = cfg["evaluation"]

    base, queries, ground_truth = load_dataset(dataset_cfg["name"], dataset_cfg["path"])
    queries_eval, gt_eval = select_queries(
        queries, ground_truth, evaluation_cfg["n_queries"]
    )

    parameters = [parameter] if parameter else DEFAULT_ABLATIONS.keys()
    for param in parameters:
        logger.info("Sweeping parameter %s", param)
        for value in DEFAULT_ABLATIONS[param]:
            cfg_mod = copy.deepcopy(cfg)
            if param in cfg_mod["indexing"]:
                cfg_mod["indexing"][param] = value
            elif param in cfg_mod["query"]:
                cfg_mod["query"][param] = value
            else:
                logger.warning("Parameter %s not found in config; skipping.", param)
                continue

            summary = evaluate_configuration(
                cfg_mod,
                base,
                queries_eval,
                gt_eval,
                evaluation_cfg["recall_at_k"],
            )
            logger.info("  %s -> %s", param, summary)


def main() -> None:
    """Entry point for invoking the ablation sweep from the CLI."""
    args = parse_args()
    cfg = load_config(args.config)
    run_ablation(cfg, args.parameter)


if __name__ == "__main__":
    main()
