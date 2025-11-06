"""Run baseline comparisons between IVF, IVF-CE, FAISS, and brute-force."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from time import perf_counter

from config import load_config
from data.dataset_loader import load_dataset
from evaluation import Evaluator
from experiments.utils import format_results, select_queries
from src.baselines import BruteForceSearch, FAISSIVFBaseline
from src.index import IVFCEIndex, IVFIndex
from src.search import IVFCEExplorer, IVFSearcher
from src.utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IVF/IVF-CE baseline comparison.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default_config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--subset-base",
        type=int,
        default=None,
        help="Use only the first N database vectors (recomputes ground truth).",
    )
    parser.add_argument(
        "--subset-queries",
        type=int,
        default=None,
        help="Use only the first M queries for evaluation.",
    )
    return parser.parse_args()


def build_ivf_index(cfg: Dict[str, Any], database: np.ndarray) -> IVFIndex:
    clustering_cfg = cfg["clustering"]
    index = IVFIndex(
        dimension=database.shape[1],
        n_clusters=clustering_cfg["n_clusters"],
        n_init=clustering_cfg.get("n_init", 10),
        max_iter=clustering_cfg.get("max_iter", 100),
        seed=clustering_cfg.get("seed"),
    )
    index.build(database)
    return index


def build_ivf_ce_index(cfg: Dict[str, Any], database: np.ndarray) -> IVFCEIndex:
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


def _compute_ground_truth_with_bruteforce(
    searcher: BruteForceSearch, queries: np.ndarray, k: int
) -> np.ndarray:
    if queries.shape[0] == 0:
        return np.empty((0, k), dtype=np.int32)

    ground_truth = np.empty((queries.shape[0], k), dtype=np.int32)
    ground_truth.fill(-1)
    for idx, query in enumerate(queries):
        ids, _ = searcher.search(query, k)
        limit = min(ids.shape[0], k)
        ground_truth[idx, :limit] = ids[:limit]
    return ground_truth


def evaluate_methods(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    logger = setup_logger("experiments")
    dataset_cfg = cfg["dataset"]
    evaluation_cfg = cfg["evaluation"]
    query_cfg = cfg["query"]

    base, queries, ground_truth = load_dataset(dataset_cfg["name"], dataset_cfg["path"])

    if args.subset_base is not None:
        base = base[: args.subset_base]
    if args.subset_queries is not None:
        queries = queries[: args.subset_queries]
        ground_truth = ground_truth[: args.subset_queries]
    else:
        ground_truth = ground_truth[: queries.shape[0]]

    logger.info(
        "Using %d base vectors and %d queries for evaluation.",
        base.shape[0],
        queries.shape[0],
    )

    bf = BruteForceSearch()
    start = perf_counter()
    bf.build(base)
    bf_build_time = perf_counter() - start
    logger.info("BruteForce build time: %.3f s", bf_build_time)

    if args.subset_base is not None:
        gt_k = ground_truth.shape[1] if ground_truth.size else 0
        if gt_k == 0:
            gt_k = min(100, base.shape[0])
        else:
            gt_k = min(gt_k, base.shape[0])
        ground_truth = _compute_ground_truth_with_bruteforce(bf, queries, gt_k)
        logger.info("Recomputed ground truth for subset (k=%d).", gt_k)

    queries_eval, ground_truth_eval = select_queries(
        queries, ground_truth, evaluation_cfg["n_queries"]
    )
    evaluator = Evaluator(ground_truth_eval)

    results: List[str] = []

    # Brute-force
    bf_result = evaluator.evaluate(bf, queries_eval, evaluation_cfg["recall_at_k"])
    results.append(format_results("BruteForce", bf_result))
    logger.info(results[-1])

    # Our IVF baseline
    start = perf_counter()
    ivf_index = build_ivf_index(cfg, base)
    ivf_build_time = perf_counter() - start
    logger.info("IVF build time: %.3f s", ivf_build_time)
    nprobe = query_cfg.get("nprobe", query_cfg["n1"] + query_cfg["n2"])
    ivf_searcher = IVFSearcher(ivf_index, nprobe=nprobe)
    ivf_result = evaluator.evaluate(ivf_searcher, queries_eval, evaluation_cfg["recall_at_k"])
    results.append(format_results(f"IVF (nprobe={nprobe})", ivf_result))
    logger.info(results[-1])

    # IVF-CE
    start = perf_counter()
    ivf_ce_index = build_ivf_ce_index(cfg, base)
    ivf_ce_build_time = perf_counter() - start
    logger.info("IVF-CE build time: %.3f s", ivf_ce_build_time)
    ivf_ce_searcher = IVFCEExplorer(
        ivf_ce_index,
        n1=query_cfg["n1"],
        n2=query_cfg["n2"],
        k2=query_cfg["k2"],
    )
    ivf_ce_result = evaluator.evaluate(
        ivf_ce_searcher, queries_eval, evaluation_cfg["recall_at_k"]
    )
    results.append(format_results("IVF-CE", ivf_ce_result))
    logger.info(results[-1])

    # FAISS baseline (optional)
    try:
        faiss_baseline = FAISSIVFBaseline(
            base.shape[1],
            nlist=cfg["clustering"]["n_clusters"],
            nprobe=nprobe,
        )
        start = perf_counter()
        faiss_baseline.build(base)
        faiss_build_time = perf_counter() - start
        logger.info("FAISS IVF build time: %.3f s", faiss_build_time)
        faiss_result = evaluator.evaluate(
            faiss_baseline, queries_eval, evaluation_cfg["recall_at_k"]
        )
        results.append(format_results("FAISS IVF", faiss_result))
        logger.info(results[-1])
    except ImportError:
        logger.warning("faiss is not installed; skipping FAISS baseline.")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    evaluate_methods(cfg, args)


if __name__ == "__main__":
    main()
