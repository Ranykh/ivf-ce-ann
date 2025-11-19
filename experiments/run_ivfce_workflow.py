"""CLI to build IVF-CE indices and run decoupled search sweeps."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from config import load_config
from data.dataset_loader import load_dataset
from evaluation import Evaluator
from experiments.utils import select_queries
from src.index import IVFCEIndex, IndexMetadata, available_preset_names, get_preset
from src.search import IVFCEExplorer
from src.utils import setup_logger


class RunLogger:
    """Append-only JSON-lines logger."""

    def __init__(self, path: Path) -> None:
        """Create a logger that appends records to the given file path."""
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict) -> None:
        """Append a serialized JSON payload as a single line."""
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the IVF-CE workflow script."""
    parser = argparse.ArgumentParser(description="IVF-CE index workflow CLI.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default_config.yaml"),
        help="Path to YAML configuration file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preset_choices = available_preset_names()
    build_parser = subparsers.add_parser(
        "build-index", help="Build and persist an IVF-CE index."
    )
    build_parser.add_argument(
        "--preset",
        required=True,
        choices=preset_choices,
        help="Named IVF-CE preset to use (A, B, C, ...).",
    )
    build_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/indexes"),
        help="Directory to write the serialized index into.",
    )
    build_parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional custom filename for the saved index.",
    )
    build_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing index file if it already exists.",
    )

    search_parser = subparsers.add_parser(
        "run-search", help="Run IVF-CE sweeps using a persisted index."
    )
    search_parser.add_argument(
        "--preset",
        required=True,
        choices=preset_choices,
        help="Preset file describing dataset/index/search knobs.",
    )
    search_parser.add_argument(
        "--index-path",
        type=Path,
        required=True,
        help="Path to a serialized IVF-CE index file.",
    )
    search_parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("results/logs/search_runs.jsonl"),
        help="JSON-lines log file to append search run summaries to.",
    )
    search_parser.add_argument(
        "--b-values",
        type=int,
        nargs="+",
        default=None,
        help="Override B values (n1+n2) defined by the search preset.",
    )
    search_parser.add_argument(
        "--k2-values",
        type=int,
        nargs="+",
        default=None,
        help="Override k2 values defined by the search preset.",
    )
    search_parser.add_argument(
        "--query-count",
        type=int,
        help="Override number of queries to evaluate (otherwise preset / config).",
    )
    search_parser.add_argument(
        "--query-seed",
        type=int,
        help="Override query sampling seed (otherwise preset value).",
    )
    return parser.parse_args()


def _default_output_name(dataset_name: str, preset_name: str) -> str:
    """Generate a deterministic filename for a serialized index."""
    timestamp = datetime.utcnow().strftime("%Y%m%d")
    return f"ivfce_{dataset_name}_{preset_name.lower()}_{timestamp}.idx"


def _build_index(args: argparse.Namespace) -> None:
    """Build and persist an IVF-CE index according to the CLI preset."""
    preset = get_preset(args.preset)
    logger = setup_logger("experiments.ivfce_build")
    dataset_cfg = preset.dataset
    clustering_cfg = preset.clustering
    indexing_cfg = preset.indexing

    base, _queries, _ground_truth = load_dataset(dataset_cfg.name, dataset_cfg.path)
    logger.info(
        "Loaded dataset '%s' with %d vectors to build preset %s.",
        dataset_cfg.name,
        base.shape[0],
        preset.name,
    )

    index = IVFCEIndex(
        dimension=base.shape[1],
        n_clusters=clustering_cfg.n_clusters,
        k1=indexing_cfg.k1,
        m_max=indexing_cfg.m_max,
        p_index=indexing_cfg.p_index,
        n_init=clustering_cfg.n_init,
        max_iter=clustering_cfg.max_iter,
        seed=clustering_cfg.seed,
        progress_logger=logger,
    )
    index.build(base)

    if index.build_stats is None:
        raise RuntimeError("Build statistics are missing after index build.")

    metadata = IndexMetadata.create(
        config_name=preset.name,
        dataset_name=dataset_cfg.name,
        dataset_path=dataset_cfg.path,
        dataset_size=base.shape[0],
        n_clusters=clustering_cfg.n_clusters,
        dimension=base.shape[1],
        m_max=indexing_cfg.m_max,
        k1=indexing_cfg.k1,
        p_index=indexing_cfg.p_index,
        seed=clustering_cfg.seed,
        build_stats=index.build_stats,
        extra={"config_path": str(args.config)},
    )
    index.metadata = metadata

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or _default_output_name(dataset_cfg.name, preset.name)
    output_path = output_dir / output_name

    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Index file {output_path} already exists. "
            "Use --force to overwrite."
        )

    index.save(output_path)
    logger.info(
        "Saved IVF-CE index preset %s to %s (build_time_total=%.3fs).",
        preset.name,
        output_path,
        index.build_stats.total,
    )


def _enumerate_n1_n2_pairs(b: int) -> List[Tuple[int, int]]:
    """Yield feasible (n1, n2) splits for a total cluster budget."""
    return [(n1, b - n1) for n1 in range(1, b) if b - n1 > 0]


def _run_search(args: argparse.Namespace) -> None:
    """Execute IVF-CE sweeps for different budgets and log the outcomes."""
    logger = setup_logger("experiments.ivfce_search")
    index = IVFCEIndex.load(args.index_path)
    metadata = index.metadata
    if metadata is None:
        raise RuntimeError("Loaded index does not contain metadata; rebuild it with the build-index command.")
    if index.build_stats is None:
        raise RuntimeError("Loaded index is missing build statistics.")

    preset = get_preset(args.preset)
    cfg = load_config(args.config)
    evaluation_cfg = cfg["evaluation"]

    dataset_cfg = preset.dataset
    _base, queries, ground_truth = load_dataset(dataset_cfg.name, dataset_cfg.path)
    if dataset_cfg.name != metadata.dataset_name:
        logger.warning(
            "Dataset name from preset (%s) differs from index metadata (%s).",
            dataset_cfg.name,
            metadata.dataset_name,
        )
    if metadata.dataset_path and metadata.dataset_path != dataset_cfg.path:
        logger.warning(
            "Dataset path from preset (%s) differs from index metadata (%s).",
            dataset_cfg.path,
            metadata.dataset_path,
        )

    search_cfg = preset.search
    query_count = (
        args.query_count
        or search_cfg.query_count
        or evaluation_cfg["n_queries"]
    )
    query_seed = args.query_seed if args.query_seed is not None else search_cfg.query_seed
    queries_eval, gt_eval = select_queries(
        queries, ground_truth, query_count, seed=query_seed
    )
    evaluator = Evaluator(gt_eval)
    run_logger = RunLogger(args.log_path)
    logger.info(
        "Running sweeps for preset %s with %d queries (seed=%d).",
        preset.name,
        queries_eval.shape[0],
        query_seed,
    )

    b_values = sorted(set(args.b_values or search_cfg.B_values))
    k2_values = sorted(set(args.k2_values or search_cfg.k2_values))
    if any(b < 2 for b in b_values):
        raise ValueError("Each B must be >= 2 to allow n1>0 and n2>0.")
    if any(k2 <= 0 for k2 in k2_values):
        raise ValueError("All k2 values must be positive.")
    k_values = [10, 50, 100]

    for b in b_values:
        pairs = _enumerate_n1_n2_pairs(b)
        for n1, n2 in pairs:
            for k2 in k2_values:
                searcher = IVFCEExplorer(
                    index,
                    n1=n1,
                    n2=n2,
                    k2=k2,
                )
                result = evaluator.evaluate(searcher, queries_eval, k_values)
                payload = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "index_path": str(args.index_path),
                    "config_name": metadata.config_name,
                    "preset_used": preset.name,
                    "dataset_name": metadata.dataset_name,
                    "dataset_path": metadata.dataset_path,
                    "dataset_size": metadata.dataset_size,
                    "n_clusters": metadata.n_clusters,
                    "dimension": metadata.dimension,
                    "m_max": metadata.m_max,
                    "k1": metadata.k1,
                    "p_index": metadata.p_index,
                    "clustering_seed": metadata.seed,
                    "B": b,
                    "n1": n1,
                    "n2": n2,
                    "k2": k2,
                    "query_count": queries_eval.shape[0],
                    "query_seed": query_seed,
                    "build_time_total": index.build_stats.total,
                    "build_time_components": {
                        **index.build_stats.as_dict(),
                    },
                    "avg_query_time_ms": result.query_time_ms,
                    "latency_p50_ms": result.latency_p50_ms,
                    "latency_p95_ms": result.latency_p95_ms,
                    "recall_at_10": result.recalls[10],
                    "recall_at_50": result.recalls[50],
                    "recall_at_100": result.recalls[100],
                }
                run_logger.write(payload)
                logger.info(
                    "Logged run preset=%s B=%d (n1=%d,n2=%d) k2=%d: R@10=%.4f R@50=%.4f R@100=%.4f avg_q=%.3fms",
                    preset.name,
                    b,
                    n1,
                    n2,
                    k2,
                    result.recalls[10],
                    result.recalls[50],
                    result.recalls[100],
                    result.query_time_ms,
                )


def main() -> None:
    """Dispatch CLI commands for building indexes or running sweeps."""
    args = parse_args()
    if args.command == "build-index":
        _build_index(args)
    elif args.command == "run-search":
        _run_search(args)
    else:  # pragma: no cover - handled by argparse
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
