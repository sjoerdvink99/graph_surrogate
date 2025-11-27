#!/usr/bin/env python3
"""
Run training, evaluation, and latency benchmarks across multiple datasets.

Output structure:
    experiments/
    └── {run_id}/                    # e.g., "2024-01-15_gpu" or "run_001"
        ├── config.json              # Experiment configuration
        ├── results_summary.json     # Combined results
        ├── {dataset}/               # Per-dataset outputs
        │   ├── model.pt
        │   ├── model_info.json
        │   ├── training_history.json
        │   ├── encoder_config.json
        │   ├── graph.gml
        │   ├── evaluation_results.json
        │   ├── latency_benchmark.json
        │   └── figures/             # Dataset-specific figures
        │       ├── training_curves.pdf
        │       ├── latency_comparison.pdf
        │       └── ...
        └── figures/                 # Cross-dataset comparison figures
            └── comparison.pdf
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


DATASETS = ["bron", "github", "deezer", "facebook", "amazon", "dblp", "youtube", "reddit"]


def run_command(cmd: list[str], description: str, capture_output: bool = False) -> tuple[bool, str]:
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    if capture_output:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout + result.stderr
    else:
        result = subprocess.run(cmd)
        return result.returncode == 0, ""


def train_dataset(
    dataset: str,
    output_dir: Path,
    num_train: int = 100000,
    num_val: int = 10000,
    num_test: int = 10000,
    epochs: int = 200,
    batch_size: int = 256,
    streaming: bool = False,
) -> bool:
    cmd = [
        sys.executable, "-m", "training.train",
        "--dataset", dataset,
        "--output-dir", str(output_dir),
        "--num-train", str(num_train),
        "--num-val", str(num_val),
        "--num-test", str(num_test),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]

    if streaming:
        cmd.append("--streaming")

    success, _ = run_command(cmd, f"Training on {dataset}")
    return success


def evaluate_dataset(output_dir: Path, num_samples: int = 2000) -> bool:
    if not (output_dir / "model.pt").exists():
        print(f"No trained model found at {output_dir}, skipping evaluation")
        return False

    cmd = [
        sys.executable, "-m", "scripts.evaluate",
        "--model-dir", str(output_dir),
        "--num-samples", str(num_samples),
        "--batch-throughput",
    ]

    success, _ = run_command(cmd, f"Evaluating {output_dir.name}")
    return success


def run_latency_benchmark(
    output_dir: Path,
    num_queries: int = 2000,
    include_kuzu: bool = False,
) -> bool:
    if not (output_dir / "model.pt").exists():
        print(f"No trained model found at {output_dir}, skipping latency benchmark")
        return False

    cmd = [
        sys.executable, "-m", "scripts.benchmark_latency",
        "--model-dir", str(output_dir),
        "--num-queries", str(num_queries),
    ]

    if include_kuzu:
        cmd.append("--kuzu")

    success, _ = run_command(cmd, f"Latency benchmark for {output_dir.name}")
    return success


def generate_dataset_figures(output_dir: Path) -> bool:
    """Generate figures for a single dataset."""
    figures_dir = output_dir / "figures"

    if not (output_dir / "model.pt").exists():
        print(f"No model found at {output_dir}, skipping figures")
        return False

    cmd = [
        sys.executable, "-m", "scripts.visualizations",
        "--output-dir", str(output_dir),
        "--figures-dir", str(figures_dir),
        "--latency",
    ]

    success, _ = run_command(cmd, f"Generating figures for {output_dir.name}")
    return success


def generate_comparison_figures(dataset_dirs: list[Path], figures_dir: Path) -> bool:
    """Generate cross-dataset comparison figures."""
    valid_dirs = []
    valid_names = []

    for output_dir in dataset_dirs:
        if (output_dir / "evaluation_results.json").exists():
            valid_dirs.append(str(output_dir))
            valid_names.append(output_dir.name.upper())

    if len(valid_dirs) < 2:
        print("Need at least 2 datasets with results for comparison")
        return False

    cmd = [
        sys.executable, "-m", "scripts.visualizations",
        "--figures-dir", str(figures_dir),
        "--compare", *valid_dirs,
        "--compare-names", *valid_names,
        "--latency",
    ]

    success, _ = run_command(cmd, "Generating comparison figures")
    return success


def create_results_summary(dataset_dirs: list[Path], config: dict) -> dict:
    """Create combined results summary from all datasets."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "datasets": {},
    }

    for output_dir in dataset_dirs:
        dataset = output_dir.name
        eval_path = output_dir / "evaluation_results.json"
        latency_path = output_dir / "latency_benchmark.json"

        dataset_results = {}

        if eval_path.exists():
            with open(eval_path) as f:
                results = json.load(f)

            dataset_results.update({
                "count_mae": results.get("accuracy", {}).get("count_mae"),
                "count_rmse": results.get("accuracy", {}).get("count_rmse"),
                "count_mape": results.get("accuracy", {}).get("count_mape"),
                "dist_mae": results.get("accuracy", {}).get("dist_mae"),
                "dist_accuracy_1hop": results.get("accuracy", {}).get("dist_accuracy_1hop"),
                "model_size_kb": results.get("memory", {}).get("model_size_kb"),
                "num_params": results.get("model", {}).get("num_params"),
            })

        if latency_path.exists():
            with open(latency_path) as f:
                latency = json.load(f)

            overall = latency.get("overall", {})
            dataset_results.update({
                "model_latency_ms": overall.get("model", {}).get("total", {}).get("mean_ms"),
                "model_p99_ms": overall.get("model", {}).get("total", {}).get("p99_ms"),
                "gt_latency_ms": overall.get("ground_truth", {}).get("mean_ms"),
                "speedup_mean": overall.get("speedup_mean"),
                "speedup_p99": overall.get("speedup_p99"),
                "graph_nodes": latency.get("config", {}).get("graph_nodes"),
                "graph_edges": latency.get("config", {}).get("graph_edges"),
            })

            # Batch throughput
            batch = latency.get("batch_throughput", {})
            if batch:
                max_batch = max(batch.keys())
                dataset_results["max_throughput_qps"] = batch[max_batch].get("throughput_qps")

            # Kuzu results if available
            kuzu = latency.get("kuzu", {})
            if kuzu and "error" not in kuzu:
                dataset_results["kuzu_latency_ms"] = kuzu.get("mean_ms")
                dataset_results["kuzu_speedup_vs_model"] = kuzu.get("speedup_vs_model")

        if dataset_results:
            summary["datasets"][dataset] = dataset_results

    return summary


def print_results_table(summary: dict):
    print("\n" + "="*120)
    print("RESULTS SUMMARY")
    print("="*120)

    headers = ["Dataset", "Nodes", "Edges", "Count MAE", "Dist MAE", "Latency", "GT Lat.", "Speedup", "Throughput"]
    widths = [10, 10, 10, 10, 10, 10, 10, 10, 14]

    header_str = " | ".join(h.center(w) for h, w in zip(headers, widths))
    print(header_str)
    print("-" * len(header_str))

    for dataset, m in summary.get("datasets", {}).items():
        row = [
            dataset[:10],
            f"{m.get('graph_nodes', 0):,}"[:10] if m.get('graph_nodes') else "N/A",
            f"{m.get('graph_edges', 0):,}"[:10] if m.get('graph_edges') else "N/A",
            f"{m.get('count_mae', 0):.2f}" if m.get('count_mae') else "N/A",
            f"{m.get('dist_mae', 0):.2f}" if m.get('dist_mae') else "N/A",
            f"{m.get('model_latency_ms', 0):.2f}ms" if m.get('model_latency_ms') else "N/A",
            f"{m.get('gt_latency_ms', 0):.2f}ms" if m.get('gt_latency_ms') else "N/A",
            f"{m.get('speedup_mean', 0):.1f}x" if m.get('speedup_mean') else "N/A",
            f"{m.get('max_throughput_qps', 0):,.0f} q/s" if m.get('max_throughput_qps') else "N/A",
        ]
        row_str = " | ".join(v.center(w) for v, w in zip(row, widths))
        print(row_str)

    print("="*120)


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments across multiple datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset and output options
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=["deezer", "github", "facebook"],
        help=f"Datasets to use. Available: {', '.join(DATASETS)}"
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run identifier (default: timestamp). Used for output directory naming."
    )
    parser.add_argument("--output-base", type=str, default="experiments", help="Base output directory")

    # Training options
    parser.add_argument("--num-train", type=int, default=100000, help="Training samples per dataset")
    parser.add_argument("--num-val", type=int, default=10000, help="Validation samples")
    parser.add_argument("--num-test", type=int, default=10000, help="Test samples")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode")

    # Latency benchmark options
    parser.add_argument("--latency-queries", type=int, default=2000, help="Queries for latency benchmark")
    parser.add_argument("--kuzu", action="store_true", help="Include Kuzu database in latency benchmark")

    # Run mode options
    parser.add_argument("--train-only", action="store_true", help="Only train, skip evaluation")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing models")
    parser.add_argument("--latency-only", action="store_true", help="Only run latency benchmark")
    parser.add_argument("--figures-only", action="store_true", help="Only generate figures")
    parser.add_argument("--skip-latency", action="store_true", help="Skip latency benchmark")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure generation")

    args = parser.parse_args()

    # Setup run directory
    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(args.output_base) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Validate datasets
    for dataset in args.datasets:
        if dataset not in DATASETS:
            print(f"Warning: Unknown dataset '{dataset}'. Available: {', '.join(DATASETS)}")

    # Save experiment config
    config = {
        "run_id": run_id,
        "datasets": args.datasets,
        "num_train": args.num_train,
        "num_val": args.num_val,
        "num_test": args.num_test,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "streaming": args.streaming,
        "latency_queries": args.latency_queries,
        "include_kuzu": args.kuzu,
        "started_at": datetime.now().isoformat(),
    }

    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Experiment Run: {run_id}")
    print(f"{'='*70}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Output: {run_dir}")
    print(f"{'='*70}\n")

    # Track dataset directories
    dataset_dirs = [run_dir / dataset for dataset in args.datasets]

    # Phase 1: Training
    if not (args.eval_only or args.latency_only or args.figures_only):
        print("\n" + "="*70)
        print("PHASE 1: TRAINING")
        print("="*70)

        for dataset in args.datasets:
            output_dir = run_dir / dataset
            output_dir.mkdir(exist_ok=True)

            success = train_dataset(
                dataset=dataset,
                output_dir=output_dir,
                num_train=args.num_train,
                num_val=args.num_val,
                num_test=args.num_test,
                epochs=args.epochs,
                batch_size=args.batch_size,
                streaming=args.streaming,
            )
            if not success:
                print(f"Training failed for {dataset}")

    # Phase 2: Evaluation
    if not (args.train_only or args.latency_only or args.figures_only):
        print("\n" + "="*70)
        print("PHASE 2: EVALUATION")
        print("="*70)

        for output_dir in dataset_dirs:
            evaluate_dataset(output_dir, num_samples=args.num_test)

    # Phase 3: Latency benchmark
    if not (args.train_only or args.skip_latency or args.figures_only) or args.latency_only:
        print("\n" + "="*70)
        print("PHASE 3: LATENCY BENCHMARK")
        print("="*70)

        for output_dir in dataset_dirs:
            run_latency_benchmark(
                output_dir,
                num_queries=args.latency_queries,
                include_kuzu=args.kuzu,
            )

    # Phase 4: Generate figures
    if not (args.train_only or args.skip_figures):
        print("\n" + "="*70)
        print("PHASE 4: GENERATING FIGURES")
        print("="*70)

        # Per-dataset figures
        for output_dir in dataset_dirs:
            generate_dataset_figures(output_dir)

        # Cross-dataset comparison
        comparison_figures_dir = run_dir / "figures"
        generate_comparison_figures(dataset_dirs, comparison_figures_dir)

    # Create and print summary
    summary = create_results_summary(dataset_dirs, config)
    print_results_table(summary)

    summary["completed_at"] = datetime.now().isoformat()
    summary_path = run_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {run_dir}/")
    print(f"  - config.json: Experiment configuration")
    print(f"  - results_summary.json: Combined results")
    for dataset in args.datasets:
        print(f"  - {dataset}/: Dataset outputs and figures")
    print(f"  - figures/: Cross-dataset comparison figures")


if __name__ == "__main__":
    main()
