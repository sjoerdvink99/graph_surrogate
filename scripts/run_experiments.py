#!/usr/bin/env python3
"""
Run training, evaluation, and latency benchmarks across multiple datasets.

Supports both V1 (original) and V2 (NeurIPS) training pipelines.

Output structure:
    experiments/
    └── {run_id}/                    # e.g., "2024-01-15_gpu" or "neurips_v2"
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
        └── figures/                 # Cross-dataset comparison figures
            └── comparison.pdf
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


# All supported datasets (V2 includes more)
DATASETS = [
    # Original datasets
    "bron", "github", "deezer", "facebook", "amazon", "dblp", "youtube", "reddit",
    # V2 additions
    "wikitalk", "pokec", "ogbn-arxiv", "ogbn-products",
    # Synthetic
    "synthetic-er-100k", "synthetic-ba-100k", "synthetic-sbm-100k",
]

# Default datasets for quick experiments
QUICK_DATASETS = ["bron", "deezer", "facebook"]

# Default datasets for full NeurIPS evaluation
NEURIPS_DATASETS = ["bron", "wikitalk", "amazon", "pokec"]


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
    use_v2: bool = False,
    v2_options: dict = None,
) -> bool:
    """Train model on a dataset."""
    v2_options = v2_options or {}

    if use_v2:
        # Use V2 training pipeline with all enhancements
        cmd = [
            sys.executable, "-m", "training.train",
            "--dataset", dataset,
            "--output-dir", str(output_dir),
            "--num-train", str(num_train),
            "--num-val", str(num_val),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
        ]

        # V2-specific options
        if v2_options.get("hidden_dim"):
            cmd.extend(["--hidden-dim", str(v2_options["hidden_dim"])])
        if v2_options.get("num_layers"):
            cmd.extend(["--num-layers", str(v2_options["num_layers"])])
        if v2_options.get("latent_dim"):
            cmd.extend(["--latent-dim", str(v2_options["latent_dim"])])
        if v2_options.get("lr"):
            cmd.extend(["--lr", str(v2_options["lr"])])
        if v2_options.get("dropout"):
            cmd.extend(["--dropout", str(v2_options["dropout"])])
        if v2_options.get("no_log_transform"):
            cmd.append("--no-log-transform")
        if v2_options.get("no_structural"):
            cmd.append("--no-structural")
        if v2_options.get("no_stratified"):
            cmd.append("--no-stratified")
        if v2_options.get("warmup_ratio"):
            cmd.extend(["--warmup-ratio", str(v2_options["warmup_ratio"])])

    else:
        # Use original training pipeline
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

    success, _ = run_command(cmd, f"Training on {dataset} (V2={use_v2})")
    return success


def evaluate_dataset(output_dir: Path, num_samples: int = 2000, use_v2: bool = False) -> bool:
    """Evaluate trained model."""
    if not (output_dir / "model.pt").exists():
        print(f"No trained model found at {output_dir}, skipping evaluation")
        return False

    if use_v2:
        # Use comprehensive NeurIPS evaluation
        cmd = [
            sys.executable, "-m", "scripts.neurips_evaluation",
            "--model-dir", str(output_dir),
            "--output-dir", str(output_dir),
        ]
    else:
        cmd = [
            sys.executable, "-m", "scripts.evaluate",
            "--model-dir", str(output_dir),
            "--num-samples", str(num_samples),
            "--batch-throughput",
        ]

    success, _ = run_command(cmd, f"Evaluating {output_dir.name} (V2={use_v2})")
    return success


def run_latency_benchmark(
    output_dir: Path,
    num_queries: int = 2000,
    include_kuzu: bool = False,
) -> bool:
    """Run latency benchmark."""
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


def run_ablation_study(
    dataset: str,
    output_base: Path,
    ablations: list[str],
    num_train: int = 50000,
    num_val: int = 5000,
    epochs: int = 100,
) -> bool:
    """Run ablation study."""
    cmd = [
        sys.executable, "-m", "scripts.ablation_study",
        "--dataset", dataset,
        "--output-base", str(output_base),
        "--ablations", *ablations,
        "--num-train", str(num_train),
        "--num-val", str(num_val),
        "--epochs", str(epochs),
    ]

    success, _ = run_command(cmd, f"Ablation study on {dataset}")
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


def generate_publication_figures(results_dir: Path, figures_dir: Path) -> bool:
    """Generate publication-quality figures."""
    cmd = [
        sys.executable, "-m", "scripts.generate_figures",
        "--results-dir", str(results_dir),
        "--output-dir", str(figures_dir),
    ]

    success, _ = run_command(cmd, "Generating publication figures")
    return success


def generate_latex_tables(results_dir: Path, tables_dir: Path) -> bool:
    """Generate LaTeX tables."""
    cmd = [
        sys.executable, "-m", "scripts.generate_tables",
        "--results-dir", str(results_dir),
        "--output-dir", str(tables_dir),
    ]

    success, _ = run_command(cmd, "Generating LaTeX tables")
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


def create_results_summary(dataset_dirs: list[Path], config: dict, use_v2: bool = False) -> dict:
    """Create combined results summary from all datasets."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "version": "v2" if use_v2 else "v1",
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

            # V2 metrics (Q-error based)
            if use_v2 and "accuracy" in results:
                acc = results["accuracy"]
                dataset_results.update({
                    "median_qerror": acc.get("count_median_qerror"),
                    "mean_qerror": acc.get("count_mean_qerror"),
                    "within_2x": acc.get("count_within_2x"),
                    "within_5x": acc.get("count_within_5x"),
                    "p95_qerror": acc.get("count_p95_qerror"),
                    "dist_mae": acc.get("dist_mae"),
                    "dist_accuracy_1hop": acc.get("dist_accuracy_1hop"),
                })
            else:
                # V1 metrics
                dataset_results.update({
                    "count_mae": results.get("accuracy", {}).get("count_mae"),
                    "count_rmse": results.get("accuracy", {}).get("count_rmse"),
                    "count_mape": results.get("accuracy", {}).get("count_mape"),
                    "dist_mae": results.get("accuracy", {}).get("dist_mae"),
                    "dist_accuracy_1hop": results.get("accuracy", {}).get("dist_accuracy_1hop"),
                })

            dataset_results.update({
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

        if dataset_results:
            summary["datasets"][dataset] = dataset_results

    return summary


def print_results_table(summary: dict, use_v2: bool = False):
    """Print results summary table."""
    print("\n" + "="*130)
    print("RESULTS SUMMARY" + (" (V2 - NeurIPS)" if use_v2 else ""))
    print("="*130)

    if use_v2:
        headers = ["Dataset", "Nodes", "Edges", "Med Q-err", "Within 2x", "Dist MAE", "Latency", "Speedup"]
        widths = [12, 12, 12, 12, 12, 12, 12, 12]

        header_str = " | ".join(h.center(w) for h, w in zip(headers, widths))
        print(header_str)
        print("-" * len(header_str))

        for dataset, m in summary.get("datasets", {}).items():
            row = [
                dataset[:12],
                f"{m.get('graph_nodes', 0):,}"[:12] if m.get('graph_nodes') else "N/A",
                f"{m.get('graph_edges', 0):,}"[:12] if m.get('graph_edges') else "N/A",
                f"{m.get('median_qerror', 0):.2f}" if m.get('median_qerror') else "N/A",
                f"{m.get('within_2x', 0):.1f}%" if m.get('within_2x') else "N/A",
                f"{m.get('dist_mae', 0):.2f}" if m.get('dist_mae') else "N/A",
                f"{m.get('model_latency_ms', 0):.2f}ms" if m.get('model_latency_ms') else "N/A",
                f"{m.get('speedup_mean', 0):.1f}x" if m.get('speedup_mean') else "N/A",
            ]
            row_str = " | ".join(v.center(w) for v, w in zip(row, widths))
            print(row_str)
    else:
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

    print("="*130)


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments across multiple datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset and output options
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=None,
        help=f"Datasets to use. Available: {', '.join(DATASETS)}"
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run identifier (default: timestamp). Used for output directory naming."
    )
    parser.add_argument("--output-base", type=str, default="experiments", help="Base output directory")

    # Version selection
    parser.add_argument("--v2", action="store_true", help="Use V2 training pipeline (NeurIPS version)")
    parser.add_argument("--quick", action="store_true", help="Quick test run with minimal settings")
    parser.add_argument("--neurips", action="store_true", help="Full NeurIPS evaluation (implies --v2)")

    # Training options
    parser.add_argument("--num-train", type=int, default=None, help="Training samples per dataset")
    parser.add_argument("--num-val", type=int, default=None, help="Validation samples")
    parser.add_argument("--num-test", type=int, default=10000, help="Test samples")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode (V1 only)")

    # V2 model options
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Latency benchmark options
    parser.add_argument("--latency-queries", type=int, default=2000, help="Queries for latency benchmark")
    parser.add_argument("--kuzu", action="store_true", help="Include Kuzu database in latency benchmark")

    # Ablation study
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--ablation-configs", type=str, nargs="+",
                       default=["full", "no_log_transform", "no_structural", "no_stratified", "small_model"],
                       help="Ablation configurations to test")

    # Run mode options
    parser.add_argument("--train-only", action="store_true", help="Only train, skip evaluation")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing models")
    parser.add_argument("--latency-only", action="store_true", help="Only run latency benchmark")
    parser.add_argument("--figures-only", action="store_true", help="Only generate figures")
    parser.add_argument("--skip-latency", action="store_true", help="Skip latency benchmark")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure generation")

    args = parser.parse_args()

    # Apply presets
    use_v2 = args.v2 or args.neurips

    if args.quick:
        datasets = args.datasets or QUICK_DATASETS[:2]
        num_train = args.num_train or 5000
        num_val = args.num_val or 1000
        epochs = args.epochs or 10
        batch_size = args.batch_size or 256
    elif args.neurips:
        datasets = args.datasets or NEURIPS_DATASETS
        num_train = args.num_train or 100000
        num_val = args.num_val or 10000
        epochs = args.epochs or 100
        batch_size = args.batch_size or 512
    else:
        datasets = args.datasets or QUICK_DATASETS
        num_train = args.num_train or 100000
        num_val = args.num_val or 10000
        epochs = args.epochs or 200
        batch_size = args.batch_size or 256

    # Setup run directory
    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if use_v2 and not args.run_id:
        run_id = f"v2_{run_id}"
    run_dir = Path(args.output_base) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # V2 options
    v2_options = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "latent_dim": args.latent_dim,
        "lr": args.lr,
        "dropout": args.dropout,
    }

    # Save experiment config
    config = {
        "run_id": run_id,
        "version": "v2" if use_v2 else "v1",
        "datasets": datasets,
        "num_train": num_train,
        "num_val": num_val,
        "num_test": args.num_test,
        "epochs": epochs,
        "batch_size": batch_size,
        "streaming": args.streaming,
        "latency_queries": args.latency_queries,
        "include_kuzu": args.kuzu,
        "v2_options": v2_options if use_v2 else None,
        "started_at": datetime.now().isoformat(),
    }

    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Experiment Run: {run_id}")
    print(f"{'='*70}")
    print(f"Version: {'V2 (NeurIPS)' if use_v2 else 'V1 (Original)'}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Training: {num_train} samples, {epochs} epochs")
    print(f"Output: {run_dir}")
    print(f"{'='*70}\n")

    # Track dataset directories
    dataset_dirs = [run_dir / dataset for dataset in datasets]

    # Phase 1: Training
    if not (args.eval_only or args.latency_only or args.figures_only):
        print("\n" + "="*70)
        print("PHASE 1: TRAINING")
        print("="*70)

        for dataset in datasets:
            output_dir = run_dir / dataset
            output_dir.mkdir(exist_ok=True)

            success = train_dataset(
                dataset=dataset,
                output_dir=output_dir,
                num_train=num_train,
                num_val=num_val,
                num_test=args.num_test,
                epochs=epochs,
                batch_size=batch_size,
                streaming=args.streaming,
                use_v2=use_v2,
                v2_options=v2_options,
            )
            if not success:
                print(f"Training failed for {dataset}")

    # Phase 2: Evaluation
    if not (args.train_only or args.latency_only or args.figures_only):
        print("\n" + "="*70)
        print("PHASE 2: EVALUATION")
        print("="*70)

        for output_dir in dataset_dirs:
            evaluate_dataset(output_dir, num_samples=args.num_test, use_v2=use_v2)

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

    # Phase 4: Ablation study (V2 only)
    if args.ablation and use_v2:
        print("\n" + "="*70)
        print("PHASE 4: ABLATION STUDY")
        print("="*70)

        for dataset in datasets:
            run_ablation_study(
                dataset=dataset,
                output_base=run_dir / "ablation",
                ablations=args.ablation_configs,
                num_train=num_train // 2,  # Faster ablations
                num_val=num_val // 2,
                epochs=epochs // 2,
            )

    # Phase 5: Generate figures
    if not (args.train_only or args.skip_figures):
        print("\n" + "="*70)
        print("PHASE 5: GENERATING FIGURES")
        print("="*70)

        # Per-dataset figures
        for output_dir in dataset_dirs:
            generate_dataset_figures(output_dir)

        # Cross-dataset comparison
        comparison_figures_dir = run_dir / "figures"
        generate_comparison_figures(dataset_dirs, comparison_figures_dir)

        # Publication figures (V2 only)
        if use_v2:
            generate_publication_figures(run_dir, run_dir / "publication_figures")
            generate_latex_tables(run_dir, run_dir / "tables")

    # Create and print summary
    summary = create_results_summary(dataset_dirs, config, use_v2=use_v2)
    print_results_table(summary, use_v2=use_v2)

    summary["completed_at"] = datetime.now().isoformat()
    summary_path = run_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {run_dir}/")
    print(f"  - config.json: Experiment configuration")
    print(f"  - results_summary.json: Combined results")
    for dataset in datasets:
        print(f"  - {dataset}/: Dataset outputs")
    if use_v2:
        print(f"  - publication_figures/: Publication-ready figures")
        print(f"  - tables/: LaTeX tables")
    print(f"  - figures/: Comparison figures")


if __name__ == "__main__":
    main()
