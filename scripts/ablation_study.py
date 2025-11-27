"""
Ablation study framework for NeurIPS submission.

Systematically evaluates the contribution of each model component.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Ablation configurations
ABLATIONS = {
    # Full model (baseline for comparison)
    "full": {
        "description": "Full model with all features",
        "flags": [],
    },

    # Architectural ablations
    "small_model": {
        "description": "Smaller model (128 hidden, 3 layers)",
        "flags": ["--hidden-dim", "128", "--num-layers", "3", "--latent-dim", "32"],
    },
    "medium_model": {
        "description": "Medium model (192 hidden, 4 layers)",
        "flags": ["--hidden-dim", "192", "--num-layers", "4", "--latent-dim", "48"],
    },
    "large_model": {
        "description": "Larger model (384 hidden, 8 layers)",
        "flags": ["--hidden-dim", "384", "--num-layers", "8", "--latent-dim", "96"],
    },

    # Feature ablations
    "no_log_transform": {
        "description": "Without log transform for counts",
        "flags": ["--no-log-transform"],
    },
    "no_structural": {
        "description": "Without structural node features",
        "flags": ["--no-structural"],
    },
    "no_stratified": {
        "description": "Without stratified sampling",
        "flags": ["--no-stratified"],
    },

    # Training ablations
    "low_lr": {
        "description": "Lower learning rate (1e-4)",
        "flags": ["--lr", "1e-4"],
    },
    "high_lr": {
        "description": "Higher learning rate (1e-3)",
        "flags": ["--lr", "1e-3"],
    },
    "no_warmup": {
        "description": "No learning rate warmup",
        "flags": ["--warmup-ratio", "0"],
    },
    "small_batch": {
        "description": "Smaller batch size (128)",
        "flags": ["--batch-size", "128"],
    },
    "large_batch": {
        "description": "Larger batch size (1024)",
        "flags": ["--batch-size", "1024"],
    },

    # Dropout ablations
    "no_dropout": {
        "description": "No dropout",
        "flags": ["--dropout", "0"],
    },
    "high_dropout": {
        "description": "Higher dropout (0.2)",
        "flags": ["--dropout", "0.2"],
    },

    # Combined ablations
    "minimal": {
        "description": "Minimal model (small, no structural, no log)",
        "flags": [
            "--hidden-dim", "128",
            "--num-layers", "3",
            "--no-log-transform",
            "--no-structural",
        ],
    },
}


def run_ablation(
    ablation_name: str,
    dataset: str,
    output_dir: Path,
    base_flags: list[str],
    dry_run: bool = False,
) -> dict:
    """Run a single ablation experiment."""
    config = ABLATIONS[ablation_name]

    ablation_dir = output_dir / ablation_name
    ablation_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "training.train",
        "--dataset", dataset,
        "--output-dir", str(ablation_dir),
        *base_flags,
        *config["flags"],
    ]

    print(f"\n{'='*70}")
    print(f"Ablation: {ablation_name}")
    print(f"Description: {config['description']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    if dry_run:
        return {"status": "dry_run", "command": cmd}

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        # Load results
        try:
            with open(ablation_dir / "model_info.json") as f:
                model_info = json.load(f)
            return {
                "status": "success",
                "best_val_median_qerror": model_info.get("best_val_median_qerror"),
                "best_val_count_within_2x": model_info.get("best_val_count_within_2x"),
                "best_epoch": model_info.get("best_epoch"),
            }
        except Exception as e:
            return {"status": "success", "error": str(e)}
    else:
        return {
            "status": "failed",
            "returncode": result.returncode,
            "stderr": result.stderr[-1000:] if result.stderr else "",
        }


def run_ablation_study(
    dataset: str,
    output_base: Path,
    ablations: list[str],
    num_train: int = 50000,
    num_val: int = 5000,
    epochs: int = 100,
    dry_run: bool = False,
) -> dict:
    """Run full ablation study."""
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = output_base / f"ablation_{dataset}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_flags = [
        "--num-train", str(num_train),
        "--num-val", str(num_val),
        "--epochs", str(epochs),
    ]

    results = {
        "run_id": run_id,
        "dataset": dataset,
        "base_flags": base_flags,
        "ablations": {},
        "started_at": datetime.now().isoformat(),
    }

    for ablation_name in ablations:
        if ablation_name not in ABLATIONS:
            print(f"Warning: Unknown ablation '{ablation_name}', skipping")
            continue

        result = run_ablation(ablation_name, dataset, output_dir, base_flags, dry_run)
        results["ablations"][ablation_name] = result

    results["completed_at"] = datetime.now().isoformat()

    # Save results
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    print(f"\n{'Ablation':<20} | {'Q-error':>10} | {'<2x':>8} | {'Status':>10}")
    print("-" * 60)

    for name, res in results["ablations"].items():
        if res["status"] == "success":
            qerr = res.get("best_val_median_qerror", "N/A")
            within2x = res.get("best_val_count_within_2x", "N/A")
            qerr_str = f"{qerr:.2f}" if isinstance(qerr, (int, float)) else str(qerr)
            within2x_str = f"{within2x:.1f}%" if isinstance(within2x, (int, float)) else str(within2x)
            print(f"{name:<20} | {qerr_str:>10} | {within2x_str:>8} | {'success':>10}")
        else:
            print(f"{name:<20} | {'N/A':>10} | {'N/A':>8} | {res['status']:>10}")

    print("=" * 70)
    print(f"\nResults saved to {output_dir}/")

    return results


def generate_ablation_table(results: dict) -> str:
    """Generate LaTeX table for ablation results."""
    lines = [
        r"\begin{table}[t]",
        r"\caption{Ablation study results. $\downarrow$ lower is better for Q-error, $\uparrow$ higher is better for accuracy.}",
        r"\label{tab:ablation}",
        r"\centering",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Configuration & Median Q-error $\downarrow$ & Within 2$\times$ $\uparrow$ \\",
        r"\midrule",
    ]

    for name, res in results["ablations"].items():
        if res["status"] == "success":
            qerr = res.get("best_val_median_qerror", "N/A")
            within2x = res.get("best_val_count_within_2x", "N/A")
            desc = ABLATIONS[name]["description"]

            qerr_str = f"{qerr:.2f}" if isinstance(qerr, (int, float)) else "N/A"
            within2x_str = f"{within2x:.1f}\\%" if isinstance(within2x, (int, float)) else "N/A"

            # Highlight full model
            if name == "full":
                lines.append(f"\\textbf{{{desc}}} & \\textbf{{{qerr_str}}} & \\textbf{{{within2x_str}}} \\\\")
            else:
                lines.append(f"{desc} & {qerr_str} & {within2x_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--output-base", type=str, default="experiments", help="Output base directory")
    parser.add_argument(
        "--ablations", type=str, nargs="+",
        default=["full", "no_log_transform", "no_structural", "no_stratified", "small_model"],
        help="Ablations to run"
    )
    parser.add_argument("--num-train", type=int, default=50000, help="Training samples")
    parser.add_argument("--num-val", type=int, default=5000, help="Validation samples")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--list", action="store_true", help="List available ablations")
    parser.add_argument("--all", action="store_true", help="Run all ablations")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable ablations:")
        for name, config in ABLATIONS.items():
            print(f"  {name:<20} - {config['description']}")
        return

    ablations = list(ABLATIONS.keys()) if args.all else args.ablations

    results = run_ablation_study(
        dataset=args.dataset,
        output_base=Path(args.output_base),
        ablations=ablations,
        num_train=args.num_train,
        num_val=args.num_val,
        epochs=args.epochs,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        # Generate LaTeX table
        table = generate_ablation_table(results)
        print("\nLaTeX table:")
        print(table)


if __name__ == "__main__":
    main()
