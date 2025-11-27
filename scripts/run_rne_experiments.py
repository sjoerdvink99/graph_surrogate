#!/usr/bin/env python3
"""
Run Hierarchical Road Network Embedding (RNE) experiments for NeurIPS submission.

This script orchestrates the complete experimental pipeline:
1. Load/download road network datasets
2. Create graph partitions
3. Train Hierarchical RNE with 3-phase training
4. Train baseline models for comparison
5. Evaluate all models (distance, range, kNN queries)
6. Generate publication-quality figures

Usage:
    python -m scripts.run_rne_experiments --datasets road_ny road_cal --rne
    python -m scripts.run_rne_experiments --full-comparison --neurips

For SLURM execution, set environment variables:
    RNE_DATASETS="road_ny road_cal" sbatch run_rne.sbatch
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch
import networkx as nx
from tqdm import tqdm

# Project imports
from datasets import load_dataset, get_dataset_info, list_datasets
from training.graph_loader import load_graph, save_graph, GraphMetadata


# Road network datasets (can be added to registry)
ROAD_NETWORK_DATASETS = [
    "road_ny",      # New York road network
    "road_cal",     # California road network
    "road_tx",      # Texas road network
    "road_europe",  # European road network
]

# Synthetic benchmarks for scalability testing
SYNTHETIC_DATASETS = [
    "synthetic-grid-10k",
    "synthetic-grid-100k",
    "synthetic-ba-100k",
]

# Default experiment configurations
EXPERIMENT_CONFIGS = {
    "quick": {
        "num_train_pairs": 10000,
        "num_val_pairs": 2000,
        "phase1_epochs": 5,
        "phase2_epochs": 10,
        "phase3_epochs": 20,
        "batch_size": 512,
    },
    "standard": {
        "num_train_pairs": 100000,
        "num_val_pairs": 10000,
        "phase1_epochs": 20,
        "phase2_epochs": 50,
        "phase3_epochs": 100,
        "batch_size": 1024,
    },
    "neurips": {
        "num_train_pairs": 500000,
        "num_val_pairs": 50000,
        "phase1_epochs": 30,
        "phase2_epochs": 100,
        "phase3_epochs": 200,
        "batch_size": 2048,
    },
}


def create_synthetic_road_network(
    num_nodes: int,
    network_type: str = "grid",
    seed: int = 42,
) -> tuple[nx.Graph, GraphMetadata]:
    """
    Create synthetic road network for testing.

    Args:
        num_nodes: Approximate number of nodes
        network_type: "grid" or "random"
        seed: Random seed

    Returns:
        Graph and metadata
    """
    import random
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    if network_type == "grid":
        # Create grid graph with some random perturbations
        side = int(np.sqrt(num_nodes))
        G = nx.grid_2d_graph(side, side)

        # Convert to regular graph with integer node IDs
        mapping = {(i, j): i * side + j for i in range(side) for j in range(side)}
        G = nx.relabel_nodes(G, mapping)

        # Add coordinates
        for node in G.nodes():
            i, j = node // side, node % side
            # Add small random perturbation
            G.nodes[node]['x'] = i + np_rng.randn() * 0.1
            G.nodes[node]['y'] = j + np_rng.randn() * 0.1
            G.nodes[node]['node_type'] = 'intersection'

        # Add some random edges for realism
        num_random_edges = int(G.number_of_edges() * 0.1)
        nodes = list(G.nodes())
        for _ in range(num_random_edges):
            u, v = rng.sample(nodes, 2)
            if not G.has_edge(u, v):
                G.add_edge(u, v)

    else:
        # Barabási-Albert model
        G = nx.barabasi_albert_graph(num_nodes, 3, seed=seed)

        # Add random positions
        for node in G.nodes():
            G.nodes[node]['x'] = np_rng.randn()
            G.nodes[node]['y'] = np_rng.randn()
            G.nodes[node]['node_type'] = 'intersection'

    metadata = GraphMetadata(
        name=f"synthetic_{network_type}_{num_nodes}",
        num_nodes=G.number_of_nodes(),
        num_edges=G.number_of_edges(),
        node_types=['intersection'],
        edge_types=['road'],
    )

    return G, metadata


def setup_device():
    """Setup compute device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def train_hierarchical_rne_experiment(
    G: nx.Graph,
    output_dir: Path,
    config: dict,
    device: torch.device,
    seed: int = 42,
) -> dict:
    """
    Train Hierarchical RNE model.

    Returns training results and metrics.
    """
    from model.hierarchical_rne import HierarchicalRNE, HierarchicalRNEConfig
    from model.partitioning import MetisPartitioner
    from training.hierarchical_trainer import (
        HierarchicalTrainer,
        HierarchicalTrainingConfig,
        DistancePairDataset,
        generate_distance_pairs,
    )

    print("\n" + "=" * 70)
    print("Training Hierarchical RNE")
    print("=" * 70)

    start_time = time.time()

    # Create partition hierarchy
    print("\n1. Creating partition hierarchy...")
    partitioner = MetisPartitioner(
        num_levels=4,
        min_partition_size=10,
        seed=seed,
    )
    partition_tree = partitioner.partition(G)
    print(f"   Levels: {partition_tree.num_levels}, Sizes: {partition_tree.level_sizes}")

    # Generate training data
    print("\n2. Generating training pairs...")
    source_train, target_train, dist_train = generate_distance_pairs(
        G, config['num_train_pairs'], seed=seed, show_progress=True
    )
    source_val, target_val, dist_val = generate_distance_pairs(
        G, config['num_val_pairs'], seed=seed + 1, show_progress=True
    )

    train_data = DistancePairDataset(source_train, target_train, dist_train)
    val_data = DistancePairDataset(source_val, target_val, dist_val)

    print(f"   Training pairs: {len(train_data):,}")
    print(f"   Validation pairs: {len(val_data):,}")
    print(f"   Distance range: [{dist_train.min():.0f}, {dist_train.max():.0f}]")

    # Create model
    print("\n3. Creating model...")
    rne_config = HierarchicalRNEConfig(
        embed_dim=128,
        num_levels=partition_tree.num_levels,
        distance_metric='l1',
        dropout=0.1,
        max_nodes=max(G.nodes()) + 1,
    )

    model = HierarchicalRNE(
        config=rne_config,
        partition_tree=partition_tree,
        use_mlp_refinement=True,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")

    # Training config
    train_config = HierarchicalTrainingConfig(
        seed=seed,
        phase1_epochs_per_level=config['phase1_epochs'] // partition_tree.num_levels,
        phase2_epochs=config['phase2_epochs'],
        phase3_epochs=config['phase3_epochs'],
        batch_size=config['batch_size'],
        use_amp=True,
    )

    # Train
    print("\n4. Training (3-phase)...")
    trainer = HierarchicalTrainer(
        model=model,
        config=train_config,
        train_data=train_data,
        val_data=val_data,
        output_dir=output_dir,
    )

    results = trainer.train()

    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time:.1f}s")

    # Save model
    torch.save(model.state_dict(), output_dir / "rne_model.pt")

    # Save partition tree info
    partition_info = {
        'num_levels': partition_tree.num_levels,
        'level_sizes': partition_tree.level_sizes,
    }
    with open(output_dir / "partition_info.json", 'w') as f:
        json.dump(partition_info, f, indent=2)

    return {
        'model': model,
        'partition_tree': partition_tree,
        'training_time': training_time,
        'final_metrics': results['final_metrics'],
        'history': results['history'],
    }


def train_baseline_models(
    G: nx.Graph,
    output_dir: Path,
    config: dict,
    device: torch.device,
    seed: int = 42,
) -> dict:
    """
    Train baseline models for comparison.

    Returns dict of trained models and their metrics.
    """
    from model.baselines import create_baseline_model, FlatRNE, S2GNN, DEAR
    from training.hierarchical_trainer import (
        DistancePairDataset,
        generate_distance_pairs,
    )
    from model.hierarchical_rne import RNELoss

    print("\n" + "=" * 70)
    print("Training Baseline Models")
    print("=" * 70)

    # Generate training data (same for all baselines)
    source_train, target_train, dist_train = generate_distance_pairs(
        G, config['num_train_pairs'], seed=seed, show_progress=True
    )
    source_val, target_val, dist_val = generate_distance_pairs(
        G, config['num_val_pairs'], seed=seed + 1, show_progress=False
    )

    train_data = DistancePairDataset(source_train, target_train, dist_train)
    val_data = DistancePairDataset(source_val, target_val, dist_val)

    max_nodes = max(G.nodes()) + 1
    baseline_results = {}

    # Baseline models to train
    baselines = {
        'flat_rne': {'embed_dim': 128, 'distance_metric': 'l1'},
        's2gnn': {'embed_dim': 128, 'hidden_dim': 256, 'num_layers': 4},
        'dear': {'embed_dim': 128, 'hidden_dim': 256, 'num_iterations': 10},
    }

    for baseline_name, baseline_config in baselines.items():
        print(f"\n--- Training {baseline_name.upper()} ---")

        try:
            model = create_baseline_model(
                baseline_name,
                config=baseline_config,
                max_nodes=max_nodes,
            )
            model = model.to(device)

            # Simple training loop
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = RNELoss()

            from torch.utils.data import DataLoader
            train_loader = DataLoader(
                train_data,
                batch_size=config['batch_size'],
                shuffle=True,
            )

            num_epochs = config['phase2_epochs']  # Use phase 2 epochs for baselines

            best_mre = float('inf')
            model.train()

            for epoch in range(num_epochs):
                epoch_loss = 0
                for batch in train_loader:
                    source_ids = batch['source_id'].to(device)
                    target_ids = batch['target_id'].to(device)
                    distances = batch['distance'].to(device)

                    optimizer.zero_grad()
                    pred_dist, source_emb, target_emb = model(source_ids, target_ids)
                    loss, _ = criterion(pred_dist, distances, source_emb, target_emb)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

            # Evaluate
            model.eval()
            with torch.no_grad():
                val_source = source_val.to(device)
                val_target = target_val.to(device)
                pred, _, _ = model(val_source, val_target)
                pred = pred.cpu()

                rel_errors = torch.abs(pred - dist_val) / (dist_val + 1.0)
                mre = rel_errors.mean().item()
                mae = torch.abs(pred - dist_val).mean().item()

            baseline_results[baseline_name] = {
                'model': model,
                'mre': mre,
                'mae': mae,
            }

            # Save model
            torch.save(model.state_dict(), output_dir / f"{baseline_name}_model.pt")

            print(f"  {baseline_name}: MRE={mre:.4f}, MAE={mae:.4f}")

        except Exception as e:
            print(f"  Warning: Failed to train {baseline_name}: {e}")

    return baseline_results


def evaluate_all_models(
    models: dict,
    G: nx.Graph,
    output_dir: Path,
    num_queries: int = 5000,
    device: torch.device = None,
) -> dict:
    """
    Comprehensive evaluation of all models.
    """
    from training.rne_evaluation import (
        evaluate_distance_approximation,
        evaluate_range_queries,
        evaluate_knn_queries,
        RNESearchIndex,
        print_evaluation_report,
    )

    print("\n" + "=" * 70)
    print("Model Evaluation")
    print("=" * 70)

    evaluation_results = {}

    for model_name, model_data in models.items():
        print(f"\n--- Evaluating {model_name} ---")

        model = model_data['model'] if isinstance(model_data, dict) else model_data

        try:
            metrics = evaluate_distance_approximation(
                model=model,
                G=G,
                num_queries=num_queries,
                device=device,
            )

            print_evaluation_report(metrics, title=f"{model_name} Results")

            evaluation_results[model_name] = {
                'mre': metrics.mre,
                'mae': metrics.mae,
                'median_ae': metrics.median_ae,
                'pct_within_5': metrics.pct_within_5,
                'pct_within_10': metrics.pct_within_10,
                'mean_query_time_ns': metrics.mean_query_time_ns,
                'median_query_time_ns': metrics.median_query_time_ns,
            }

        except Exception as e:
            print(f"  Warning: Evaluation failed: {e}")

    # Save results
    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    return evaluation_results


def run_full_experiment(
    dataset_name: str,
    output_dir: Path,
    experiment_config: str = "standard",
    include_baselines: bool = True,
    seed: int = 42,
):
    """
    Run complete experiment for a single dataset.
    """
    print("\n" + "=" * 70)
    print(f"Running Experiment: {dataset_name}")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    device = setup_device()
    config = EXPERIMENT_CONFIGS[experiment_config]

    # Load or create dataset
    print("\n1. Loading dataset...")
    if dataset_name.startswith("synthetic"):
        parts = dataset_name.split("-")
        network_type = parts[1] if len(parts) > 1 else "grid"
        num_nodes = int(parts[2].replace("k", "000")) if len(parts) > 2 else 10000
        G, metadata = create_synthetic_road_network(num_nodes, network_type, seed)
    else:
        try:
            G, meta_dict = load_dataset(dataset_name)
            metadata = GraphMetadata.from_dict(meta_dict) if isinstance(meta_dict, dict) else meta_dict
        except Exception as e:
            print(f"  Warning: Could not load {dataset_name}, creating synthetic: {e}")
            G, metadata = create_synthetic_road_network(10000, "grid", seed)

    print(f"   Nodes: {G.number_of_nodes():,}")
    print(f"   Edges: {G.number_of_edges():,}")

    # Save experiment config
    experiment_info = {
        'dataset': dataset_name,
        'config': experiment_config,
        'graph_nodes': G.number_of_nodes(),
        'graph_edges': G.number_of_edges(),
        'seed': seed,
        'started_at': datetime.now().isoformat(),
    }
    with open(output_dir / "experiment_info.json", 'w') as f:
        json.dump(experiment_info, f, indent=2)

    # Train Hierarchical RNE
    rne_results = train_hierarchical_rne_experiment(G, output_dir, config, device, seed)

    # Train baselines
    baseline_results = {}
    if include_baselines:
        baseline_results = train_baseline_models(G, output_dir, config, device, seed)

    # Combine all models for evaluation
    all_models = {
        'Hierarchical RNE': rne_results,
        **{name: data for name, data in baseline_results.items()},
    }

    # Evaluate
    eval_results = evaluate_all_models(all_models, G, output_dir, device=device)

    # Generate figures
    print("\n" + "=" * 70)
    print("Generating Figures")
    print("=" * 70)

    from scripts.rne_visualizations import generate_all_figures
    figures_dir = output_dir / "figures"
    generate_all_figures(output_dir, figures_dir)

    # Final summary
    experiment_info['completed_at'] = datetime.now().isoformat()
    experiment_info['results'] = eval_results
    with open(output_dir / "experiment_info.json", 'w') as f:
        json.dump(experiment_info, f, indent=2)

    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")

    return eval_results


def main():
    parser = argparse.ArgumentParser(
        description="Run Hierarchical RNE experiments for NeurIPS submission",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset options
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=["synthetic-grid-10k"],
        help="Datasets to use"
    )
    parser.add_argument(
        "--output-base", type=str, default="experiments/rne",
        help="Base output directory"
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run identifier"
    )

    # Experiment configuration
    parser.add_argument(
        "--config", type=str, default="standard",
        choices=["quick", "standard", "neurips"],
        help="Experiment configuration preset"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model options
    parser.add_argument("--rne-only", action="store_true", help="Only train Hierarchical RNE")
    parser.add_argument("--baselines-only", action="store_true", help="Only train baselines")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing models")

    # NeurIPS mode
    parser.add_argument("--neurips", action="store_true",
                       help="Full NeurIPS evaluation (uses 'neurips' config)")

    args = parser.parse_args()

    # Apply presets
    if args.neurips:
        config = "neurips"
    else:
        config = args.config

    # Setup run directory
    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(args.output_base) / run_id

    print(f"\n{'='*70}")
    print(f"Hierarchical RNE Experiment Suite")
    print(f"{'='*70}")
    print(f"Run ID: {run_id}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Config: {config}")
    print(f"Output: {run_dir}")
    print(f"{'='*70}\n")

    # Run experiments for each dataset
    all_results = {}

    for dataset in args.datasets:
        dataset_dir = run_dir / dataset

        results = run_full_experiment(
            dataset_name=dataset,
            output_dir=dataset_dir,
            experiment_config=config,
            include_baselines=not args.rne_only,
            seed=args.seed,
        )

        all_results[dataset] = results

    # Save combined results
    summary = {
        'run_id': run_id,
        'config': config,
        'datasets': all_results,
        'completed_at': datetime.now().isoformat(),
    }

    with open(run_dir / "results_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print("\n" + "=" * 70)
    print("All Experiments Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {run_dir}")

    for dataset, results in all_results.items():
        print(f"\n{dataset}:")
        if 'Hierarchical RNE' in results:
            rne = results['Hierarchical RNE']
            print(f"  Hierarchical RNE: MRE={rne.get('mre', 'N/A'):.4f}, "
                  f"MAE={rne.get('mae', 'N/A'):.4f}, "
                  f"Query Time={rne.get('mean_query_time_ns', 0)/1000:.1f}μs")


if __name__ == "__main__":
    main()
