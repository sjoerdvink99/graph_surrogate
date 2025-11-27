import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.encoder import EncoderConfig, QueryEncoder, create_encoder_config
from model.network import GraphSurrogate, ModelConfig, ImprovedLoss
from model.structural_features import StructuralFeatureComputer, compute_or_load_features

from .graph_loader import load_graph, save_graph, GraphMetadata
from .query_sampler import (
    QueryType,
    StratifiedQuerySampler,
    SamplingConfig,
    SamplingStrategy,
    analyze_query_difficulty,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_qerror(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1.0) -> torch.Tensor:
    """Compute Q-error: max(pred/target, target/pred)."""
    pred = torch.clamp(pred, min=epsilon)
    target = torch.clamp(target, min=epsilon)
    return torch.max(pred / target, target / pred)


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    use_log_transform: bool = True,
) -> dict:
    """Compute comprehensive metrics for evaluation."""
    if use_log_transform:
        # Convert from log space to original scale
        pred_original = torch.expm1(pred).clamp(min=0)
    else:
        pred_original = pred

    # Absolute error
    abs_error = torch.abs(pred_original - target)

    # Q-error
    qerror = compute_qerror(pred_original, target)

    return {
        'mae': abs_error.mean().item(),
        'rmse': torch.sqrt((abs_error ** 2).mean()).item(),
        'median_error': abs_error.median().item(),
        'median_qerror': qerror.median().item(),
        'mean_qerror': qerror.mean().item(),
        'p90_qerror': torch.quantile(qerror, 0.9).item(),
        'p95_qerror': torch.quantile(qerror, 0.95).item(),
        'p99_qerror': torch.quantile(qerror, 0.99).item(),
        'within_2x': (qerror <= 2).float().mean().item() * 100,
        'within_5x': (qerror <= 5).float().mean().item() * 100,
        'within_10x': (qerror <= 10).float().mean().item() * 100,
    }


def prepare_data(
    G,
    metadata: GraphMetadata,
    output_dir: Path,
    num_train: int = 100000,
    num_val: int = 10000,
    num_test: int = 10000,
    seed: int = 42,
    use_stratified: bool = True,
    use_structural_features: bool = True,
) -> tuple:
    """
    Prepare training data with all V2 enhancements.
    """
    node_types = metadata.node_types

    # Create encoder config
    config = create_encoder_config(G, node_types, metadata)
    config.use_structural_features = use_structural_features
    encoder = QueryEncoder(config)

    # Compute or load structural features
    structural_computer = None
    if use_structural_features:
        structural_computer = compute_or_load_features(
            G, output_dir, show_progress=True
        )
        config.num_structural_features = structural_computer.num_features

    # Sampling configuration
    sampling_config = SamplingConfig(
        strategy=SamplingStrategy.STRATIFIED if use_stratified else SamplingStrategy.UNIFORM,
    )

    # Generate training data
    print(f"\nSampling queries...")
    print(f"  Training: {num_train:,} queries (seed={seed})")

    train_sampler = StratifiedQuerySampler(G, seed=seed, node_types=node_types, config=sampling_config)
    if use_stratified:
        train_data = train_sampler.generate_stratified_dataset(num_train, show_progress=True)
    else:
        train_data = train_sampler.generate_dataset(num_train, show_progress=True)

    print(f"  Validation: {num_val:,} queries (seed={seed + 1})")
    val_sampler = StratifiedQuerySampler(G, seed=seed + 1, node_types=node_types, config=sampling_config)
    val_data = val_sampler.generate_dataset(num_val)

    print(f"  Test: {num_test:,} queries (seed={seed + 2})")
    test_sampler = StratifiedQuerySampler(G, seed=seed + 2, node_types=node_types, config=sampling_config)
    test_data = test_sampler.generate_dataset(num_test)

    # Analyze query difficulty
    train_queries = [q for q, _ in train_data]
    train_results = [r for _, r in train_data]
    difficulty = analyze_query_difficulty(G, train_queries, train_results)
    print(f"\nQuery difficulty analysis:")
    if 'count' in difficulty:
        d = difficulty['count']
        print(f"  Count: mean={d['mean']:.1f}, median={d['median']:.1f}, "
              f"range=[{d['min']}, {d['max']}], log_mean={d['log_mean']:.2f}")
    if 'distance' in difficulty:
        d = difficulty['distance']
        print(f"  Distance: mean={d['mean']:.2f}, unreachable={d['unreachable_ratio']*100:.1f}%")

    # Encode queries
    print("\nEncoding queries...")

    def encode_dataset(data, desc=""):
        queries = [q for q, _ in data]
        results = [r for _, r in data]

        # Use index-based encoding for learned embeddings
        indices = encoder.encode_indices_batch(queries)

        y = torch.tensor(results, dtype=torch.float32)
        qt = torch.tensor([0 if q.query_type == QueryType.COUNT else 1 for q in queries])

        # Get structural features
        struct_feat = None
        if use_structural_features and structural_computer is not None:
            node_ids = [q.start_node_id for q in queries]
            struct_feat = structural_computer.get_features_batch(node_ids)

        return indices, y, qt, struct_feat

    train_indices, y_train, qt_train, train_struct = encode_dataset(train_data, "train")
    val_indices, y_val, qt_val, val_struct = encode_dataset(val_data, "val")
    test_indices, y_test, qt_test, test_struct = encode_dataset(test_data, "test")

    # Save encoder config and test data
    config.save(output_dir / "encoder_config.json")

    torch.save({
        "indices": test_indices, "y": y_test, "qt": qt_test,
        "struct": test_struct,
        "val_indices": val_indices, "y_val": y_val, "qt_val": qt_val,
        "val_struct": val_struct,
    }, output_dir / "test_data.pt")

    test_queries_json = [q.to_dict() | {"result": r} for q, r in test_data]
    with open(output_dir / "test_queries.json", "w") as f:
        json.dump(test_queries_json, f)

    # Save difficulty analysis
    with open(output_dir / "query_difficulty.json", "w") as f:
        json.dump(difficulty, f, indent=2)

    return (train_indices, y_train, qt_train, train_struct,
            val_indices, y_val, qt_val, val_struct,
            test_indices, y_test, qt_test, test_struct, config)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine learning rate schedule with warmup."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class IndexDataset(torch.utils.data.Dataset):
    """Dataset for index-based encoding with learned embeddings."""

    def __init__(self, indices: dict, y: torch.Tensor, qt: torch.Tensor, struct_feat: Optional[torch.Tensor] = None):
        self.indices = indices  # dict of key -> tensor
        self.y = y
        self.qt = qt
        self.struct_feat = struct_feat
        self.n = len(y)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.indices.items()}
        item['y'] = self.y[idx]
        item['qt'] = self.qt[idx]
        if self.struct_feat is not None:
            item['struct'] = self.struct_feat[idx]
        return item


def train_model(
    train_data: tuple,
    val_data: tuple,
    encoder_config: EncoderConfig,
    model_config: Optional[ModelConfig] = None,
    epochs: int = 200,
    batch_size: int = 512,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    warmup_ratio: float = 0.1,
    grad_clip: float = 1.0,
    patience: int = 30,
    seed: int = 42,
    use_amp: bool = True,
    output_dir: Optional[Path] = None,
) -> GraphSurrogate:
    """
    Train GraphSurrogate model with learned embeddings.
    """
    set_seed(seed)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple MPS"
        use_amp = False
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        use_amp = False

    # Model configuration
    if model_config is None:
        model_config = ModelConfig()

    # Update model config from encoder config
    model_config.num_node_types = len(encoder_config.node_types)
    model_config.num_degree_bins = len(encoder_config.degree_bins) + 1  # +1 for overflow
    model_config.num_radii = len(encoder_config.radii) + 1
    model_config.num_attr_names = len(encoder_config.attribute_names) + 1
    model_config.num_attr_values = sum(len(v) for v in encoder_config.attribute_values.values()) + 1
    model_config.num_max_hops = len(encoder_config.max_hops_options) + 1
    model_config.num_structural_features = encoder_config.num_structural_features

    print(f"\n{'='*70}")
    print(f"Training Configuration (V2)")
    print(f"{'='*70}")
    print(f"Device: {device_name}")
    print(f"Mixed precision: {use_amp}")
    print(f"Model: hidden_dim={model_config.hidden_dim}, latent_dim={model_config.latent_dim}, "
          f"num_layers={model_config.num_layers}")
    print(f"Training: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"Log transform: {model_config.use_log_transform}")
    print(f"Structural features: {model_config.use_structural_features}")

    # Unpack training data
    train_indices, y_train, qt_train, train_struct = train_data
    val_indices, y_val, qt_val, val_struct = val_data

    # Create datasets
    train_dataset = IndexDataset(train_indices, y_train, qt_train, train_struct)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(device.type == 'cuda'),
        num_workers=4 if device.type == 'cuda' else 0,
    )

    # Create model
    model = GraphSurrogate(model_config)
    model = model.to(device)

    # Compile model if available
    if device.type == 'cuda' and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled with torch.compile()")
        except Exception:
            pass

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,} total, {trainable_params:,} trainable")
    print(f"Training samples: {len(train_dataset):,}")

    # Loss function
    criterion = ImprovedLoss(
        use_log_transform=model_config.use_log_transform,
        count_beta=0.5,
        distance_beta=1.0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # Learning rate schedule
    steps_per_epoch = len(train_loader)
    num_training_steps = steps_per_epoch * epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"LR schedule: {num_warmup_steps} warmup steps, {num_training_steps} total steps")

    # Move validation data to device
    val_indices_dev = {k: v.to(device) for k, v in val_indices.items()}
    y_val_dev = y_val.to(device)
    qt_val_dev = qt_val.to(device)
    val_struct_dev = val_struct.to(device) if val_struct is not None else None

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device.type == 'cuda'))

    # Training state
    best_val_metric = float("inf")
    epochs_without_improvement = 0
    best_epoch = 0

    history = {
        "train_loss": [],
        "val_count_mae": [],
        "val_count_median_qerror": [],
        "val_count_within_2x": [],
        "val_dist_mae": [],
        "val_dist_acc_1hop": [],
        "lr": [],
        "epoch_time": [],
    }

    print(f"\n{'='*70}")
    print(f"Starting training for {epochs} epochs...")
    print(f"{'='*70}")
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Q-err':>7} | {'<2x':>6} | {'Dist':>6} | {'LR':>10} | {'Time':>6}")
    print(f"{'-'*70}")

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # Move batch to device
            node_type_idx = batch['node_type'].to(device, non_blocking=True)
            degree_bin_idx = batch['degree_bin'].to(device, non_blocking=True)
            radius_idx = batch['radius'].to(device, non_blocking=True)
            attr_name_idx = batch['attr_name'].to(device, non_blocking=True)
            attr_value_idx = batch['attr_value'].to(device, non_blocking=True)
            target_type_idx = batch['target_type'].to(device, non_blocking=True)
            max_hops_idx = batch['max_hops'].to(device, non_blocking=True)
            query_type_idx = batch['query_type'].to(device, non_blocking=True)
            y_batch = batch['y'].to(device, non_blocking=True)
            qt_batch = batch['qt'].to(device, non_blocking=True)
            struct_batch = batch.get('struct')
            if struct_batch is not None:
                struct_batch = struct_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
                # Encode query using learned embeddings (includes structural features)
                x = model.encode_query(
                    node_type_idx, degree_bin_idx, radius_idx,
                    attr_name_idx, attr_value_idx, target_type_idx,
                    max_hops_idx, query_type_idx, struct_batch
                )
                count_pred, dist_pred, latent = model(x)
                loss, _, _ = criterion(count_pred, y_batch, dist_pred, y_batch, qt_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= max(num_batches, 1)
        epoch_time = time.time() - epoch_start

        # Validation
        model.eval()
        with torch.no_grad():
            # Encode validation queries (includes structural features)
            x_val = model.encode_query(
                val_indices_dev['node_type'], val_indices_dev['degree_bin'],
                val_indices_dev['radius'], val_indices_dev['attr_name'],
                val_indices_dev['attr_value'], val_indices_dev['target_type'],
                val_indices_dev['max_hops'], val_indices_dev['query_type'],
                val_struct_dev
            )
            count_pred, dist_pred, _ = model(x_val)

            count_mask = qt_val_dev == 0
            dist_mask = qt_val_dev == 1

            # Count metrics
            if count_mask.any():
                count_metrics = compute_metrics(
                    count_pred[count_mask],
                    y_val_dev[count_mask],
                    use_log_transform=model_config.use_log_transform,
                )
                count_mae = count_metrics['mae']
                count_median_qerror = count_metrics['median_qerror']
                count_within_2x = count_metrics['within_2x']
            else:
                count_mae = count_median_qerror = count_within_2x = 0

            # Distance metrics
            if dist_mask.any():
                dist_errors = torch.abs(dist_pred[dist_mask] - y_val_dev[dist_mask])
                dist_mae = dist_errors.mean().item()
                dist_acc_1hop = (dist_errors <= 1.0).float().mean().item() * 100
            else:
                dist_mae = dist_acc_1hop = 0

        # Use median Q-error as primary validation metric
        val_metric = count_median_qerror

        current_lr = optimizer.param_groups[0]['lr']
        history["train_loss"].append(epoch_loss)
        history["val_count_mae"].append(count_mae)
        history["val_count_median_qerror"].append(count_median_qerror)
        history["val_count_within_2x"].append(count_within_2x)
        history["val_dist_mae"].append(dist_mae)
        history["val_dist_acc_1hop"].append(dist_acc_1hop)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)

        if val_metric < best_val_metric:
            best_val_metric = val_metric
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            if output_dir:
                torch.save(model.state_dict(), output_dir / "model.pt")
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0 or epoch == 0 or epochs_without_improvement == 0:
            marker = " *" if epochs_without_improvement == 0 else ""
            print(f"{epoch + 1:>6} | {epoch_loss:>8.4f} | {count_median_qerror:>7.2f} | "
                  f"{count_within_2x:>5.1f}% | {dist_mae:>6.2f} | {current_lr:>10.2e} | {epoch_time:>5.1f}s{marker}")

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    print(f"{'-'*70}")
    print(f"Best validation median Q-error: {best_val_metric:.3f} at epoch {best_epoch}")
    print(f"Total training time: {sum(history['epoch_time']):.1f}s")

    # Load best model
    if output_dir and (output_dir / "model.pt").exists():
        state_dict = torch.load(output_dir / "model.pt", weights_only=True)
        # Handle compiled model prefix
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    # Save training artifacts
    if output_dir:
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        model_info = {
            "model_version": "v2",
            "input_dim": encoder_config.input_dim_with_structural,
            "model_config": model_config.to_dict(),
            "num_params": num_params,
            "best_epoch": best_epoch,
            "best_val_median_qerror": best_val_metric,
            "best_val_count_within_2x": max(history["val_count_within_2x"]),
            "training_config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "grad_clip": grad_clip,
                "seed": seed,
            },
        }
        with open(output_dir / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train GraphSurrogate V2 model for NeurIPS-quality results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data options
    parser.add_argument("--graph", type=str, help="Path to graph file")
    parser.add_argument("--dataset", type=str, help="Dataset name from registry")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--num-train", type=int, default=100000, help="Training samples")
    parser.add_argument("--num-val", type=int, default=10000, help="Validation samples")
    parser.add_argument("--num-test", type=int, default=10000, help="Test samples")

    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training options
    parser.add_argument("--epochs", type=int, default=200, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")

    # Feature options
    parser.add_argument("--no-log-transform", action="store_true", help="Disable log transform")
    parser.add_argument("--no-structural", action="store_true", help="Disable structural features")
    parser.add_argument("--no-stratified", action="store_true", help="Disable stratified sampling")
    parser.add_argument("--use-mdn", action="store_true", help="Use MDN for uncertainty")

    args = parser.parse_args()

    if args.graph is None and args.dataset is None:
        parser.error("Either --graph or --dataset must be specified")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    # Load graph
    if args.dataset:
        from datasets import load_dataset
        print(f"Using dataset: {args.dataset}")
        G, meta_dict = load_dataset(args.dataset)
        metadata = GraphMetadata.from_dict(meta_dict)
    else:
        G, metadata = load_graph(args.graph)

    print(f"Graph: {metadata.num_nodes:,} nodes, {metadata.num_edges:,} edges")
    print(f"Node types: {metadata.node_types}")

    # Save graph
    save_graph(G, output_dir / "graph.gml", metadata)
    with open(output_dir / "graph_metadata.json", "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    # Prepare data
    (train_indices, y_train, qt_train, train_struct,
     val_indices, y_val, qt_val, val_struct,
     test_indices, y_test, qt_test, test_struct, config) = prepare_data(
        G, metadata,
        output_dir=output_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        seed=args.seed,
        use_stratified=not args.no_stratified,
        use_structural_features=not args.no_structural,
    )

    # Model config
    model_config = ModelConfig(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_log_transform=not args.no_log_transform,
        use_structural_features=not args.no_structural,
        use_mdn=args.use_mdn,
    )

    # Train model
    model = train_model(
        (train_indices, y_train, qt_train, train_struct),
        (val_indices, y_val, qt_val, val_struct),
        config,
        model_config=model_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
        patience=args.patience,
        seed=args.seed,
        use_amp=not args.no_amp,
        output_dir=output_dir,
    )

    # Final evaluation
    print(f"\n{'='*70}")
    print("Final Test Set Evaluation")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    model.eval()

    test_indices_dev = {k: v.to(device) for k, v in test_indices.items()}
    y_test_dev = y_test.to(device)
    qt_test_dev = qt_test.to(device)
    test_struct_dev = test_struct.to(device) if test_struct is not None else None

    with torch.no_grad():
        # Encode test queries (includes structural features)
        x_test = model.encode_query(
            test_indices_dev['node_type'], test_indices_dev['degree_bin'],
            test_indices_dev['radius'], test_indices_dev['attr_name'],
            test_indices_dev['attr_value'], test_indices_dev['target_type'],
            test_indices_dev['max_hops'], test_indices_dev['query_type'],
            test_struct_dev
        )
        count_pred, dist_pred, _ = model(x_test)

        count_mask = qt_test_dev == 0
        dist_mask = qt_test_dev == 1

        if count_mask.any():
            count_metrics = compute_metrics(
                count_pred[count_mask],
                y_test_dev[count_mask],
                use_log_transform=model_config.use_log_transform,
            )
            print(f"\nCount queries ({count_mask.sum().item()} samples):")
            print(f"  MAE:              {count_metrics['mae']:.2f}")
            print(f"  Median Q-error:   {count_metrics['median_qerror']:.2f}")
            print(f"  Mean Q-error:     {count_metrics['mean_qerror']:.2f}")
            print(f"  95th %ile Q-err:  {count_metrics['p95_qerror']:.2f}")
            print(f"  Within 2x:        {count_metrics['within_2x']:.1f}%")
            print(f"  Within 5x:        {count_metrics['within_5x']:.1f}%")

        if dist_mask.any():
            dist_errors = torch.abs(dist_pred[dist_mask] - y_test_dev[dist_mask])
            dist_mae = dist_errors.mean().item()
            dist_acc_exact = (dist_errors < 0.5).float().mean().item() * 100
            dist_acc_1hop = (dist_errors <= 1.0).float().mean().item() * 100
            dist_acc_2hop = (dist_errors <= 2.0).float().mean().item() * 100

            print(f"\nDistance queries ({dist_mask.sum().item()} samples):")
            print(f"  MAE:              {dist_mae:.2f}")
            print(f"  Exact (±0.5):     {dist_acc_exact:.1f}%")
            print(f"  Within 1 hop:     {dist_acc_1hop:.1f}%")
            print(f"  Within 2 hops:    {dist_acc_2hop:.1f}%")

    print(f"\n{'='*70}")
    print(f"Training complete. Artifacts saved to {output_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
