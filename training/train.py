import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from tqdm import tqdm

from model.encoder import EncoderConfig, QueryEncoder
from model.network import GraphSurrogate, TwoHeadLoss

from .graph_loader import load_graph, save_graph, GraphMetadata
from .query_sampler import QuerySampler, QueryType, ChunkedQueryGenerator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_encoder_config(G, node_types: list[str], metadata: GraphMetadata) -> EncoderConfig:
    attr_values = {}
    for _, data in G.nodes(data=True):
        for key, val in data.items():
            if key not in ("node_type", "degree_bin") and val is not None:
                if key not in attr_values:
                    attr_values[key] = set()
                attr_values[key].add(val)

    attr_values["node_type"] = set(node_types)

    attribute_values = {}
    for k, v in attr_values.items():
        if len(v) <= 100:
            attribute_values[k] = sorted(v, key=str)

    attribute_names = sorted(attribute_values.keys())

    return EncoderConfig(
        node_types=node_types,
        attribute_names=attribute_names,
        attribute_values=attribute_values,
    )


def compute_target_statistics(y_train: torch.Tensor, qt_train: torch.Tensor) -> dict:
    count_mask = qt_train == 0
    dist_mask = qt_train == 1

    stats = {}

    if count_mask.any():
        count_vals = y_train[count_mask]
        stats['count_mean'] = count_vals.mean().item()
        stats['count_std'] = count_vals.std().item() + 1e-8
        stats['count_min'] = count_vals.min().item()
        stats['count_max'] = count_vals.max().item()
        stats['count_median'] = count_vals.median().item()
        stats['count_p90'] = torch.quantile(count_vals.float(), 0.9).item()
        stats['count_p99'] = torch.quantile(count_vals.float(), 0.99).item()
    else:
        stats.update({f'count_{k}': 0.0 for k in ['mean', 'std', 'min', 'max', 'median', 'p90', 'p99']})

    if dist_mask.any():
        dist_vals = y_train[dist_mask]
        stats['dist_mean'] = dist_vals.mean().item()
        stats['dist_std'] = dist_vals.std().item() + 1e-8
        stats['dist_min'] = dist_vals.min().item()
        stats['dist_max'] = dist_vals.max().item()
        stats['dist_median'] = dist_vals.median().item()
        unique, counts = torch.unique(dist_vals, return_counts=True)
        stats['dist_distribution'] = {int(u.item()): int(c.item()) for u, c in zip(unique, counts)}
    else:
        stats.update({f'dist_{k}': 0.0 for k in ['mean', 'std', 'min', 'max', 'median']})
        stats['dist_distribution'] = {}

    return stats


def prepare_data_inmemory(
    G,
    metadata: GraphMetadata,
    num_train: int = 50000,
    num_val: int = 5000,
    num_test: int = 10000,
    output_dir: Path | None = None,
    seed: int = 42,
) -> tuple:
    node_types = metadata.node_types
    config = create_encoder_config(G, node_types, metadata)
    encoder = QueryEncoder(config)

    print(f"\nSampling queries...")
    print(f"  Training: {num_train:,} queries (seed={seed})")
    train_sampler = QuerySampler(G, seed=seed, node_types=node_types)
    train_data = train_sampler.generate_dataset(num_train)

    print(f"  Validation: {num_val:,} queries (seed={seed+1})")
    val_sampler = QuerySampler(G, seed=seed + 1, node_types=node_types)
    val_data = val_sampler.generate_dataset(num_val)

    print(f"  Test: {num_test:,} queries (seed={seed+2})")
    test_sampler = QuerySampler(G, seed=seed + 2, node_types=node_types)
    test_data = test_sampler.generate_dataset(num_test)

    print("Encoding queries...")
    X_train = encoder.encode_batch([q for q, _ in train_data])
    y_train = torch.tensor([r for _, r in train_data], dtype=torch.float32)
    qt_train = torch.tensor([0 if q.query_type == QueryType.COUNT else 1 for q, _ in train_data])

    X_val = encoder.encode_batch([q for q, _ in val_data])
    y_val = torch.tensor([r for _, r in val_data], dtype=torch.float32)
    qt_val = torch.tensor([0 if q.query_type == QueryType.COUNT else 1 for q, _ in val_data])

    X_test = encoder.encode_batch([q for q, _ in test_data])
    y_test = torch.tensor([r for _, r in test_data], dtype=torch.float32)
    qt_test = torch.tensor([0 if q.query_type == QueryType.COUNT else 1 for q, _ in test_data])

    target_stats = compute_target_statistics(y_train, qt_train)
    print(f"\nTarget statistics (training set):")
    print(f"  Count:    mean={target_stats['count_mean']:.1f}, std={target_stats['count_std']:.1f}, "
          f"range=[{target_stats['count_min']:.0f}, {target_stats['count_max']:.0f}]")
    print(f"  Distance: mean={target_stats['dist_mean']:.1f}, std={target_stats['dist_std']:.1f}, "
          f"range=[{target_stats['dist_min']:.0f}, {target_stats['dist_max']:.0f}]")

    if output_dir:
        config.save(output_dir / "encoder_config.json")
        torch.save({
            "X": X_test, "y": y_test, "qt": qt_test,
            "X_val": X_val, "y_val": y_val, "qt_val": qt_val,
        }, output_dir / "test_data.pt")

        test_queries_json = [q.to_dict() | {"result": r} for q, r in test_data]
        with open(output_dir / "test_queries.json", "w") as f:
            json.dump(test_queries_json, f, indent=2)

    return X_train, y_train, qt_train, X_val, y_val, qt_val, X_test, y_test, qt_test, config


class StreamingQueryDataset(IterableDataset):
    def __init__(
        self,
        G,
        encoder: QueryEncoder,
        num_samples: int,
        seed: int,
        node_types: list[str],
    ):
        self.G = G
        self.encoder = encoder
        self.num_samples = num_samples
        self.seed = seed
        self.node_types = node_types

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            seed = self.seed
            num_samples = self.num_samples
        else:
            per_worker = self.num_samples // worker_info.num_workers
            seed = self.seed + worker_info.id
            num_samples = per_worker

        sampler = QuerySampler(self.G, seed=seed, node_types=self.node_types)
        from .query_sampler import execute_query

        random.seed(seed)
        for _ in range(num_samples):
            query = sampler.sample_query()
            result = execute_query(self.G, query)

            x = self.encoder.encode(query)
            y = torch.tensor(result, dtype=torch.float32)
            qt = torch.tensor(0 if query.query_type == QueryType.COUNT else 1)

            yield x, y, qt


def prepare_data_streaming(
    G,
    metadata: GraphMetadata,
    num_train: int,
    num_val: int,
    num_test: int,
    output_dir: Path | None,
    seed: int,
) -> tuple:
    node_types = metadata.node_types
    config = create_encoder_config(G, node_types, metadata)
    encoder = QueryEncoder(config)

    if output_dir:
        config.save(output_dir / "encoder_config.json")

    print(f"\nGenerating validation data ({num_val:,} queries)...")
    val_sampler = QuerySampler(G, seed=seed + 1, node_types=node_types)
    val_data = val_sampler.generate_dataset(num_val)
    X_val = encoder.encode_batch([q for q, _ in val_data])
    y_val = torch.tensor([r for _, r in val_data], dtype=torch.float32)
    qt_val = torch.tensor([0 if q.query_type == QueryType.COUNT else 1 for q, _ in val_data])

    print(f"Generating test data ({num_test:,} queries)...")
    test_sampler = QuerySampler(G, seed=seed + 2, node_types=node_types)
    test_data = test_sampler.generate_dataset(num_test)
    X_test = encoder.encode_batch([q for q, _ in test_data])
    y_test = torch.tensor([r for _, r in test_data], dtype=torch.float32)
    qt_test = torch.tensor([0 if q.query_type == QueryType.COUNT else 1 for q, _ in test_data])

    if output_dir:
        torch.save({
            "X": X_test, "y": y_test, "qt": qt_test,
            "X_val": X_val, "y_val": y_val, "qt_val": qt_val,
        }, output_dir / "test_data.pt")

        test_queries_json = [q.to_dict() | {"result": r} for q, r in test_data]
        with open(output_dir / "test_queries.json", "w") as f:
            json.dump(test_queries_json, f, indent=2)

    train_dataset = StreamingQueryDataset(G, encoder, num_train, seed, node_types)

    return train_dataset, X_val, y_val, qt_val, X_test, y_test, qt_test, config


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(
    train_data,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    qt_val: torch.Tensor,
    config: EncoderConfig,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 3e-4,
    hidden_dim: int = 128,
    latent_dim: int = 32,
    num_layers: int = 3,
    dropout: float = 0.1,
    weight_decay: float = 1e-2,
    warmup_ratio: float = 0.1,
    grad_clip: float = 1.0,
    patience: int = 30,
    seed: int = 42,
    use_amp: bool = True,
    output_dir: Path | None = None,
    streaming: bool = False,
    num_workers: int = 0,
) -> GraphSurrogate:
    set_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        # Enable TF32 for Ampere+ GPUs (faster matmul with minimal precision loss)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable cuDNN autotuner for optimal convolution algorithms
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple MPS"
        use_amp = False
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        use_amp = False

    print(f"\n{'='*70}")
    print(f"Training Configuration")
    print(f"{'='*70}")
    print(f"Device: {device_name}")
    print(f"Mixed precision: {use_amp}")
    print(f"Streaming mode: {streaming}")
    print(f"Model: hidden_dim={hidden_dim}, latent_dim={latent_dim}, num_layers={num_layers}")
    print(f"Training: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"Regularization: dropout={dropout}, weight_decay={weight_decay}")

    if streaming:
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(device.type == 'cuda'),
        )
        num_train_samples = len(train_data)
    else:
        X_train, y_train, qt_train = train_data
        target_stats = compute_target_statistics(y_train, qt_train)
        train_loader = DataLoader(
            TensorDataset(X_train, y_train, qt_train),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=(device.type == 'cuda'),
            num_workers=num_workers,
        )
        num_train_samples = len(X_train)

    model = GraphSurrogate(
        input_dim=config.input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    model = model.to(device)

    # Compile model for faster execution (PyTorch 2.0+)
    if device.type == 'cuda' and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled with torch.compile()")
        except Exception:
            pass  # Fall back to eager mode if compilation fails

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,} total, {trainable_params:,} trainable")
    print(f"Training samples: {num_train_samples:,}")

    criterion = TwoHeadLoss(
        count_delta=10.0,
        distance_delta=1.0,
        learn_weights=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    steps_per_epoch = (num_train_samples + batch_size - 1) // batch_size
    num_training_steps = steps_per_epoch * epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"LR schedule: {num_warmup_steps} warmup steps, {num_training_steps} total steps")

    X_val_dev = X_val.to(device)
    y_val_dev = y_val.to(device)
    qt_val_dev = qt_val.to(device)

    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device.type == 'cuda'))

    best_val_mae = float("inf")
    epochs_without_improvement = 0
    best_epoch = 0

    history = {
        "train_loss": [],
        "val_count_mae": [],
        "val_dist_mae": [],
        "val_combined_mae": [],
        "lr": [],
        "epoch_time": [],
    }

    print(f"\n{'='*70}")
    print(f"Starting training for {epochs} epochs...")
    print(f"{'='*70}")
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Count MAE':>10} | {'Dist MAE':>10} | {'LR':>10} | {'Time':>6}")
    print(f"{'-'*70}")

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            if len(batch) == 3:
                X_batch, y_batch, qt_batch = batch
            else:
                X_batch, y_batch, qt_batch = batch[0], batch[1], batch[2]

            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            qt_batch = qt_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
                count_pred, dist_pred, _ = model(X_batch)
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

        model.eval()
        with torch.no_grad():
            count_pred, dist_pred, _ = model(X_val_dev)

            count_mask = qt_val_dev == 0
            dist_mask = qt_val_dev == 1

            count_mae = torch.abs(count_pred[count_mask] - y_val_dev[count_mask]).mean().item() if count_mask.any() else 0
            dist_mae = torch.abs(dist_pred[dist_mask] - y_val_dev[dist_mask]).mean().item() if dist_mask.any() else 0
            combined_mae = (count_mae + dist_mae) / 2

        current_lr = optimizer.param_groups[0]['lr']
        history["train_loss"].append(epoch_loss)
        history["val_count_mae"].append(count_mae)
        history["val_dist_mae"].append(dist_mae)
        history["val_combined_mae"].append(combined_mae)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)

        if combined_mae < best_val_mae:
            best_val_mae = combined_mae
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            if output_dir:
                torch.save(model.state_dict(), output_dir / "model.pt")
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0 or epoch == 0 or epochs_without_improvement == 0:
            marker = " *" if epochs_without_improvement == 0 else ""
            print(f"{epoch + 1:>6} | {epoch_loss:>8.4f} | {count_mae:>10.2f} | {dist_mae:>10.2f} | {current_lr:>10.2e} | {epoch_time:>5.1f}s{marker}")

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    print(f"{'-'*70}")
    print(f"Best validation MAE: {best_val_mae:.3f} at epoch {best_epoch}")
    print(f"Total training time: {sum(history['epoch_time']):.1f}s")

    if output_dir and (output_dir / "model.pt").exists():
        model.load_state_dict(torch.load(output_dir / "model.pt", weights_only=True))

    if output_dir:
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        model_info = {
            "input_dim": config.input_dim,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "num_params": num_params,
            "best_epoch": best_epoch,
            "best_val_count_mae": min(history["val_count_mae"]),
            "best_val_dist_mae": min(history["val_dist_mae"]),
            "best_val_combined_mae": best_val_mae,
            "training_config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "grad_clip": grad_clip,
                "seed": seed,
                "streaming": streaming,
            },
        }
        with open(output_dir / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train GraphSurrogate model for graph aggregate query approximation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--graph", type=str, help="Path to graph file (JSON, edge list, GML, GraphML)")
    parser.add_argument("--dataset", type=str, help="Dataset name from registry (bron, amazon, youtube, etc.)")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--num-train", type=int, default=100000, help="Training samples")
    parser.add_argument("--num-val", type=int, default=10000, help="Validation samples")
    parser.add_argument("--num-test", type=int, default=10000, help="Test samples")

    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of encoder layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    parser.add_argument("--epochs", type=int, default=200, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")

    parser.add_argument("--streaming", action="store_true", help="Use streaming mode for large datasets")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (0 for main process only)")

    args = parser.parse_args()

    if args.graph is None and args.dataset is None:
        parser.error("Either --graph or --dataset must be specified")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if args.dataset:
        from datasets import download_dataset, load_dataset
        print(f"Using dataset: {args.dataset}")
        G, meta_dict = load_dataset(args.dataset)
        metadata = GraphMetadata.from_dict(meta_dict)
        graph_path = None
    else:
        G, metadata = load_graph(args.graph)
        graph_path = args.graph

    print(f"Graph: {metadata.num_nodes:,} nodes, {metadata.num_edges:,} edges")
    print(f"Node types: {metadata.node_types}")

    save_graph(G, output_dir / "graph.gml", metadata)

    with open(output_dir / "graph_metadata.json", "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    if args.streaming:
        print("\nUsing streaming mode for memory efficiency...")
        train_data, X_val, y_val, qt_val, X_test, y_test, qt_test, config = prepare_data_streaming(
            G, metadata,
            num_train=args.num_train,
            num_val=args.num_val,
            num_test=args.num_test,
            output_dir=output_dir,
            seed=args.seed,
        )
        streaming = True
    else:
        X_train, y_train, qt_train, X_val, y_val, qt_val, X_test, y_test, qt_test, config = prepare_data_inmemory(
            G, metadata,
            num_train=args.num_train,
            num_val=args.num_val,
            num_test=args.num_test,
            output_dir=output_dir,
            seed=args.seed,
        )
        train_data = (X_train, y_train, qt_train)
        streaming = False

    if args.streaming:
        gc.collect()

    model = train_model(
        train_data,
        X_val, y_val, qt_val,
        config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
        patience=args.patience,
        seed=args.seed,
        use_amp=not args.no_amp,
        output_dir=output_dir,
        streaming=streaming,
        num_workers=args.num_workers,
    )

    print(f"\n{'='*70}")
    print("Final Test Set Evaluation")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    model.eval()

    X_test_dev = X_test.to(device)
    y_test_dev = y_test.to(device)
    qt_test_dev = qt_test.to(device)

    with torch.no_grad():
        count_pred, dist_pred, _ = model(X_test_dev)

        count_mask = qt_test_dev == 0
        dist_mask = qt_test_dev == 1

        if count_mask.any():
            count_errors = torch.abs(count_pred[count_mask] - y_test_dev[count_mask])
            count_mae = count_errors.mean().item()
            count_rmse = torch.sqrt((count_errors ** 2).mean()).item()
            count_mape = (count_errors / (y_test_dev[count_mask].abs() + 1)).mean().item() * 100
        else:
            count_mae = count_rmse = count_mape = 0

        if dist_mask.any():
            dist_errors = torch.abs(dist_pred[dist_mask] - y_test_dev[dist_mask])
            dist_mae = dist_errors.mean().item()
            dist_rmse = torch.sqrt((dist_errors ** 2).mean()).item()
            dist_acc_1 = (dist_errors <= 1.0).float().mean().item() * 100
        else:
            dist_mae = dist_rmse = dist_acc_1 = 0

    print(f"Count queries ({count_mask.sum().item()} samples):")
    print(f"  MAE:  {count_mae:.2f}")
    print(f"  RMSE: {count_rmse:.2f}")
    print(f"  MAPE: {count_mape:.1f}%")
    print(f"\nDistance queries ({dist_mask.sum().item()} samples):")
    print(f"  MAE:  {dist_mae:.2f}")
    print(f"  RMSE: {dist_rmse:.2f}")
    print(f"  Accuracy (±1 hop): {dist_acc_1:.1f}%")

    print(f"\n{'='*70}")
    print(f"Training complete. Artifacts saved to {output_dir}/")
    print(f"  - model.pt: Model weights")
    print(f"  - model_info.json: Architecture and metrics")
    print(f"  - training_history.json: Training curves")
    print(f"  - encoder_config.json: Query encoder config")
    print(f"  - graph.gml: Graph data")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
