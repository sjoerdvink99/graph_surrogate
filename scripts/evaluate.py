import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from model.encoder import EncoderConfig, QueryEncoder
from model.network import GraphSurrogate
from training.graph_loader import load_graph
from training.query_sampler import QuerySampler, QueryType, execute_query


def load_model(model_dir: Path, device: Optional[str] = None) -> tuple:
    config = EncoderConfig.load(model_dir / "encoder_config.json")
    encoder = QueryEncoder(config)

    with open(model_dir / "model_info.json") as f:
        model_info = json.load(f)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = GraphSurrogate(
        input_dim=model_info["input_dim"],
        hidden_dim=model_info.get("hidden_dim", 128),
        latent_dim=model_info.get("latent_dim", 32),
        num_layers=model_info.get("num_layers", 3),
        dropout=model_info.get("dropout", 0.1),
    )
    state_dict = torch.load(model_dir / "model.pt", weights_only=True, map_location=device)
    # Handle torch.compile() prefix - strip "_orig_mod." if present
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, encoder, model_info, device


def evaluate_accuracy(
    model_dir: Path,
    num_samples: int = 2000,
    device: Optional[str] = None,
    graph_path: Optional[Path] = None,
) -> dict:
    model, encoder, model_info, device = load_model(model_dir, device)

    if graph_path:
        graph, _ = load_graph(graph_path)
    else:
        graph_file = model_dir / "graph.gml"
        if graph_file.exists():
            graph, _ = load_graph(str(graph_file))
        else:
            raise FileNotFoundError(f"No graph found in {model_dir}")

    sampler = QuerySampler(graph, seed=999)
    test_data = sampler.generate_dataset(num_samples)

    count_preds, count_trues = [], []
    dist_preds, dist_trues = [], []

    with torch.no_grad():
        for query, y_true in test_data:
            x = encoder.encode(query).unsqueeze(0).to(device)
            count_pred, dist_pred, _ = model(x)

            if query.query_type == QueryType.COUNT:
                count_preds.append(max(0.0, count_pred.item()))
                count_trues.append(float(y_true))
            else:
                dist_preds.append(max(0.0, dist_pred.item()))
                dist_trues.append(float(y_true))

    results = {"num_samples": num_samples}

    if count_preds:
        count_preds = np.array(count_preds)
        count_trues = np.array(count_trues)
        count_errors = np.abs(count_preds - count_trues)

        results.update({
            "count_samples": len(count_preds),
            "count_mae": float(np.mean(count_errors)),
            "count_rmse": float(np.sqrt(np.mean(count_errors ** 2))),
            "count_mape": float(np.mean(count_errors / (count_trues + 1)) * 100),
            "count_median_error": float(np.median(count_errors)),
            "count_p90_error": float(np.percentile(count_errors, 90)),
            "count_p99_error": float(np.percentile(count_errors, 99)),
            "count_max_error": float(np.max(count_errors)),
            "count_pearson_r": float(np.corrcoef(count_preds, count_trues)[0, 1]) if len(count_preds) > 1 else 0,
        })

    if dist_preds:
        dist_preds = np.array(dist_preds)
        dist_trues = np.array(dist_trues)
        dist_errors = np.abs(dist_preds - dist_trues)

        results.update({
            "dist_samples": len(dist_preds),
            "dist_mae": float(np.mean(dist_errors)),
            "dist_rmse": float(np.sqrt(np.mean(dist_errors ** 2))),
            "dist_median_error": float(np.median(dist_errors)),
            "dist_accuracy_exact": float(np.mean(dist_errors < 0.5) * 100),
            "dist_accuracy_1hop": float(np.mean(dist_errors <= 1.0) * 100),
            "dist_accuracy_2hop": float(np.mean(dist_errors <= 2.0) * 100),
            "dist_p90_error": float(np.percentile(dist_errors, 90)),
            "dist_p99_error": float(np.percentile(dist_errors, 99)),
        })

    return results


def evaluate_latency(
    model_dir: Path,
    num_runs: int = 1000,
    warmup_runs: int = 100,
    device: Optional[str] = None,
) -> dict:
    model, encoder, _, device = load_model(model_dir, device)

    graph_file = model_dir / "graph.gml"
    graph, _ = load_graph(str(graph_file))

    sampler = QuerySampler(graph, seed=777)
    queries = [q for q, _ in sampler.generate_dataset(num_runs + warmup_runs)]

    print(f"  Warming up ({warmup_runs} runs)...")
    for i in range(warmup_runs):
        x = encoder.encode(queries[i]).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    print(f"  Benchmarking model ({num_runs} runs)...")
    model_times = []
    for query in queries[warmup_runs:]:
        x = encoder.encode(query).unsqueeze(0).to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = model(x)

        if device == "cuda":
            torch.cuda.synchronize()
        model_times.append((time.perf_counter() - start) * 1000)

    print(f"  Benchmarking ground truth ({num_runs} runs)...")
    db_times = []
    for query in queries[warmup_runs:]:
        start = time.perf_counter()
        execute_query(graph, query)
        db_times.append((time.perf_counter() - start) * 1000)

    model_times = np.array(model_times)
    db_times = np.array(db_times)

    return {
        "num_runs": num_runs,
        "model_mean_ms": float(np.mean(model_times)),
        "model_std_ms": float(np.std(model_times)),
        "model_p50_ms": float(np.percentile(model_times, 50)),
        "model_p95_ms": float(np.percentile(model_times, 95)),
        "model_p99_ms": float(np.percentile(model_times, 99)),
        "model_min_ms": float(np.min(model_times)),
        "model_max_ms": float(np.max(model_times)),
        "db_mean_ms": float(np.mean(db_times)),
        "db_std_ms": float(np.std(db_times)),
        "db_p50_ms": float(np.percentile(db_times, 50)),
        "db_p95_ms": float(np.percentile(db_times, 95)),
        "db_p99_ms": float(np.percentile(db_times, 99)),
        "speedup_mean": float(np.mean(db_times) / np.mean(model_times)) if np.mean(model_times) > 0 else 0,
        "speedup_p99": float(np.percentile(db_times, 99) / np.percentile(model_times, 99)) if np.percentile(model_times, 99) > 0 else 0,
    }


def evaluate_memory(model_dir: Path) -> dict:
    graph_path = model_dir / "graph.gml"
    model_path = model_dir / "model.pt"
    config_path = model_dir / "encoder_config.json"

    graph_bytes = os.path.getsize(graph_path) if graph_path.exists() else 0
    model_bytes = os.path.getsize(model_path) if model_path.exists() else 0
    config_bytes = os.path.getsize(config_path) if config_path.exists() else 0

    return {
        "graph_size_kb": graph_bytes / 1024,
        "graph_size_mb": graph_bytes / (1024 * 1024),
        "model_size_kb": model_bytes / 1024,
        "model_size_mb": model_bytes / (1024 * 1024),
        "config_size_kb": config_bytes / 1024,
        "total_model_kb": (model_bytes + config_bytes) / 1024,
        "compression_ratio": graph_bytes / model_bytes if model_bytes > 0 else 0,
    }


def evaluate_batch_throughput(
    model_dir: Path,
    batch_sizes: list[int] = [1, 8, 32, 128, 512],
    num_batches: int = 100,
    device: Optional[str] = None,
) -> dict:
    model, encoder, _, device = load_model(model_dir, device)

    graph_file = model_dir / "graph.gml"
    graph, _ = load_graph(str(graph_file))

    sampler = QuerySampler(graph, seed=888)
    max_batch = max(batch_sizes)
    queries = [q for q, _ in sampler.generate_dataset(max_batch * num_batches)]

    results = {}

    for bs in batch_sizes:
        batch_times = []

        for i in range(num_batches):
            batch_queries = queries[i * bs:(i + 1) * bs]
            x_batch = torch.stack([encoder.encode(q) for q in batch_queries]).to(device)

            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                _ = model(x_batch)

            if device == "cuda":
                torch.cuda.synchronize()
            batch_times.append(time.perf_counter() - start)

        mean_time = np.mean(batch_times)
        throughput = bs / mean_time

        results[f"batch_{bs}"] = {
            "mean_time_ms": float(mean_time * 1000),
            "throughput_qps": float(throughput),
            "latency_per_query_ms": float(mean_time * 1000 / bs),
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GraphSurrogate model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=str, default="output", help="Model directory")
    parser.add_argument("--num-samples", type=int, default=2000, help="Accuracy evaluation samples")
    parser.add_argument("--latency-runs", type=int, default=1000, help="Latency benchmark runs")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--skip-latency", action="store_true", help="Skip latency benchmark")
    parser.add_argument("--batch-throughput", action="store_true", help="Run batch throughput test")
    parser.add_argument("--graph", type=str, default=None, help="Alternative graph for evaluation")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    with open(model_dir / "model_info.json") as f:
        model_info = json.load(f)

    print("=" * 70)
    print("GraphSurrogate Evaluation")
    print("=" * 70)

    print("\nModel Architecture")
    print("-" * 50)
    print(f"  Parameters:    {model_info.get('num_params', 'N/A'):,}")
    print(f"  Hidden dim:    {model_info.get('hidden_dim', 128)}")
    print(f"  Latent dim:    {model_info.get('latent_dim', 32)}")
    print(f"  Num layers:    {model_info.get('num_layers', 3)}")
    print(f"  Dropout:       {model_info.get('dropout', 0.1)}")

    if 'best_epoch' in model_info:
        print(f"  Best epoch:    {model_info['best_epoch']}")

    print("\nAccuracy Evaluation")
    print("-" * 50)
    print(f"  Running {args.num_samples} test queries...")
    graph_path = Path(args.graph) if args.graph else None
    acc = evaluate_accuracy(model_dir, num_samples=args.num_samples, device=args.device, graph_path=graph_path)

    print(f"\n  Count Queries ({acc.get('count_samples', 0)} samples):")
    print(f"    MAE:            {acc.get('count_mae', 0):.2f}")
    print(f"    RMSE:           {acc.get('count_rmse', 0):.2f}")
    print(f"    MAPE:           {acc.get('count_mape', 0):.1f}%")
    print(f"    Median error:   {acc.get('count_median_error', 0):.2f}")
    print(f"    P90 error:      {acc.get('count_p90_error', 0):.2f}")
    print(f"    P99 error:      {acc.get('count_p99_error', 0):.2f}")
    print(f"    Pearson r:      {acc.get('count_pearson_r', 0):.3f}")

    print(f"\n  Distance Queries ({acc.get('dist_samples', 0)} samples):")
    print(f"    MAE:            {acc.get('dist_mae', 0):.2f}")
    print(f"    RMSE:           {acc.get('dist_rmse', 0):.2f}")
    print(f"    Median error:   {acc.get('dist_median_error', 0):.2f}")
    print(f"    Exact (±0.5):   {acc.get('dist_accuracy_exact', 0):.1f}%")
    print(f"    Within 1 hop:   {acc.get('dist_accuracy_1hop', 0):.1f}%")
    print(f"    Within 2 hops:  {acc.get('dist_accuracy_2hop', 0):.1f}%")

    if not args.skip_latency:
        print("\nLatency Evaluation")
        print("-" * 50)
        lat = evaluate_latency(model_dir, num_runs=args.latency_runs, device=args.device)

        print(f"\n  Model Inference:")
        print(f"    Mean:   {lat['model_mean_ms']:.3f} ms")
        print(f"    Median: {lat['model_p50_ms']:.3f} ms")
        print(f"    P95:    {lat['model_p95_ms']:.3f} ms")
        print(f"    P99:    {lat['model_p99_ms']:.3f} ms")

        print(f"\n  Ground Truth (NetworkX):")
        print(f"    Mean:   {lat['db_mean_ms']:.3f} ms")
        print(f"    Median: {lat['db_p50_ms']:.3f} ms")
        print(f"    P95:    {lat['db_p95_ms']:.3f} ms")
        print(f"    P99:    {lat['db_p99_ms']:.3f} ms")

        print(f"\n  Speedup:")
        print(f"    Mean:   {lat['speedup_mean']:.1f}x")
        print(f"    P99:    {lat['speedup_p99']:.1f}x")
    else:
        lat = {}

    if args.batch_throughput:
        print("\nBatch Throughput")
        print("-" * 50)
        batch_results = evaluate_batch_throughput(model_dir, device=args.device)

        print(f"  {'Batch Size':>12} | {'Latency/query':>14} | {'Throughput':>12}")
        print(f"  {'-'*12}-+-{'-'*14}-+-{'-'*12}")
        for key, val in batch_results.items():
            bs = key.split("_")[1]
            print(f"  {bs:>12} | {val['latency_per_query_ms']:>11.3f} ms | {val['throughput_qps']:>9,.0f} q/s")
    else:
        batch_results = {}

    print("\nMemory Usage")
    print("-" * 50)
    mem = evaluate_memory(model_dir)
    print(f"  Graph data:     {mem['graph_size_mb']:.2f} MB")
    print(f"  Model weights:  {mem['model_size_kb']:.1f} KB")
    print(f"  Total model:    {mem['total_model_kb']:.1f} KB")
    print(f"  Compression:    {mem['compression_ratio']:.0f}x")

    print("\n" + "=" * 70)

    results = {
        "model": model_info,
        "accuracy": acc,
        "latency": lat,
        "memory": mem,
        "batch_throughput": batch_results,
    }
    output_path = model_dir / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
