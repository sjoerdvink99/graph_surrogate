"""
Comprehensive evaluation script for NeurIPS submission.

Generates all metrics, analyses, and comparisons needed for publication.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from model.encoder import EncoderConfig, QueryEncoder
from model.network import GraphSurrogate, ModelConfig
from model.structural_features import compute_or_load_features
from training.graph_loader import load_graph
from training.query_sampler import QuerySampler, QueryType, execute_query
from training.query_sampler import StratifiedQuerySampler, SamplingConfig


def compute_qerror(pred: np.ndarray, target: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """Compute Q-error: max(pred/target, target/pred)."""
    pred = np.maximum(pred, epsilon)
    target = np.maximum(target, epsilon)
    return np.maximum(pred / target, target / pred)


def load_model(model_dir: Path, device: Optional[str] = None):
    """Load GraphSurrogate model."""
    # Load config
    config_path = model_dir / "encoder_config.json"
    encoder_config = EncoderConfig.load(config_path)
    encoder = QueryEncoder(encoder_config)

    # Load model info
    with open(model_dir / "model_info.json") as f:
        model_info = json.load(f)

    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Create model
    if "model_config" in model_info:
        model_config = ModelConfig.from_dict(model_info["model_config"])
    else:
        model_config = ModelConfig()

    # Update dimensions from encoder
    model_config.num_node_types = len(encoder_config.node_types)
    model_config.num_degree_bins = len(encoder_config.degree_bins) + 1
    model_config.num_radii = len(encoder_config.radii) + 1
    model_config.num_attr_names = len(encoder_config.attribute_names) + 1
    model_config.num_attr_values = sum(len(v) for v in encoder_config.attribute_values.values()) + 1
    model_config.num_max_hops = len(encoder_config.max_hops_options) + 1

    model = GraphSurrogate(model_config)

    # Load weights
    state_dict = torch.load(model_dir / "model.pt", weights_only=True, map_location=device)
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, encoder, encoder_config, model_config, device


def encode_queries_batch(
    encoder: QueryEncoder,
    model: GraphSurrogate,
    queries: list,
    structural_computer,
    device: str,
) -> torch.Tensor:
    """Encode a batch of queries using index-based encoding with learned embeddings."""
    indices = encoder.encode_indices_batch(queries)
    indices_dev = {k: v.to(device) for k, v in indices.items()}

    # Get structural features if available
    struct_feat = None
    if structural_computer is not None:
        node_ids = [q.start_node_id for q in queries]
        struct_feat = structural_computer.get_features_batch(node_ids).to(device)

    # Encode using model's learned embeddings
    x = model.encode_query(
        indices_dev['node_type'],
        indices_dev['degree_bin'],
        indices_dev['radius'],
        indices_dev['attr_name'],
        indices_dev['attr_value'],
        indices_dev['target_type'],
        indices_dev['max_hops'],
        indices_dev['query_type'],
        struct_feat,
    )
    return x


def evaluate_accuracy_comprehensive(
    model_dir: Path,
    num_samples: int = 5000,
    device: Optional[str] = None,
    batch_size: int = 256,
) -> dict:
    """
    Comprehensive accuracy evaluation with Q-error metrics.
    """
    model, encoder, encoder_config, model_config, device = load_model(model_dir, device)

    # Load graph
    graph_path = model_dir / "graph.gml"
    G, metadata = load_graph(str(graph_path))

    # Load structural features if used
    structural_computer = None
    if model_config.use_structural_features:
        structural_computer = compute_or_load_features(G, model_dir, show_progress=False)

    # Generate test queries with stratified sampling
    sampler = StratifiedQuerySampler(G, seed=999, node_types=metadata.node_types)
    test_data = sampler.generate_stratified_dataset(num_samples, show_progress=True)

    # Evaluate in batches
    count_preds, count_trues = [], []
    dist_preds, dist_trues = [], []
    count_degrees, dist_degrees = [], []

    with torch.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
            batch_data = test_data[i:i + batch_size]
            batch_queries = [q for q, _ in batch_data]
            batch_trues = [y for _, y in batch_data]

            # Encode batch using learned embeddings
            x = encode_queries_batch(encoder, model, batch_queries, structural_computer, device)

            # Forward pass
            count_pred, dist_pred, _ = model(x)

            for j, (query, y_true) in enumerate(zip(batch_queries, batch_trues)):
                if query.query_type == QueryType.COUNT:
                    if model_config.use_log_transform:
                        pred = torch.expm1(count_pred[j]).clamp(min=0).item()
                    else:
                        pred = max(0, count_pred[j].item())
                    count_preds.append(pred)
                    count_trues.append(float(y_true))
                    count_degrees.append(query.start_degree_bin)
                else:
                    dist_preds.append(max(0, dist_pred[j].item()))
                    dist_trues.append(float(y_true))
                    dist_degrees.append(query.start_degree_bin)

    results = {"num_samples": num_samples}

    # Count metrics
    if count_preds:
        count_preds = np.array(count_preds)
        count_trues = np.array(count_trues)
        count_degrees = np.array(count_degrees)
        count_errors = np.abs(count_preds - count_trues)
        count_qerror = compute_qerror(count_preds, count_trues)

        results["count"] = {
            "num_samples": len(count_preds),
            "mae": float(np.mean(count_errors)),
            "rmse": float(np.sqrt(np.mean(count_errors ** 2))),
            "mape": float(np.mean(count_errors / (count_trues + 1)) * 100),
            "median_error": float(np.median(count_errors)),
            "median_qerror": float(np.median(count_qerror)),
            "mean_qerror": float(np.mean(count_qerror)),
            "p50_qerror": float(np.percentile(count_qerror, 50)),
            "p75_qerror": float(np.percentile(count_qerror, 75)),
            "p90_qerror": float(np.percentile(count_qerror, 90)),
            "p95_qerror": float(np.percentile(count_qerror, 95)),
            "p99_qerror": float(np.percentile(count_qerror, 99)),
            "max_qerror": float(np.max(count_qerror)),
            "within_1.5x": float(np.mean(count_qerror <= 1.5) * 100),
            "within_2x": float(np.mean(count_qerror <= 2) * 100),
            "within_3x": float(np.mean(count_qerror <= 3) * 100),
            "within_5x": float(np.mean(count_qerror <= 5) * 100),
            "within_10x": float(np.mean(count_qerror <= 10) * 100),
            "pearson_r": float(np.corrcoef(count_preds, count_trues)[0, 1]) if len(count_preds) > 1 else 0,
        }

        # By degree stratum
        results["count_by_degree"] = {}
        for deg in np.unique(count_degrees):
            mask = count_degrees == deg
            if mask.sum() > 10:
                qe = count_qerror[mask]
                results["count_by_degree"][int(deg)] = {
                    "num_samples": int(mask.sum()),
                    "median_qerror": float(np.median(qe)),
                    "within_2x": float(np.mean(qe <= 2) * 100),
                }

    # Distance metrics
    if dist_preds:
        dist_preds = np.array(dist_preds)
        dist_trues = np.array(dist_trues)
        dist_errors = np.abs(dist_preds - dist_trues)

        results["distance"] = {
            "num_samples": len(dist_preds),
            "mae": float(np.mean(dist_errors)),
            "rmse": float(np.sqrt(np.mean(dist_errors ** 2))),
            "median_error": float(np.median(dist_errors)),
            "accuracy_exact": float(np.mean(dist_errors < 0.5) * 100),
            "accuracy_1hop": float(np.mean(dist_errors <= 1.0) * 100),
            "accuracy_2hop": float(np.mean(dist_errors <= 2.0) * 100),
            "p90_error": float(np.percentile(dist_errors, 90)),
            "p95_error": float(np.percentile(dist_errors, 95)),
            "p99_error": float(np.percentile(dist_errors, 99)),
        }

    return results


def evaluate_uncertainty_calibration(
    model_dir: Path,
    num_samples: int = 2000,
    mc_samples: int = 30,
    device: Optional[str] = None,
) -> dict:
    """
    Evaluate uncertainty calibration using MC-Dropout.
    """
    model, encoder, encoder_config, model_config, device = load_model(model_dir, device)

    # Load graph
    graph_path = model_dir / "graph.gml"
    G, metadata = load_graph(str(graph_path))

    structural_computer = None
    if model_config.use_structural_features:
        structural_computer = compute_or_load_features(G, model_dir, show_progress=False)

    sampler = StratifiedQuerySampler(G, seed=888, node_types=metadata.node_types)
    test_data = sampler.generate_dataset(num_samples)

    # Collect predictions with uncertainty
    predictions = []
    uncertainties = []
    actuals = []
    query_types_list = []

    model.train()  # Enable dropout

    for query, y_true in tqdm(test_data, desc="MC-Dropout sampling"):
        # Encode using index-based encoding
        x = encode_queries_batch(encoder, model, [query], structural_computer, device)

        # MC-Dropout sampling
        samples = []
        with torch.no_grad():
            for _ in range(mc_samples):
                count_pred, dist_pred, _ = model(x)
                if query.query_type == QueryType.COUNT:
                    if model_config.use_log_transform:
                        pred = torch.expm1(count_pred[0]).clamp(min=0).item()
                    else:
                        pred = count_pred[0].item()
                else:
                    pred = dist_pred[0].item()
                samples.append(pred)

        samples = np.array(samples)
        predictions.append(np.mean(samples))
        uncertainties.append(np.std(samples))
        actuals.append(float(y_true))
        query_types_list.append(0 if query.query_type == QueryType.COUNT else 1)

    model.eval()

    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)
    actuals = np.array(actuals)
    query_types = np.array(query_types_list)

    # Compute calibration metrics
    errors = np.abs(predictions - actuals)

    # Expected calibration error
    num_bins = 10
    bin_boundaries = np.linspace(0, np.max(uncertainties) + 1e-8, num_bins + 1)
    calibration_data = []

    for i in range(num_bins):
        mask = (uncertainties >= bin_boundaries[i]) & (uncertainties < bin_boundaries[i + 1])
        if mask.sum() > 0:
            mean_uncertainty = uncertainties[mask].mean()
            mean_error = errors[mask].mean()
            calibration_data.append({
                "bin": i,
                "mean_uncertainty": float(mean_uncertainty),
                "mean_error": float(mean_error),
                "count": int(mask.sum()),
            })

    # Correlation between uncertainty and error
    uncertainty_error_corr = float(np.corrcoef(uncertainties, errors)[0, 1]) if len(uncertainties) > 1 else 0

    return {
        "num_samples": num_samples,
        "mc_samples": mc_samples,
        "uncertainty_error_correlation": uncertainty_error_corr,
        "mean_uncertainty": float(np.mean(uncertainties)),
        "calibration_bins": calibration_data,
    }


def evaluate_latency_comprehensive(
    model_dir: Path,
    num_runs: int = 2000,
    warmup_runs: int = 100,
    device: Optional[str] = None,
) -> dict:
    """
    Comprehensive latency evaluation.
    """
    model, encoder, encoder_config, model_config, device = load_model(model_dir, device)

    graph_path = model_dir / "graph.gml"
    G, metadata = load_graph(str(graph_path))

    structural_computer = None
    if model_config.use_structural_features:
        structural_computer = compute_or_load_features(G, model_dir, show_progress=False)

    sampler = QuerySampler(G, seed=777)
    queries = [q for q, _ in sampler.generate_dataset(num_runs + warmup_runs)]

    # Warmup
    print(f"  Warming up ({warmup_runs} runs)...")
    for i in range(warmup_runs):
        x = encode_queries_batch(encoder, model, [queries[i]], structural_computer, device)
        with torch.no_grad():
            _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark model
    print(f"  Benchmarking model ({num_runs} runs)...")
    model_times = []
    for query in tqdm(queries[warmup_runs:], desc="Model inference"):
        x = encode_queries_batch(encoder, model, [query], structural_computer, device)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = model(x)

        if device == "cuda":
            torch.cuda.synchronize()
        model_times.append((time.perf_counter() - start) * 1000)

    # Benchmark ground truth
    print(f"  Benchmarking ground truth ({num_runs} runs)...")
    gt_times = []
    for query in tqdm(queries[warmup_runs:], desc="Ground truth"):
        start = time.perf_counter()
        execute_query(G, query)
        gt_times.append((time.perf_counter() - start) * 1000)

    model_times = np.array(model_times)
    gt_times = np.array(gt_times)

    return {
        "num_runs": num_runs,
        "model": {
            "mean_ms": float(np.mean(model_times)),
            "std_ms": float(np.std(model_times)),
            "median_ms": float(np.median(model_times)),
            "p50_ms": float(np.percentile(model_times, 50)),
            "p90_ms": float(np.percentile(model_times, 90)),
            "p95_ms": float(np.percentile(model_times, 95)),
            "p99_ms": float(np.percentile(model_times, 99)),
            "min_ms": float(np.min(model_times)),
            "max_ms": float(np.max(model_times)),
        },
        "ground_truth": {
            "mean_ms": float(np.mean(gt_times)),
            "std_ms": float(np.std(gt_times)),
            "median_ms": float(np.median(gt_times)),
            "p50_ms": float(np.percentile(gt_times, 50)),
            "p90_ms": float(np.percentile(gt_times, 90)),
            "p95_ms": float(np.percentile(gt_times, 95)),
            "p99_ms": float(np.percentile(gt_times, 99)),
        },
        "speedup": {
            "mean": float(np.mean(gt_times) / np.mean(model_times)),
            "median": float(np.median(gt_times) / np.median(model_times)),
            "p99": float(np.percentile(gt_times, 99) / np.percentile(model_times, 99)),
        },
    }


def evaluate_batch_throughput(
    model_dir: Path,
    batch_sizes: list[int] = [1, 8, 32, 128, 512, 1024, 2048],
    num_batches: int = 100,
    device: Optional[str] = None,
) -> dict:
    """Evaluate throughput at different batch sizes."""
    model, encoder, encoder_config, model_config, device = load_model(model_dir, device)

    graph_path = model_dir / "graph.gml"
    G, metadata = load_graph(str(graph_path))

    structural_computer = None
    if model_config.use_structural_features:
        structural_computer = compute_or_load_features(G, model_dir, show_progress=False)

    sampler = QuerySampler(G, seed=666)
    max_batch = max(batch_sizes)
    queries = [q for q, _ in sampler.generate_dataset(max_batch * num_batches)]

    results = {}

    for bs in batch_sizes:
        batch_times = []

        for i in range(num_batches):
            batch_queries = queries[i * bs:(i + 1) * bs]
            x_batch = encode_queries_batch(encoder, model, batch_queries, structural_computer, device)

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

        results[str(bs)] = {
            "batch_size": bs,
            "mean_time_ms": float(mean_time * 1000),
            "throughput_qps": float(throughput),
            "latency_per_query_us": float(mean_time * 1e6 / bs),
        }

    # Find peak throughput
    peak_bs = max(results.keys(), key=lambda k: results[k]["throughput_qps"])
    results["peak"] = {
        "batch_size": results[peak_bs]["batch_size"],
        "throughput_qps": results[peak_bs]["throughput_qps"],
    }

    return results


def find_failure_cases(
    model_dir: Path,
    num_samples: int = 5000,
    top_k: int = 100,
    device: Optional[str] = None,
    batch_size: int = 256,
) -> dict:
    """Find worst prediction cases for error analysis."""
    model, encoder, encoder_config, model_config, device = load_model(model_dir, device)

    graph_path = model_dir / "graph.gml"
    G, metadata = load_graph(str(graph_path))

    structural_computer = None
    if model_config.use_structural_features:
        structural_computer = compute_or_load_features(G, model_dir, show_progress=False)

    sampler = StratifiedQuerySampler(G, seed=555, node_types=metadata.node_types)
    test_data = sampler.generate_stratified_dataset(num_samples)

    cases = []

    with torch.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size), desc="Finding failures"):
            batch_data = test_data[i:i + batch_size]
            batch_queries = [q for q, _ in batch_data]
            batch_trues = [y for _, y in batch_data]

            x = encode_queries_batch(encoder, model, batch_queries, structural_computer, device)
            count_pred, dist_pred, _ = model(x)

            for j, (query, y_true) in enumerate(zip(batch_queries, batch_trues)):
                if query.query_type == QueryType.COUNT:
                    if model_config.use_log_transform:
                        pred = torch.expm1(count_pred[j]).clamp(min=0).item()
                    else:
                        pred = count_pred[j].item()
                    qerror = compute_qerror(np.array([pred]), np.array([y_true]))[0]
                else:
                    pred = dist_pred[j].item()
                    qerror = abs(pred - y_true)

                cases.append({
                    "query_type": query.query_type.value,
                    "node_type": query.start_node_type,
                    "degree_bin": query.start_degree_bin,
                    "radius": query.radius if query.query_type == QueryType.COUNT else None,
                    "target_type": query.target_type if query.query_type == QueryType.DISTANCE else None,
                    "prediction": float(pred),
                    "actual": float(y_true),
                    "error": float(abs(pred - y_true)),
                    "qerror": float(qerror),
                })

    # Sort by Q-error for count, by error for distance
    count_cases = [c for c in cases if c["query_type"] == "count"]
    dist_cases = [c for c in cases if c["query_type"] == "distance"]

    count_cases.sort(key=lambda x: x["qerror"], reverse=True)
    dist_cases.sort(key=lambda x: x["error"], reverse=True)

    return {
        "worst_count_cases": count_cases[:top_k],
        "worst_distance_cases": dist_cases[:top_k],
        "count_qerror_distribution": {
            "p50": float(np.percentile([c["qerror"] for c in count_cases], 50)) if count_cases else 0,
            "p90": float(np.percentile([c["qerror"] for c in count_cases], 90)) if count_cases else 0,
            "p99": float(np.percentile([c["qerror"] for c in count_cases], 99)) if count_cases else 0,
        },
    }


def run_full_evaluation(
    model_dir: Path,
    output_dir: Optional[Path] = None,
    device: Optional[str] = None,
) -> dict:
    """Run all evaluations and save results."""
    if output_dir is None:
        output_dir = model_dir

    print("=" * 70)
    print("NeurIPS Comprehensive Evaluation")
    print("=" * 70)

    results = {}

    # 1. Accuracy
    print("\n1. Evaluating accuracy...")
    results["accuracy"] = evaluate_accuracy_comprehensive(model_dir, device=device)

    # 2. Latency
    print("\n2. Evaluating latency...")
    results["latency"] = evaluate_latency_comprehensive(model_dir, device=device)

    # 3. Throughput
    print("\n3. Evaluating batch throughput...")
    results["throughput"] = evaluate_batch_throughput(model_dir, device=device)

    # 4. Uncertainty calibration
    print("\n4. Evaluating uncertainty calibration...")
    results["uncertainty"] = evaluate_uncertainty_calibration(model_dir, device=device)

    # 5. Failure analysis
    print("\n5. Analyzing failure cases...")
    results["failures"] = find_failure_cases(model_dir, device=device)

    # Save results
    output_path = output_dir / "neurips_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    if "count" in results["accuracy"]:
        c = results["accuracy"]["count"]
        print(f"\nCount Queries:")
        print(f"  Median Q-error:  {c['median_qerror']:.2f}")
        print(f"  Within 2x:       {c['within_2x']:.1f}%")
        print(f"  Within 5x:       {c['within_5x']:.1f}%")
        print(f"  95th %ile Q-err: {c['p95_qerror']:.2f}")

    if "distance" in results["accuracy"]:
        d = results["accuracy"]["distance"]
        print(f"\nDistance Queries:")
        print(f"  MAE:             {d['mae']:.2f}")
        print(f"  Within 1 hop:    {d['accuracy_1hop']:.1f}%")

    print(f"\nLatency:")
    print(f"  Model mean:      {results['latency']['model']['mean_ms']:.3f} ms")
    print(f"  Speedup:         {results['latency']['speedup']['mean']:.1f}x")

    print(f"\nThroughput:")
    print(f"  Peak:            {results['throughput']['peak']['throughput_qps']:,.0f} q/s")

    print(f"\n{'=' * 70}")
    print(f"Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="NeurIPS comprehensive evaluation")
    parser.add_argument("--model-dir", type=str, required=True, help="Model directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--accuracy-only", action="store_true", help="Only run accuracy evaluation")
    parser.add_argument("--latency-only", action="store_true", help="Only run latency evaluation")

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir

    if args.accuracy_only:
        results = evaluate_accuracy_comprehensive(model_dir, device=args.device)
        print(json.dumps(results, indent=2))
    elif args.latency_only:
        results = evaluate_latency_comprehensive(model_dir, device=args.device)
        print(json.dumps(results, indent=2))
    else:
        run_full_evaluation(model_dir, output_dir, device=args.device)


if __name__ == "__main__":
    main()
