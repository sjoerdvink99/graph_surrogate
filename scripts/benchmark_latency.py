"""
Comprehensive latency benchmark for comparing NN model inference vs graph queries.

This script provides detailed latency analysis to demonstrate the speedup advantage
of using a learned neural network model over direct graph database queries.

Supports benchmarking against:
- NetworkX (in-memory Python graph library)
- Kuzu (embedded graph database with Cypher queries)
"""

import argparse
import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from model.encoder import EncoderConfig, QueryEncoder
from model.network import GraphSurrogate
from training.graph_loader import load_graph
from training.query_sampler import Query, QuerySampler, QueryType, execute_query

try:
    import kuzu
    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False


@dataclass
class LatencyStats:
    """Statistics for latency measurements."""
    times_ms: np.ndarray

    @property
    def mean(self) -> float:
        return float(np.mean(self.times_ms))

    @property
    def std(self) -> float:
        return float(np.std(self.times_ms))

    @property
    def median(self) -> float:
        return float(np.median(self.times_ms))

    @property
    def p50(self) -> float:
        return float(np.percentile(self.times_ms, 50))

    @property
    def p90(self) -> float:
        return float(np.percentile(self.times_ms, 90))

    @property
    def p95(self) -> float:
        return float(np.percentile(self.times_ms, 95))

    @property
    def p99(self) -> float:
        return float(np.percentile(self.times_ms, 99))

    @property
    def min(self) -> float:
        return float(np.min(self.times_ms))

    @property
    def max(self) -> float:
        return float(np.max(self.times_ms))

    def ci_95(self) -> tuple[float, float]:
        """95% confidence interval using bootstrap."""
        n_bootstrap = 1000
        means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(self.times_ms, size=len(self.times_ms), replace=True)
            means.append(np.mean(sample))
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    def to_dict(self) -> dict:
        ci_low, ci_high = self.ci_95()
        return {
            "mean_ms": self.mean,
            "std_ms": self.std,
            "median_ms": self.median,
            "p50_ms": self.p50,
            "p90_ms": self.p90,
            "p95_ms": self.p95,
            "p99_ms": self.p99,
            "min_ms": self.min,
            "max_ms": self.max,
            "ci_95_low_ms": ci_low,
            "ci_95_high_ms": ci_high,
            "n_samples": len(self.times_ms),
        }


@dataclass
class LatencyBreakdown:
    """Breakdown of model inference latency into components."""
    encoding_times_ms: np.ndarray
    inference_times_ms: np.ndarray
    total_times_ms: np.ndarray

    @property
    def encoding_stats(self) -> LatencyStats:
        return LatencyStats(self.encoding_times_ms)

    @property
    def inference_stats(self) -> LatencyStats:
        return LatencyStats(self.inference_times_ms)

    @property
    def total_stats(self) -> LatencyStats:
        return LatencyStats(self.total_times_ms)

    def to_dict(self) -> dict:
        return {
            "encoding": self.encoding_stats.to_dict(),
            "inference": self.inference_stats.to_dict(),
            "total": self.total_stats.to_dict(),
        }


def load_model(model_dir: Path, device: Optional[str] = None) -> tuple:
    """Load trained model and encoder."""
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


def benchmark_model_detailed(
    model: torch.nn.Module,
    encoder: QueryEncoder,
    queries: list[Query],
    device: str,
    warmup_runs: int = 100,
) -> LatencyBreakdown:
    """
    Benchmark model inference with detailed timing breakdown.

    Returns separate timings for encoding and inference phases.
    """
    # Warmup
    for i in range(min(warmup_runs, len(queries))):
        x = encoder.encode(queries[i]).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    encoding_times = []
    inference_times = []
    total_times = []

    for query in queries:
        # Total time
        if device == "cuda":
            torch.cuda.synchronize()
        total_start = time.perf_counter()

        # Encoding time
        encode_start = time.perf_counter()
        x = encoder.encode(query).unsqueeze(0).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        encode_end = time.perf_counter()

        # Inference time
        infer_start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        infer_end = time.perf_counter()

        total_end = time.perf_counter()

        encoding_times.append((encode_end - encode_start) * 1000)
        inference_times.append((infer_end - infer_start) * 1000)
        total_times.append((total_end - total_start) * 1000)

    return LatencyBreakdown(
        encoding_times_ms=np.array(encoding_times),
        inference_times_ms=np.array(inference_times),
        total_times_ms=np.array(total_times),
    )


def benchmark_ground_truth(
    graph,
    queries: list[Query],
    warmup_runs: int = 100,
) -> LatencyStats:
    """Benchmark ground truth query execution on the graph (NetworkX)."""
    # Warmup
    for i in range(min(warmup_runs, len(queries))):
        execute_query(graph, queries[i])

    times = []
    for query in queries:
        start = time.perf_counter()
        execute_query(graph, query)
        times.append((time.perf_counter() - start) * 1000)

    return LatencyStats(np.array(times))


class KuzuBenchmark:
    """Benchmark using Kuzu embedded graph database with Cypher queries."""

    def __init__(self, graph, db_path: Optional[Path] = None):
        if not KUZU_AVAILABLE:
            raise ImportError("Kuzu is not installed. Run: pip install kuzu")

        self.temp_dir = None
        if db_path is None:
            # Create a temp directory and use a subdirectory for the database
            self.temp_dir = tempfile.mkdtemp(prefix="kuzu_benchmark_")
            db_path = Path(self.temp_dir) / "db"

        self.db_path = db_path
        self.db = kuzu.Database(str(db_path))
        self.conn = kuzu.Connection(self.db)

        self._load_graph(graph)

    def _load_graph(self, graph):
        """Load NetworkX graph into Kuzu."""
        # Create node table
        node_types = set()
        for node in graph.nodes():
            node_types.add(graph.nodes[node].get("node_type", "default"))

        # Create a single Node table with properties
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Node(
                id INT64,
                node_type STRING,
                degree_bin INT64,
                PRIMARY KEY(id)
            )
        """)

        # Create edge table
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS CONNECTED(
                FROM Node TO Node
            )
        """)

        # Insert nodes
        for node in graph.nodes():
            node_data = graph.nodes[node]
            node_type = node_data.get("node_type", "default")
            degree_bin = node_data.get("degree_bin", 0)
            # Handle both int and string node IDs
            node_id = int(node) if isinstance(node, (int, str)) and str(node).isdigit() else hash(node) % (2**31)
            self.conn.execute(
                "CREATE (n:Node {id: $id, node_type: $type, degree_bin: $deg})",
                {"id": node_id, "type": node_type, "deg": degree_bin}
            )

        # Create node ID mapping for edges
        self.node_to_id = {}
        for node in graph.nodes():
            node_id = int(node) if isinstance(node, (int, str)) and str(node).isdigit() else hash(node) % (2**31)
            self.node_to_id[node] = node_id

        # Insert edges
        for u, v in graph.edges():
            u_id = self.node_to_id[u]
            v_id = self.node_to_id[v]
            self.conn.execute(
                "MATCH (a:Node {id: $u}), (b:Node {id: $v}) CREATE (a)-[:CONNECTED]->(b)",
                {"u": u_id, "v": v_id}
            )
            # Add reverse edge for undirected graph
            self.conn.execute(
                "MATCH (a:Node {id: $u}), (b:Node {id: $v}) CREATE (a)-[:CONNECTED]->(b)",
                {"u": v_id, "v": u_id}
            )

    def execute_count_query(self, query: Query) -> int:
        """Execute a COUNT query using Cypher."""
        node_id = self.node_to_id.get(query.start_node_id, query.start_node_id)
        radius = query.radius

        # Build the path pattern for k-hop neighbors
        if query.attribute_filter.name is None:
            # Count all neighbors within k hops
            cypher = f"""
                MATCH (start:Node {{id: $start_id}})-[:CONNECTED*1..{radius}]-(neighbor:Node)
                WHERE neighbor.id <> $start_id
                RETURN COUNT(DISTINCT neighbor) as cnt
            """
            result = self.conn.execute(cypher, {"start_id": node_id})
        else:
            # Count neighbors with attribute filter
            attr_name = query.attribute_filter.name
            attr_value = query.attribute_filter.value
            cypher = f"""
                MATCH (start:Node {{id: $start_id}})-[:CONNECTED*1..{radius}]-(neighbor:Node)
                WHERE neighbor.id <> $start_id AND neighbor.{attr_name} = $attr_val
                RETURN COUNT(DISTINCT neighbor) as cnt
            """
            result = self.conn.execute(cypher, {"start_id": node_id, "attr_val": attr_value})

        row = result.get_next()
        return row[0] if row else 0

    def execute_distance_query(self, query: Query) -> int:
        """Execute a DISTANCE query using Cypher."""
        node_id = self.node_to_id.get(query.start_node_id, query.start_node_id)
        target_type = query.target_type
        max_hops = query.max_hops

        # Find shortest path to node of target type
        cypher = f"""
            MATCH path = shortestPath(
                (start:Node {{id: $start_id}})-[:CONNECTED*1..{max_hops}]-(target:Node {{node_type: $target_type}})
            )
            WHERE start.id <> target.id
            RETURN min(length(path)) as dist
        """
        result = self.conn.execute(cypher, {"start_id": node_id, "target_type": target_type})
        row = result.get_next()
        return row[0] if row and row[0] is not None else 7  # 7 = INF_DISTANCE

    def execute_query(self, query: Query) -> int:
        """Execute a query and return the result."""
        if query.query_type == QueryType.DISTANCE:
            return self.execute_distance_query(query)
        return self.execute_count_query(query)

    def benchmark(self, queries: list[Query], warmup_runs: int = 100) -> LatencyStats:
        """Benchmark query execution."""
        # Warmup
        for i in range(min(warmup_runs, len(queries))):
            try:
                self.execute_query(queries[i])
            except Exception:
                pass  # Some queries may fail, that's ok for warmup

        times = []
        for query in queries:
            start = time.perf_counter()
            try:
                self.execute_query(query)
            except Exception:
                pass  # Record time even if query fails
            times.append((time.perf_counter() - start) * 1000)

        return LatencyStats(np.array(times))

    def close(self):
        """Clean up resources."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def benchmark_batch_inference(
    model: torch.nn.Module,
    encoder: QueryEncoder,
    queries: list[Query],
    device: str,
    batch_sizes: list[int] = [1, 8, 32, 64, 128, 256, 512],
    num_batches: int = 100,
) -> dict:
    """Benchmark batch inference throughput."""
    results = {}

    for bs in batch_sizes:
        if bs * num_batches > len(queries):
            continue

        batch_times = []

        # Warmup
        for i in range(10):
            batch_queries = queries[i * bs:(i + 1) * bs]
            x_batch = torch.stack([encoder.encode(q) for q in batch_queries]).to(device)
            with torch.no_grad():
                _ = model(x_batch)

        if device == "cuda":
            torch.cuda.synchronize()

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

        results[bs] = {
            "batch_size": bs,
            "mean_batch_time_ms": float(mean_time * 1000),
            "throughput_qps": float(throughput),
            "latency_per_query_ms": float(mean_time * 1000 / bs),
            "latency_per_query_us": float(mean_time * 1_000_000 / bs),
        }

    return results


def benchmark_by_query_type(
    model: torch.nn.Module,
    encoder: QueryEncoder,
    graph,
    queries: list[Query],
    device: str,
) -> dict:
    """Benchmark latency separately for COUNT and DISTANCE queries."""
    count_queries = [q for q in queries if q.query_type == QueryType.COUNT]
    dist_queries = [q for q in queries if q.query_type == QueryType.DISTANCE]

    results = {}

    if count_queries:
        print(f"  Benchmarking {len(count_queries)} COUNT queries...")
        model_breakdown = benchmark_model_detailed(model, encoder, count_queries, device, warmup_runs=50)
        gt_stats = benchmark_ground_truth(graph, count_queries, warmup_runs=50)

        results["count"] = {
            "n_queries": len(count_queries),
            "model": model_breakdown.to_dict(),
            "ground_truth": gt_stats.to_dict(),
            "speedup_mean": gt_stats.mean / model_breakdown.total_stats.mean if model_breakdown.total_stats.mean > 0 else 0,
            "speedup_median": gt_stats.median / model_breakdown.total_stats.median if model_breakdown.total_stats.median > 0 else 0,
        }

    if dist_queries:
        print(f"  Benchmarking {len(dist_queries)} DISTANCE queries...")
        model_breakdown = benchmark_model_detailed(model, encoder, dist_queries, device, warmup_runs=50)
        gt_stats = benchmark_ground_truth(graph, dist_queries, warmup_runs=50)

        results["distance"] = {
            "n_queries": len(dist_queries),
            "model": model_breakdown.to_dict(),
            "ground_truth": gt_stats.to_dict(),
            "speedup_mean": gt_stats.mean / model_breakdown.total_stats.mean if model_breakdown.total_stats.mean > 0 else 0,
            "speedup_median": gt_stats.median / model_breakdown.total_stats.median if model_breakdown.total_stats.median > 0 else 0,
        }

    return results


def benchmark_by_complexity(
    model: torch.nn.Module,
    encoder: QueryEncoder,
    graph,
    queries: list[Query],
    device: str,
) -> dict:
    """Benchmark latency by query complexity (radius/max_hops)."""
    # Group COUNT queries by radius
    count_by_radius: dict[int, list[Query]] = {}
    for q in queries:
        if q.query_type == QueryType.COUNT:
            radius = q.radius
            if radius not in count_by_radius:
                count_by_radius[radius] = []
            count_by_radius[radius].append(q)

    # Group DISTANCE queries by max_hops
    dist_by_hops: dict[int, list[Query]] = {}
    for q in queries:
        if q.query_type == QueryType.DISTANCE:
            max_hops = q.max_hops
            if max_hops not in dist_by_hops:
                dist_by_hops[max_hops] = []
            dist_by_hops[max_hops].append(q)

    results = {
        "count_by_radius": {},
        "distance_by_max_hops": {},
    }

    for radius, qs in sorted(count_by_radius.items()):
        if len(qs) < 50:
            continue
        print(f"  COUNT queries with radius={radius}: {len(qs)} queries")
        model_breakdown = benchmark_model_detailed(model, encoder, qs[:500], device, warmup_runs=20)
        gt_stats = benchmark_ground_truth(graph, qs[:500], warmup_runs=20)

        results["count_by_radius"][radius] = {
            "n_queries": len(qs[:500]),
            "model_mean_ms": model_breakdown.total_stats.mean,
            "model_p95_ms": model_breakdown.total_stats.p95,
            "gt_mean_ms": gt_stats.mean,
            "gt_p95_ms": gt_stats.p95,
            "speedup": gt_stats.mean / model_breakdown.total_stats.mean if model_breakdown.total_stats.mean > 0 else 0,
        }

    for max_hops, qs in sorted(dist_by_hops.items()):
        if len(qs) < 50:
            continue
        print(f"  DISTANCE queries with max_hops={max_hops}: {len(qs)} queries")
        model_breakdown = benchmark_model_detailed(model, encoder, qs[:500], device, warmup_runs=20)
        gt_stats = benchmark_ground_truth(graph, qs[:500], warmup_runs=20)

        results["distance_by_max_hops"][max_hops] = {
            "n_queries": len(qs[:500]),
            "model_mean_ms": model_breakdown.total_stats.mean,
            "model_p95_ms": model_breakdown.total_stats.p95,
            "gt_mean_ms": gt_stats.mean,
            "gt_p95_ms": gt_stats.p95,
            "speedup": gt_stats.mean / model_breakdown.total_stats.mean if model_breakdown.total_stats.mean > 0 else 0,
        }

    return results


def run_comprehensive_benchmark(
    model_dir: Path,
    num_queries: int = 2000,
    device: Optional[str] = None,
    seed: int = 42,
    include_kuzu: bool = False,
) -> dict:
    """
    Run comprehensive latency benchmark comparing NN model vs graph queries.

    Returns detailed results including:
    - Overall latency comparison
    - Breakdown by encoding vs inference
    - Comparison by query type (COUNT vs DISTANCE)
    - Comparison by query complexity
    - Batch throughput analysis
    - Kuzu database comparison (optional)
    """
    print("=" * 70)
    print("Comprehensive Latency Benchmark")
    print("=" * 70)

    # Load model and graph
    print("\nLoading model and graph...")
    model, encoder, model_info, device = load_model(model_dir, device)

    graph_file = model_dir / "graph.gml"
    graph, graph_meta = load_graph(str(graph_file))

    print(f"  Model: {model_info.get('num_params', 0):,} parameters")
    print(f"  Graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
    print(f"  Device: {device}")

    # Generate queries
    print(f"\nGenerating {num_queries} test queries...")
    sampler = QuerySampler(graph, seed=seed)
    queries = [q for q, _ in sampler.generate_dataset(num_queries)]

    results = {
        "config": {
            "model_dir": str(model_dir),
            "num_queries": num_queries,
            "device": device,
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "model_params": model_info.get("num_params", 0),
        }
    }

    # 1. Overall latency comparison with breakdown
    print("\n1. Overall Latency Comparison")
    print("-" * 50)

    print("  Benchmarking model inference (with breakdown)...")
    model_breakdown = benchmark_model_detailed(model, encoder, queries, device)

    print("  Benchmarking ground truth queries...")
    gt_stats = benchmark_ground_truth(graph, queries)

    results["overall"] = {
        "model": model_breakdown.to_dict(),
        "ground_truth": gt_stats.to_dict(),
        "speedup_mean": gt_stats.mean / model_breakdown.total_stats.mean,
        "speedup_median": gt_stats.median / model_breakdown.total_stats.median,
        "speedup_p95": gt_stats.p95 / model_breakdown.total_stats.p95,
        "speedup_p99": gt_stats.p99 / model_breakdown.total_stats.p99,
    }

    print(f"\n  Model (total):     {model_breakdown.total_stats.mean:.3f} ms (mean)")
    print(f"    - Encoding:      {model_breakdown.encoding_stats.mean:.3f} ms")
    print(f"    - Inference:     {model_breakdown.inference_stats.mean:.3f} ms")
    print(f"  Ground Truth:      {gt_stats.mean:.3f} ms (mean)")
    print(f"  Speedup:           {results['overall']['speedup_mean']:.1f}x")

    # 2. By query type
    print("\n2. Latency by Query Type")
    print("-" * 50)
    results["by_query_type"] = benchmark_by_query_type(model, encoder, graph, queries, device)

    for qtype, stats in results["by_query_type"].items():
        print(f"\n  {qtype.upper()} ({stats['n_queries']} queries):")
        print(f"    Model:        {stats['model']['total']['mean_ms']:.3f} ms")
        print(f"    Ground Truth: {stats['ground_truth']['mean_ms']:.3f} ms")
        print(f"    Speedup:      {stats['speedup_mean']:.1f}x")

    # 3. By complexity
    print("\n3. Latency by Query Complexity")
    print("-" * 50)
    results["by_complexity"] = benchmark_by_complexity(model, encoder, graph, queries, device)

    if results["by_complexity"]["count_by_radius"]:
        print("\n  COUNT queries by radius:")
        for radius, stats in sorted(results["by_complexity"]["count_by_radius"].items()):
            print(f"    radius={radius}: model={stats['model_mean_ms']:.3f}ms, gt={stats['gt_mean_ms']:.3f}ms, speedup={stats['speedup']:.1f}x")

    if results["by_complexity"]["distance_by_max_hops"]:
        print("\n  DISTANCE queries by max_hops:")
        for hops, stats in sorted(results["by_complexity"]["distance_by_max_hops"].items()):
            print(f"    max_hops={hops}: model={stats['model_mean_ms']:.3f}ms, gt={stats['gt_mean_ms']:.3f}ms, speedup={stats['speedup']:.1f}x")

    # 4. Batch throughput
    print("\n4. Batch Throughput Analysis")
    print("-" * 50)

    # Generate more queries for batch testing
    batch_queries = [q for q, _ in sampler.generate_dataset(max(num_queries, 60000))]
    results["batch_throughput"] = benchmark_batch_inference(
        model, encoder, batch_queries, device,
        batch_sizes=[1, 8, 32, 64, 128, 256, 512],
        num_batches=100,
    )

    print(f"\n  {'Batch':>8} | {'Latency/query':>14} | {'Throughput':>14}")
    print(f"  {'-'*8}-+-{'-'*14}-+-{'-'*14}")
    for bs, stats in sorted(results["batch_throughput"].items()):
        print(f"  {bs:>8} | {stats['latency_per_query_ms']:>11.4f} ms | {stats['throughput_qps']:>11,.0f} q/s")

    # 5. Summary statistics
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    model_mean = results["overall"]["model"]["total"]["mean_ms"]
    gt_mean = results["overall"]["ground_truth"]["mean_ms"]
    speedup = results["overall"]["speedup_mean"]

    print(f"\n  Neural Network Model:")
    print(f"    Mean latency:     {model_mean:.3f} ms")
    print(f"    P99 latency:      {results['overall']['model']['total']['p99_ms']:.3f} ms")
    print(f"    Throughput:       {1000/model_mean:,.0f} queries/sec (single)")

    max_batch = max(results["batch_throughput"].keys())
    max_throughput = results["batch_throughput"][max_batch]["throughput_qps"]
    print(f"    Max throughput:   {max_throughput:,.0f} queries/sec (batch={max_batch})")

    print(f"\n  Ground Truth (NetworkX):")
    print(f"    Mean latency:     {gt_mean:.3f} ms")
    print(f"    P99 latency:      {results['overall']['ground_truth']['p99_ms']:.3f} ms")
    print(f"    Throughput:       {1000/gt_mean:,.0f} queries/sec")

    print(f"\n  Speedup: {speedup:.1f}x faster with NN model")

    # Store raw timing data for visualization
    results["raw_times"] = {
        "model_total_ms": model_breakdown.total_times_ms.tolist(),
        "model_encoding_ms": model_breakdown.encoding_times_ms.tolist(),
        "model_inference_ms": model_breakdown.inference_times_ms.tolist(),
        "ground_truth_ms": gt_stats.times_ms.tolist(),
    }

    # 5. Kuzu database comparison (optional)
    if include_kuzu:
        if not KUZU_AVAILABLE:
            print("\n5. Kuzu Database Benchmark")
            print("-" * 50)
            print("  Kuzu not available. Install with: pip install kuzu")
        else:
            print("\n5. Kuzu Database Benchmark")
            print("-" * 50)
            print("  Loading graph into Kuzu...")

            try:
                with KuzuBenchmark(graph) as kuzu_bench:
                    print(f"  Benchmarking Kuzu queries ({len(queries)} queries)...")
                    kuzu_stats = kuzu_bench.benchmark(queries, warmup_runs=50)

                    results["kuzu"] = kuzu_stats.to_dict()
                    results["kuzu"]["speedup_vs_model"] = kuzu_stats.mean / model_breakdown.total_stats.mean if model_breakdown.total_stats.mean > 0 else 0
                    results["kuzu"]["speedup_vs_networkx"] = kuzu_stats.mean / gt_stats.mean if gt_stats.mean > 0 else 0

                    print(f"\n  Kuzu Database:")
                    print(f"    Mean latency:     {kuzu_stats.mean:.3f} ms")
                    print(f"    P99 latency:      {kuzu_stats.p99:.3f} ms")
                    print(f"    Throughput:       {1000/kuzu_stats.mean:,.0f} queries/sec")

                    print(f"\n  Comparison:")
                    print(f"    NN Model vs Kuzu:     {results['kuzu']['speedup_vs_model']:.1f}x faster")
                    print(f"    NetworkX vs Kuzu:     {results['kuzu']['speedup_vs_networkx']:.1f}x")

                    # Store raw kuzu times
                    results["raw_times"]["kuzu_ms"] = kuzu_stats.times_ms.tolist()

            except Exception as e:
                print(f"  Kuzu benchmark failed: {e}")
                results["kuzu"] = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive latency benchmark for GraphSurrogate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=str, default="output", help="Model directory")
    parser.add_argument("--num-queries", type=int, default=2000, help="Number of queries to benchmark")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--kuzu", action="store_true", help="Include Kuzu database benchmark")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    results = run_comprehensive_benchmark(
        model_dir=model_dir,
        num_queries=args.num_queries,
        device=args.device,
        seed=args.seed,
        include_kuzu=args.kuzu,
    )

    # Save results
    output_path = Path(args.output) if args.output else model_dir / "latency_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
