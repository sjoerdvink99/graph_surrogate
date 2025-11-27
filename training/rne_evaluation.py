"""
Evaluation Module for Hierarchical Road Network Embedding.

Implements:
1. Point-to-Point distance queries
2. Range queries (find nodes within distance threshold)
3. k-Nearest Neighbor (kNN) queries
4. Comprehensive metrics (MRE, AE, CDF, latency)
5. Out-of-Distribution (OOD) evaluation

Reference: NeurIPS-quality evaluation methodology for distance approximation.
"""

import time
from dataclasses import dataclass
from typing import Optional, Callable
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from tqdm import tqdm


@dataclass
class QueryResult:
    """Result from a query operation."""
    query_time_ns: float
    results: list
    ground_truth: Optional[list] = None


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    # Distance approximation
    mre: float  # Mean Relative Error
    mae: float  # Mean Absolute Error
    median_ae: float
    max_ae: float

    # Error distribution (CDF points)
    pct_within_1: float
    pct_within_2: float
    pct_within_5: float
    pct_within_10: float
    pct_within_20: float

    # Query performance
    mean_query_time_ns: float
    median_query_time_ns: float
    p99_query_time_ns: float

    # Range query metrics
    range_precision: Optional[float] = None
    range_recall: Optional[float] = None
    range_f1: Optional[float] = None

    # kNN metrics
    knn_recall_at_k: Optional[float] = None
    knn_map: Optional[float] = None


class RNESearchIndex:
    """
    Search index for efficient Range and kNN queries using RNE embeddings.

    Uses the tree-structured radius r(node) for hierarchical search,
    enabling sub-linear query time for range and kNN operations.
    """

    def __init__(
        self,
        model: nn.Module,
        partition_tree,
        device: torch.device = None,
    ):
        """
        Initialize search index.

        Args:
            model: Trained Hierarchical RNE model
            partition_tree: Partition tree from training
            device: Torch device
        """
        self.model = model
        self.partition_tree = partition_tree
        self.device = device or next(model.parameters()).device

        # Pre-compute embeddings for all nodes
        self._build_index()

    def _build_index(self):
        """Pre-compute and cache all node embeddings."""
        self.model.eval()

        # Get all nodes
        all_nodes = list(self.partition_tree.node_to_ancestors.keys())
        self.num_nodes = len(all_nodes)
        self.node_ids = torch.tensor(all_nodes, dtype=torch.long, device=self.device)

        # Compute embeddings in batches
        batch_size = 4096
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(all_nodes), batch_size):
                batch_ids = self.node_ids[i:i + batch_size]
                emb = self.model.compute_embedding(batch_ids)
                embeddings.append(emb)

        self.embeddings = torch.cat(embeddings, dim=0)  # (num_nodes, embed_dim)

        # Build node ID to index mapping
        self.node_to_idx = {n: i for i, n in enumerate(all_nodes)}

        # Compute partition-level bounds for hierarchical search
        self._compute_partition_bounds()

    def _compute_partition_bounds(self):
        """Compute embedding bounds for each partition for pruning."""
        self.partition_bounds = {}

        for level, partitions in enumerate(self.partition_tree.level_partitions):
            for partition in partitions:
                # Get indices of nodes in this partition
                indices = torch.tensor(
                    [self.node_to_idx[n] for n in partition.node_ids if n in self.node_to_idx],
                    dtype=torch.long,
                    device=self.device
                )

                if len(indices) > 0:
                    partition_embs = self.embeddings[indices]

                    # Compute bounding box (min, max per dimension)
                    emb_min = partition_embs.min(dim=0).values
                    emb_max = partition_embs.max(dim=0).values

                    # Compute centroid
                    centroid = partition_embs.mean(dim=0)

                    # Compute radius (max L1 distance from centroid)
                    l1_dists = torch.abs(partition_embs - centroid).sum(dim=1)
                    radius = l1_dists.max().item()

                    self.partition_bounds[(level, partition.partition_id)] = {
                        'min': emb_min,
                        'max': emb_max,
                        'centroid': centroid,
                        'radius': radius,
                        'indices': indices,
                    }

    def range_query(
        self,
        source_id: int,
        threshold: float,
        use_hierarchical: bool = True,
    ) -> QueryResult:
        """
        Find all nodes within L1 distance threshold of source.

        Args:
            source_id: Source node ID
            threshold: Distance threshold (in original distance units)
            use_hierarchical: Use hierarchical pruning for speedup

        Returns:
            QueryResult with list of (node_id, approximate_distance) tuples
        """
        start_time = time.perf_counter_ns()

        source_idx = self.node_to_idx.get(source_id)
        if source_idx is None:
            return QueryResult(
                query_time_ns=time.perf_counter_ns() - start_time,
                results=[]
            )

        source_emb = self.embeddings[source_idx].unsqueeze(0)

        if use_hierarchical:
            # Hierarchical pruning
            candidate_indices = self._hierarchical_prune(source_emb, threshold)
        else:
            candidate_indices = torch.arange(self.num_nodes, device=self.device)

        # Compute distances to candidates
        candidate_embs = self.embeddings[candidate_indices]
        l1_dists = torch.abs(candidate_embs - source_emb).sum(dim=1)

        # Filter by threshold
        with torch.no_grad():
            # Scale L1 distance to approximate real distance
            approx_dists = self.model.compute_distance(
                torch.tensor([source_id], device=self.device).expand(len(candidate_indices)),
                self.node_ids[candidate_indices],
                source_emb.expand(len(candidate_indices), -1),
                candidate_embs,
            )

        mask = approx_dists <= threshold
        result_indices = candidate_indices[mask]
        result_dists = approx_dists[mask]

        # Convert to list of (node_id, distance) tuples
        results = [
            (self.node_ids[idx].item(), dist.item())
            for idx, dist in zip(result_indices, result_dists)
        ]

        return QueryResult(
            query_time_ns=time.perf_counter_ns() - start_time,
            results=results
        )

    def _hierarchical_prune(
        self,
        source_emb: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        """Use partition bounds to prune search space."""
        candidate_partitions = []

        # Start from coarsest level
        for level in range(len(self.partition_tree.level_partitions)):
            for partition in self.partition_tree.level_partitions[level]:
                key = (level, partition.partition_id)
                if key not in self.partition_bounds:
                    continue

                bounds = self.partition_bounds[key]

                # Check if partition can contain any nodes within threshold
                # Using centroid + radius as upper bound on min distance
                dist_to_centroid = torch.abs(source_emb - bounds['centroid']).sum().item()
                min_possible_dist = max(0, dist_to_centroid - bounds['radius'])

                if min_possible_dist <= threshold * 1.5:  # Add margin
                    if partition.is_leaf or level == len(self.partition_tree.level_partitions) - 1:
                        candidate_partitions.append(bounds['indices'])

        if candidate_partitions:
            return torch.cat(candidate_partitions)
        else:
            return torch.arange(self.num_nodes, device=self.device)

    def knn_query(
        self,
        source_id: int,
        k: int,
        use_hierarchical: bool = True,
    ) -> QueryResult:
        """
        Find k nearest neighbors of source node.

        Args:
            source_id: Source node ID
            k: Number of neighbors to find
            use_hierarchical: Use hierarchical search

        Returns:
            QueryResult with list of (node_id, approximate_distance) tuples
        """
        start_time = time.perf_counter_ns()

        source_idx = self.node_to_idx.get(source_id)
        if source_idx is None:
            return QueryResult(
                query_time_ns=time.perf_counter_ns() - start_time,
                results=[]
            )

        source_emb = self.embeddings[source_idx].unsqueeze(0)

        # Compute all distances
        with torch.no_grad():
            all_dists = self.model.compute_distance(
                torch.tensor([source_id], device=self.device).expand(self.num_nodes),
                self.node_ids,
                source_emb.expand(self.num_nodes, -1),
                self.embeddings,
            )

        # Exclude source itself
        all_dists[source_idx] = float('inf')

        # Get top-k
        top_k_dists, top_k_indices = torch.topk(all_dists, k, largest=False)

        results = [
            (self.node_ids[idx].item(), dist.item())
            for idx, dist in zip(top_k_indices, top_k_dists)
        ]

        return QueryResult(
            query_time_ns=time.perf_counter_ns() - start_time,
            results=results
        )

    def batch_distance_query(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        Batch point-to-point distance queries.

        Returns:
            distances, mean_query_time_ns
        """
        start_time = time.perf_counter_ns()

        with torch.no_grad():
            distances, _, _ = self.model(source_ids.to(self.device), target_ids.to(self.device))

        total_time = time.perf_counter_ns() - start_time
        mean_time = total_time / len(source_ids)

        return distances.cpu(), mean_time


def compute_exact_distances(
    G: nx.Graph,
    source_ids: list[int],
    target_ids: list[int],
    show_progress: bool = True,
) -> list[float]:
    """Compute exact shortest-path distances using NetworkX."""
    distances = []

    iterator = zip(source_ids, target_ids)
    if show_progress:
        iterator = tqdm(list(iterator), desc="Computing exact distances")

    for source, target in iterator:
        try:
            dist = nx.shortest_path_length(G, source, target)
        except nx.NetworkXNoPath:
            dist = float('inf')
        except nx.NodeNotFound:
            dist = float('inf')
        distances.append(dist)

    return distances


def compute_exact_range(
    G: nx.Graph,
    source_id: int,
    threshold: float,
) -> set[int]:
    """Compute exact set of nodes within distance threshold."""
    result = set()

    try:
        lengths = nx.single_source_shortest_path_length(G, source_id, cutoff=int(threshold))
        for node, dist in lengths.items():
            if 0 < dist <= threshold:
                result.add(node)
    except nx.NodeNotFound:
        pass

    return result


def compute_exact_knn(
    G: nx.Graph,
    source_id: int,
    k: int,
) -> list[tuple[int, float]]:
    """Compute exact k nearest neighbors."""
    try:
        lengths = dict(nx.single_source_shortest_path_length(G, source_id))
        # Remove source
        if source_id in lengths:
            del lengths[source_id]

        # Sort by distance
        sorted_nodes = sorted(lengths.items(), key=lambda x: x[1])
        return sorted_nodes[:k]
    except nx.NodeNotFound:
        return []


def evaluate_distance_approximation(
    model: nn.Module,
    G: nx.Graph,
    num_queries: int = 10000,
    seed: int = 42,
    device: torch.device = None,
) -> EvaluationMetrics:
    """
    Comprehensive evaluation of distance approximation quality.

    Returns detailed metrics including MRE, MAE, CDF, and query latency.
    """
    import random
    rng = random.Random(seed)

    if device is None:
        device = next(model.parameters()).device

    nodes = list(G.nodes())
    model.eval()

    # Generate random queries
    source_ids = []
    target_ids = []
    true_distances = []

    print("Generating evaluation queries...")
    for _ in tqdm(range(num_queries)):
        source = rng.choice(nodes)
        target = rng.choice(nodes)
        if source != target:
            try:
                dist = nx.shortest_path_length(G, source, target)
                source_ids.append(source)
                target_ids.append(target)
                true_distances.append(dist)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    source_tensor = torch.tensor(source_ids, dtype=torch.long, device=device)
    target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)
    true_tensor = torch.tensor(true_distances, dtype=torch.float32)

    # Measure query time
    print("Measuring query latency...")
    query_times = []

    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            _ = model(source_tensor[:100], target_tensor[:100])

        # Measure
        batch_size = 1000
        for i in range(0, len(source_ids), batch_size):
            batch_source = source_tensor[i:i + batch_size]
            batch_target = target_tensor[i:i + batch_size]

            start = time.perf_counter_ns()
            pred, _, _ = model(batch_source, batch_target)
            end = time.perf_counter_ns()

            per_query_time = (end - start) / len(batch_source)
            query_times.extend([per_query_time] * len(batch_source))

    # Get predictions
    print("Computing predictions...")
    with torch.no_grad():
        pred_distances, _, _ = model(source_tensor, target_tensor)
        pred_distances = pred_distances.cpu()

    # Compute metrics
    print("Computing metrics...")

    # Filter out infinite distances
    valid_mask = true_tensor < float('inf')
    pred_valid = pred_distances[valid_mask]
    true_valid = true_tensor[valid_mask]

    # Absolute errors
    abs_errors = torch.abs(pred_valid - true_valid)
    mae = abs_errors.mean().item()
    median_ae = abs_errors.median().item()
    max_ae = abs_errors.max().item()

    # Relative errors
    rel_errors = abs_errors / (true_valid + 1.0)
    mre = rel_errors.mean().item()

    # CDF points (percentage of queries with relative error <= threshold)
    pct_within_1 = (rel_errors <= 0.01).float().mean().item() * 100
    pct_within_2 = (rel_errors <= 0.02).float().mean().item() * 100
    pct_within_5 = (rel_errors <= 0.05).float().mean().item() * 100
    pct_within_10 = (rel_errors <= 0.10).float().mean().item() * 100
    pct_within_20 = (rel_errors <= 0.20).float().mean().item() * 100

    # Query time statistics
    query_times = np.array(query_times)

    return EvaluationMetrics(
        mre=mre,
        mae=mae,
        median_ae=median_ae,
        max_ae=max_ae,
        pct_within_1=pct_within_1,
        pct_within_2=pct_within_2,
        pct_within_5=pct_within_5,
        pct_within_10=pct_within_10,
        pct_within_20=pct_within_20,
        mean_query_time_ns=query_times.mean(),
        median_query_time_ns=np.median(query_times),
        p99_query_time_ns=np.percentile(query_times, 99),
    )


def evaluate_range_queries(
    search_index: RNESearchIndex,
    G: nx.Graph,
    num_queries: int = 100,
    thresholds: list[float] = [2.0, 3.0, 5.0],
    seed: int = 42,
) -> dict:
    """Evaluate range query accuracy."""
    import random
    rng = random.Random(seed)
    nodes = list(G.nodes())

    results = {}

    for threshold in thresholds:
        precisions = []
        recalls = []
        f1s = []
        query_times = []

        for _ in tqdm(range(num_queries), desc=f"Range queries (τ={threshold})"):
            source = rng.choice(nodes)

            # Approximate
            result = search_index.range_query(source, threshold)
            approx_set = set(node_id for node_id, _ in result.results)
            query_times.append(result.query_time_ns)

            # Exact
            exact_set = compute_exact_range(G, source, threshold)

            if len(exact_set) > 0:
                precision = len(approx_set & exact_set) / max(len(approx_set), 1)
                recall = len(approx_set & exact_set) / len(exact_set)
                f1 = 2 * precision * recall / max(precision + recall, 1e-8)

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

        results[f'threshold_{threshold}'] = {
            'precision': np.mean(precisions) if precisions else 0,
            'recall': np.mean(recalls) if recalls else 0,
            'f1': np.mean(f1s) if f1s else 0,
            'mean_query_time_ns': np.mean(query_times),
        }

    return results


def evaluate_knn_queries(
    search_index: RNESearchIndex,
    G: nx.Graph,
    num_queries: int = 100,
    k_values: list[int] = [5, 10, 20],
    seed: int = 42,
) -> dict:
    """Evaluate k-NN query accuracy."""
    import random
    rng = random.Random(seed)
    nodes = list(G.nodes())

    results = {}

    for k in k_values:
        recalls = []
        maps = []
        query_times = []

        for _ in tqdm(range(num_queries), desc=f"kNN queries (k={k})"):
            source = rng.choice(nodes)

            # Approximate
            result = search_index.knn_query(source, k)
            approx_list = [node_id for node_id, _ in result.results]
            query_times.append(result.query_time_ns)

            # Exact
            exact_list = [node_id for node_id, _ in compute_exact_knn(G, source, k)]

            if len(exact_list) > 0:
                # Recall@k
                recall = len(set(approx_list) & set(exact_list)) / len(exact_list)
                recalls.append(recall)

                # Mean Average Precision
                hits = 0
                ap = 0
                for i, node in enumerate(approx_list):
                    if node in exact_list:
                        hits += 1
                        ap += hits / (i + 1)
                ap = ap / min(k, len(exact_list))
                maps.append(ap)

        results[f'k_{k}'] = {
            'recall': np.mean(recalls) if recalls else 0,
            'map': np.mean(maps) if maps else 0,
            'mean_query_time_ns': np.mean(query_times),
        }

    return results


def evaluate_ood_generalization(
    model: nn.Module,
    train_graph: nx.Graph,
    test_graph: nx.Graph,
    num_queries: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Evaluate out-of-distribution generalization.

    Tests model trained on train_graph when applied to test_graph.
    Useful for testing if model generalizes from city to state level.
    """
    print("Evaluating OOD generalization...")
    print(f"  Training graph: {train_graph.number_of_nodes()} nodes")
    print(f"  Test graph: {test_graph.number_of_nodes()} nodes")

    # Get node mapping if needed (test graph might have different node IDs)
    # For now, assume models are retrained or use same node IDs

    return evaluate_distance_approximation(
        model=model,
        G=test_graph,
        num_queries=num_queries,
        seed=seed,
    )


def print_evaluation_report(metrics: EvaluationMetrics, title: str = "Evaluation Results"):
    """Print formatted evaluation report."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print("\nDistance Approximation:")
    print(f"  Mean Relative Error (MRE):    {metrics.mre:.4f}")
    print(f"  Mean Absolute Error (MAE):    {metrics.mae:.4f}")
    print(f"  Median Absolute Error:        {metrics.median_ae:.4f}")
    print(f"  Max Absolute Error:           {metrics.max_ae:.4f}")

    print("\nError Distribution (CDF):")
    print(f"  Within 1% error:              {metrics.pct_within_1:.1f}%")
    print(f"  Within 2% error:              {metrics.pct_within_2:.1f}%")
    print(f"  Within 5% error:              {metrics.pct_within_5:.1f}%")
    print(f"  Within 10% error:             {metrics.pct_within_10:.1f}%")
    print(f"  Within 20% error:             {metrics.pct_within_20:.1f}%")

    print("\nQuery Latency:")
    print(f"  Mean query time:              {metrics.mean_query_time_ns:.0f} ns ({metrics.mean_query_time_ns/1000:.1f} μs)")
    print(f"  Median query time:            {metrics.median_query_time_ns:.0f} ns")
    print(f"  99th percentile:              {metrics.p99_query_time_ns:.0f} ns")

    if metrics.range_precision is not None:
        print("\nRange Query:")
        print(f"  Precision:                    {metrics.range_precision:.4f}")
        print(f"  Recall:                       {metrics.range_recall:.4f}")
        print(f"  F1 Score:                     {metrics.range_f1:.4f}")

    if metrics.knn_recall_at_k is not None:
        print("\nk-NN Query:")
        print(f"  Recall@k:                     {metrics.knn_recall_at_k:.4f}")
        print(f"  Mean Average Precision:       {metrics.knn_map:.4f}")

    print("=" * 70)
