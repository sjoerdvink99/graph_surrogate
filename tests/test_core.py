"""
Comprehensive unit tests for GraphSurrogate core components.

These tests ensure correctness of:
1. Query encoding
2. Model architecture
3. Loss computation
4. Baseline implementations
5. Query execution
6. Failure analysis (Gap 7)
7. Uncertainty estimation (Gap 9)
8. Cardinality baselines (Gap 5)
9. Query optimizer integration (Gap 6)
10. Cross-graph generalization (Gap 4)
11. Incremental updates (Gap 10)

Run with: pytest tests/test_core.py -v
"""

import random
from pathlib import Path
from tempfile import TemporaryDirectory

import networkx as nx
import numpy as np
import pytest
import torch

from model.encoder import EncoderConfig, QueryEncoder
from model.network import GraphSurrogate, TwoHeadLoss
from model.baselines import (
    MeanBaseline,
    HistogramBaseline,
    SamplingBaseline,
    RandomWalkBaseline,
    evaluate_baseline,
)
from model.cardinality_baselines import (
    EnhancedHistogramEstimator,
    NeuroCardInspired,
    DeepDBInspired,
    evaluate_cardinality_baselines,
)
from training.query_sampler import (
    Query,
    QueryType,
    AttributeFilter,
    QuerySampler,
    execute_query,
    get_k_hop_neighbors,
    count_neighborhood,
    compute_shortest_distance,
    INF_DISTANCE,
)


# ==================== Fixtures ====================

@pytest.fixture
def sample_graph():
    """Create a small test graph with node attributes."""
    G = nx.Graph()

    # Add nodes with attributes
    for i in range(20):
        G.add_node(i, node_type="A" if i < 10 else "B", degree_bin=i % 5)

    # Add edges to create structure
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Chain
        (0, 5), (0, 6), (0, 7),           # Star from 0
        (10, 11), (11, 12), (12, 13),     # Chain in B
        (5, 10), (6, 11),                  # Cross-type edges
        (15, 16), (16, 17), (17, 18), (18, 19),  # Another chain
    ]
    G.add_edges_from(edges)
    return G


@pytest.fixture
def encoder_config():
    """Create a test encoder config."""
    return EncoderConfig(
        node_types=["A", "B"],
        attribute_names=["node_type"],
        attribute_values={"node_type": ["A", "B"]},
        degree_bins=[0, 1, 2, 3, 4],
        radii=[1, 2, 3],
        max_hops_options=[3, 4, 5, 6],
    )


@pytest.fixture
def query_encoder(encoder_config):
    """Create a query encoder."""
    return QueryEncoder(encoder_config)


@pytest.fixture
def sample_model(encoder_config):
    """Create a sample model for testing."""
    return GraphSurrogate(
        input_dim=encoder_config.input_dim,
        hidden_dim=64,
        latent_dim=16,
        num_layers=2,
        dropout=0.0,  # No dropout for deterministic tests
    )


# ==================== Query Sampler Tests ====================

class TestQuerySampler:
    """Tests for query generation and execution."""

    def test_get_k_hop_neighbors_k0(self, sample_graph):
        """k=0 should return empty set."""
        neighbors = get_k_hop_neighbors(sample_graph, 0, k=0)
        assert neighbors == set()

    def test_get_k_hop_neighbors_k1(self, sample_graph):
        """k=1 should return direct neighbors."""
        neighbors = get_k_hop_neighbors(sample_graph, 0, k=1)
        expected = {1, 5, 6, 7}
        assert neighbors == expected

    def test_get_k_hop_neighbors_k2(self, sample_graph):
        """k=2 should return 2-hop neighbors."""
        neighbors = get_k_hop_neighbors(sample_graph, 0, k=2)
        # Node 0's 1-hop: {1, 5, 6, 7}
        # Node 0's 2-hop adds: {2, 10, 11}
        assert 2 in neighbors
        assert 10 in neighbors
        assert 0 not in neighbors  # Shouldn't include start node

    def test_count_neighborhood_no_filter(self, sample_graph):
        """Count without filter returns total neighbors."""
        count = count_neighborhood(sample_graph, 0, radius=1, attr_filter=AttributeFilter())
        assert count == 4  # {1, 5, 6, 7}

    def test_count_neighborhood_with_filter(self, sample_graph):
        """Count with filter returns matching neighbors."""
        attr_filter = AttributeFilter(name="node_type", value="A")
        count = count_neighborhood(sample_graph, 0, radius=1, attr_filter=attr_filter)
        # Neighbors are {1, 5, 6, 7}, all type A (nodes < 10)
        assert count == 4

    def test_compute_shortest_distance_exists(self, sample_graph):
        """Find shortest distance when target type exists."""
        # From node 0 to type B (nodes >= 10)
        dist = compute_shortest_distance(sample_graph, 0, "B", max_hops=6)
        # 0 -> 5 -> 10 (shortest path to type B)
        assert dist == 2

    def test_compute_shortest_distance_not_found(self, sample_graph):
        """Return INF when target type not reachable."""
        # Create isolated node of type C
        sample_graph.add_node(100, node_type="C")
        dist = compute_shortest_distance(sample_graph, 0, "C", max_hops=6)
        assert dist == INF_DISTANCE

    def test_execute_query_count(self, sample_graph):
        """Execute count query returns correct count."""
        query = Query(
            query_type=QueryType.COUNT,
            start_node_id=0,
            start_node_type="A",
            radius=1,
            attribute_filter=AttributeFilter(),
        )
        result = execute_query(sample_graph, query)
        assert result == 4

    def test_execute_query_distance(self, sample_graph):
        """Execute distance query returns correct distance."""
        query = Query(
            query_type=QueryType.DISTANCE,
            start_node_id=0,
            start_node_type="A",
            target_type="B",
            max_hops=6,
        )
        result = execute_query(sample_graph, query)
        assert result == 2

    def test_query_sampler_determinism(self, sample_graph):
        """Query sampler should be deterministic with same seed."""
        import random

        # Reset global random state before each sampler
        random.seed(42)
        sampler1 = QuerySampler(sample_graph, seed=42, node_types=["A", "B"])
        queries1 = [sampler1.sample_query() for _ in range(10)]

        random.seed(42)
        sampler2 = QuerySampler(sample_graph, seed=42, node_types=["A", "B"])
        queries2 = [sampler2.sample_query() for _ in range(10)]

        for q1, q2 in zip(queries1, queries2):
            assert q1.query_type == q2.query_type
            assert q1.start_node_id == q2.start_node_id
            assert q1.radius == q2.radius

    def test_query_sampler_different_seeds(self, sample_graph):
        """Different seeds should produce different queries."""
        sampler1 = QuerySampler(sample_graph, seed=42, node_types=["A", "B"])
        sampler2 = QuerySampler(sample_graph, seed=123, node_types=["A", "B"])

        queries1 = [sampler1.sample_query() for _ in range(100)]
        queries2 = [sampler2.sample_query() for _ in range(100)]

        # Not all queries should be the same
        same_count = sum(
            1 for q1, q2 in zip(queries1, queries2)
            if q1.start_node_id == q2.start_node_id
        )
        assert same_count < 90  # Allow some coincidence


# ==================== Encoder Tests ====================

class TestQueryEncoder:
    """Tests for query encoding."""

    def test_encoder_output_dimension(self, query_encoder, encoder_config):
        """Encoded query should have correct dimension."""
        query = Query(
            query_type=QueryType.COUNT,
            start_node_id=0,
            start_node_type="A",
            radius=1,
        )
        encoded = query_encoder.encode(query)
        assert encoded.shape == (encoder_config.input_dim,)

    def test_encoder_one_hot_validity(self, query_encoder):
        """One-hot encodings should sum to 1."""
        query = Query(
            query_type=QueryType.COUNT,
            start_node_id=0,
            start_node_type="A",
            start_degree_bin=2,
            radius=2,
        )
        encoded = query_encoder.encode(query)

        # Check node type one-hot (first 2 elements)
        assert encoded[:2].sum() == 1.0

        # Check degree bin one-hot (next 5 elements)
        assert encoded[2:7].sum() == 1.0

        # Check query type one-hot (last 2 elements)
        assert encoded[-2:].sum() == 1.0

    def test_encoder_batch(self, query_encoder, encoder_config):
        """Batch encoding should work correctly."""
        queries = [
            Query(QueryType.COUNT, 0, "A", radius=1),
            Query(QueryType.DISTANCE, 1, "B", target_type="A", max_hops=4),
        ]
        encoded = query_encoder.encode_batch(queries)
        assert encoded.shape == (2, encoder_config.input_dim)

    def test_encoder_different_query_types(self, query_encoder):
        """Different query types should produce different encodings."""
        count_query = Query(QueryType.COUNT, 0, "A", radius=1)
        dist_query = Query(QueryType.DISTANCE, 0, "A", target_type="B", max_hops=4)

        enc_count = query_encoder.encode(count_query)
        enc_dist = query_encoder.encode(dist_query)

        # Query type encoding should be different
        assert not torch.allclose(enc_count[-2:], enc_dist[-2:])


# ==================== Model Tests ====================

class TestGraphSurrogate:
    """Tests for the main model architecture."""

    def test_model_forward_shape(self, sample_model, encoder_config):
        """Forward pass should return correct shapes."""
        batch_size = 8
        x = torch.randn(batch_size, encoder_config.input_dim)

        count_pred, dist_pred, latent = sample_model(x)

        assert count_pred.shape == (batch_size,)
        assert dist_pred.shape == (batch_size,)
        assert latent.shape == (batch_size, 16)  # latent_dim=16

    def test_model_output_non_negative(self, sample_model, encoder_config):
        """Model outputs should be non-negative (softplus activation)."""
        x = torch.randn(100, encoder_config.input_dim)
        count_pred, dist_pred, _ = sample_model(x)

        assert (count_pred >= 0).all()
        assert (dist_pred >= 0).all()

    def test_model_deterministic(self, sample_model, encoder_config):
        """Model should be deterministic in eval mode."""
        sample_model.eval()
        x = torch.randn(5, encoder_config.input_dim)

        with torch.no_grad():
            out1 = sample_model(x)
            out2 = sample_model(x)

        assert torch.allclose(out1[0], out2[0])
        assert torch.allclose(out1[1], out2[1])

    def test_model_gradient_flow(self, sample_model, encoder_config):
        """Gradients should flow through the model."""
        x = torch.randn(4, encoder_config.input_dim, requires_grad=True)
        count_pred, dist_pred, _ = sample_model(x)

        loss = count_pred.sum() + dist_pred.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ==================== Loss Tests ====================

class TestTwoHeadLoss:
    """Tests for the multi-task loss function."""

    def test_loss_computation(self):
        """Loss should be computed correctly."""
        criterion = TwoHeadLoss()

        count_pred = torch.tensor([10.0, 20.0, 30.0])
        count_target = torch.tensor([10.0, 20.0, 30.0])
        dist_pred = torch.tensor([3.0, 4.0])
        dist_target = torch.tensor([3.0, 4.0])
        query_types = torch.tensor([0, 0, 0, 1, 1])

        # Concatenate for proper masking
        count_pred_full = torch.tensor([10.0, 20.0, 30.0, 0.0, 0.0])
        dist_pred_full = torch.tensor([0.0, 0.0, 0.0, 3.0, 4.0])
        targets = torch.tensor([10.0, 20.0, 30.0, 3.0, 4.0])

        total_loss, count_loss, dist_loss = criterion(
            count_pred_full, targets, dist_pred_full, targets, query_types
        )

        # Perfect predictions should give near-zero loss
        assert total_loss < 0.01

    def test_loss_masking(self):
        """Loss should only compute for correct query types."""
        criterion = TwoHeadLoss()

        count_pred = torch.tensor([10.0, 100.0])  # Second is wrong type
        dist_pred = torch.tensor([100.0, 3.0])    # First is wrong type
        targets = torch.tensor([10.0, 3.0])
        query_types = torch.tensor([0, 1])  # count, distance

        total_loss, count_loss, dist_loss = criterion(
            count_pred, targets, dist_pred, targets, query_types
        )

        # Loss should be low because correct predictions are matched
        assert count_loss < 0.01  # count_pred[0] matches target[0]
        assert dist_loss < 0.01   # dist_pred[1] matches target[1]


# ==================== Baseline Tests ====================

class TestBaselines:
    """Tests for non-learned baselines."""

    def test_mean_baseline(self, sample_graph):
        """Mean baseline should predict training mean."""
        queries = [
            Query(QueryType.COUNT, 0, "A", radius=1),
            Query(QueryType.COUNT, 1, "A", radius=2),
            Query(QueryType.COUNT, 2, "A", radius=1),
        ]
        results = [execute_query(sample_graph, q) for q in queries]

        baseline = MeanBaseline().fit(queries, results)

        # Prediction should be the mean
        expected_mean = np.mean(results)
        pred = baseline.predict(queries[0])
        assert abs(pred - expected_mean) < 0.01

    def test_histogram_baseline(self, sample_graph):
        """Histogram baseline should use per-bin statistics."""
        queries = [
            Query(QueryType.COUNT, 0, "A", start_degree_bin=0, radius=1),
            Query(QueryType.COUNT, 1, "A", start_degree_bin=0, radius=1),
            Query(QueryType.COUNT, 5, "A", start_degree_bin=1, radius=1),
        ]
        results = [execute_query(sample_graph, q) for q in queries]

        baseline = HistogramBaseline().fit(queries, results)

        # Should predict based on histogram bin
        pred = baseline.predict(queries[0])
        expected = np.mean([results[0], results[1]])  # Same bin
        assert abs(pred - expected) < 0.01

    def test_sampling_baseline(self, sample_graph):
        """Sampling baseline should approximate counts."""
        baseline = SamplingBaseline(sample_graph, sample_fraction=0.5, seed=42)

        query = Query(QueryType.COUNT, 0, "A", radius=1)
        true_count = execute_query(sample_graph, query)
        pred = baseline.predict(query)

        # Should be in reasonable range (sampling is approximate)
        assert 0 <= pred <= true_count * 3

    def test_random_walk_baseline(self, sample_graph):
        """Random walk baseline should produce reasonable estimates."""
        baseline = RandomWalkBaseline(sample_graph, num_walks=50, seed=42)

        query = Query(QueryType.COUNT, 0, "A", radius=1)
        pred = baseline.predict(query)

        # Should be positive
        assert pred >= 0

    def test_evaluate_baseline(self, sample_graph):
        """evaluate_baseline should compute correct metrics."""
        queries = [
            Query(QueryType.COUNT, i, "A", radius=1)
            for i in range(5)
        ]
        ground_truth = [execute_query(sample_graph, q) for q in queries]

        baseline = MeanBaseline().fit(queries, ground_truth)
        results = evaluate_baseline(baseline, queries, ground_truth)

        assert "count" in results
        assert "mae" in results["count"]
        assert "rmse" in results["count"]
        assert results["count"]["mae"] >= 0
        assert results["count"]["rmse"] >= results["count"]["mae"]


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self, sample_graph, encoder_config):
        """Test full pipeline from query to prediction."""
        encoder = QueryEncoder(encoder_config)
        model = GraphSurrogate(
            input_dim=encoder_config.input_dim,
            hidden_dim=32,
            latent_dim=8,
            num_layers=2,
        )
        model.eval()

        # Generate queries
        sampler = QuerySampler(sample_graph, seed=42, node_types=["A", "B"])
        queries = [sampler.sample_query() for _ in range(20)]

        # Encode and predict
        encoded = encoder.encode_batch(queries)
        with torch.no_grad():
            count_pred, dist_pred, _ = model(encoded)

        # Predictions should have correct shape and be non-negative
        assert count_pred.shape == (20,)
        assert dist_pred.shape == (20,)
        assert (count_pred >= 0).all()
        assert (dist_pred >= 0).all()

    def test_training_step(self, sample_graph, encoder_config):
        """Test a single training step."""
        encoder = QueryEncoder(encoder_config)
        model = GraphSurrogate(
            input_dim=encoder_config.input_dim,
            hidden_dim=32,
            latent_dim=8,
            num_layers=2,
        )
        criterion = TwoHeadLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Generate batch
        sampler = QuerySampler(sample_graph, seed=42, node_types=["A", "B"])
        queries = [sampler.sample_query() for _ in range(8)]
        results = [execute_query(sample_graph, q) for q in queries]

        X = encoder.encode_batch(queries)
        y = torch.tensor(results, dtype=torch.float32)
        qt = torch.tensor([0 if q.query_type == QueryType.COUNT else 1 for q in queries])

        # Training step
        model.train()
        optimizer.zero_grad()
        count_pred, dist_pred, _ = model(X)
        loss, _, _ = criterion(count_pred, y, dist_pred, y, qt)
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss)
        assert loss.item() > 0


# ==================== Edge Case Tests ====================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_isolated_node(self):
        """Handle isolated node correctly."""
        G = nx.Graph()
        G.add_node(0, node_type="A")
        G.add_node(1, node_type="B")
        # No edges

        neighbors = get_k_hop_neighbors(G, 0, k=1)
        assert neighbors == set()

        count = count_neighborhood(G, 0, radius=1, attr_filter=AttributeFilter())
        assert count == 0

    def test_missing_node(self):
        """Handle non-existent node gracefully."""
        G = nx.Graph()
        G.add_node(0, node_type="A")

        dist = compute_shortest_distance(G, 999, "A", max_hops=6)
        assert dist == INF_DISTANCE

    def test_self_loop(self):
        """Handle self-loops correctly."""
        G = nx.Graph()
        G.add_node(0, node_type="A")
        G.add_edge(0, 0)  # Self-loop

        neighbors = get_k_hop_neighbors(G, 0, k=1)
        # Self-loop shouldn't count as neighbor
        assert 0 not in neighbors

    def test_empty_graph(self):
        """Handle empty graph gracefully."""
        G = nx.Graph()

        # This should not crash
        sampler = QuerySampler(G, seed=42, node_types=["A"])
        # Can't sample from empty graph, but should handle gracefully

    def test_large_radius(self, sample_graph):
        """Handle radius larger than graph diameter."""
        # Get all reachable nodes
        neighbors_r10 = get_k_hop_neighbors(sample_graph, 0, k=10)
        neighbors_r100 = get_k_hop_neighbors(sample_graph, 0, k=100)

        # Should converge to same set
        assert neighbors_r10 == neighbors_r100


# ==================== Cardinality Baseline Tests (Gap 5) ====================

class TestCardinalityBaselines:
    """Tests for learned cardinality estimation baselines."""

    def test_enhanced_histogram_fit(self, sample_graph):
        """EnhancedHistogramEstimator should fit without errors."""
        sampler = QuerySampler(sample_graph, seed=42, node_types=["A", "B"])
        queries = [sampler.sample_query() for _ in range(100)]
        results = [execute_query(sample_graph, q) for q in queries]

        estimator = EnhancedHistogramEstimator()
        estimator.fit(queries, results)

        assert len(estimator.histograms) > 0

    def test_enhanced_histogram_predict(self, sample_graph):
        """EnhancedHistogramEstimator should produce reasonable predictions."""
        sampler = QuerySampler(sample_graph, seed=42, node_types=["A", "B"])
        queries = [sampler.sample_query() for _ in range(100)]
        results = [execute_query(sample_graph, q) for q in queries]

        estimator = EnhancedHistogramEstimator().fit(queries, results)

        test_query = Query(QueryType.COUNT, 0, "A", radius=1)
        estimate = estimator.predict(test_query)

        assert estimate.estimate >= 0
        assert estimate.confidence >= 0
        assert estimate.method in ["joint_histogram", "independence_assumption", "global_mean"]

    def test_neurocard_inspired(self, sample_graph):
        """NeuroCardInspired estimator should work correctly."""
        sampler = QuerySampler(sample_graph, seed=42, node_types=["A", "B"])
        queries = [sampler.sample_query() for _ in range(100)]
        results = [execute_query(sample_graph, q) for q in queries]

        estimator = NeuroCardInspired().fit(queries, results)

        predictions = estimator.predict_batch(queries[:10])
        assert len(predictions) == 10
        assert all(p >= 0 for p in predictions)

    def test_deepdb_inspired(self, sample_graph):
        """DeepDBInspired estimator should work correctly."""
        sampler = QuerySampler(sample_graph, seed=42, node_types=["A", "B"])
        queries = [sampler.sample_query() for _ in range(100)]
        results = [execute_query(sample_graph, q) for q in queries]

        estimator = DeepDBInspired().fit(queries, results)

        test_query = Query(QueryType.COUNT, 0, "A", radius=1)
        estimate = estimator.predict(test_query)

        assert estimate.estimate >= 0

    def test_evaluate_cardinality_baselines(self, sample_graph):
        """evaluate_cardinality_baselines should compute metrics."""
        sampler = QuerySampler(sample_graph, seed=42, node_types=["A", "B"])
        queries = [sampler.sample_query() for _ in range(50)]
        results = [execute_query(sample_graph, q) for q in queries]

        baselines = {
            "histogram": EnhancedHistogramEstimator().fit(queries, results),
            "neurocard": NeuroCardInspired().fit(queries, results),
        }

        eval_results = evaluate_cardinality_baselines(queries, results, baselines)

        assert "histogram" in eval_results
        assert "neurocard" in eval_results
        assert "overall" in eval_results["histogram"]
        assert "mae" in eval_results["histogram"]["overall"]


# ==================== Analysis Module Tests (Gap 7, 9, 10) ====================

class TestAnalysisModule:
    """Tests for the analysis module components."""

    def test_failure_case_analyzer_init(self, sample_graph, encoder_config):
        """FailureCaseAnalyzer should initialize correctly."""
        from scripts.analysis import FailureCaseAnalyzer

        model = GraphSurrogate(
            input_dim=encoder_config.input_dim,
            hidden_dim=32,
            latent_dim=8,
            num_layers=2,
        )
        encoder = QueryEncoder(encoder_config)

        analyzer = FailureCaseAnalyzer(model, encoder, sample_graph)

        assert analyzer.degree_mean > 0
        assert len(analyzer.node_types) == sample_graph.number_of_nodes()

    def test_analyze_query(self, sample_graph, encoder_config):
        """FailureCaseAnalyzer should extract query features."""
        from scripts.analysis import FailureCaseAnalyzer

        model = GraphSurrogate(input_dim=encoder_config.input_dim)
        encoder = QueryEncoder(encoder_config)
        analyzer = FailureCaseAnalyzer(model, encoder, sample_graph)

        query = Query(QueryType.COUNT, 0, "A", radius=1)
        features = analyzer.analyze_query(query)

        assert "degree" in features
        assert "clustering_coef" in features
        assert "is_hub" in features
        assert "query_radius" in features

    def test_find_failure_cases(self, sample_graph, encoder_config):
        """FailureCaseAnalyzer should find failure cases."""
        from scripts.analysis import FailureCaseAnalyzer

        model = GraphSurrogate(input_dim=encoder_config.input_dim)
        encoder = QueryEncoder(encoder_config)
        analyzer = FailureCaseAnalyzer(model, encoder, sample_graph)

        sampler = QuerySampler(sample_graph, seed=42, node_types=["A", "B"])
        queries = [sampler.sample_query() for _ in range(50)]

        failures = analyzer.find_failure_cases(queries, top_k=10)

        assert len(failures) <= 10
        if failures:
            assert failures[0].absolute_error >= failures[-1].absolute_error

    def test_uncertainty_estimator(self, sample_graph, encoder_config):
        """UncertaintyEstimator should produce uncertainty estimates."""
        from scripts.analysis import UncertaintyEstimator

        model = GraphSurrogate(
            input_dim=encoder_config.input_dim,
            hidden_dim=32,
            dropout=0.1,
        )
        encoder = QueryEncoder(encoder_config)
        estimator = UncertaintyEstimator(model, encoder)

        sampler = QuerySampler(sample_graph, seed=42, node_types=["A", "B"])
        queries = [sampler.sample_query() for _ in range(10)]

        uncertainties = estimator.mc_dropout_uncertainty(queries, n_samples=5)

        assert "count_mean" in uncertainties
        assert "count_std" in uncertainties
        assert uncertainties["count_std"].shape[0] == 10
        assert (uncertainties["count_std"] >= 0).all()

    def test_graph_surrogate_with_uncertainty(self, encoder_config):
        """GraphSurrogateWithUncertainty should output variances."""
        from scripts.analysis import GraphSurrogateWithUncertainty

        model = GraphSurrogateWithUncertainty(
            input_dim=encoder_config.input_dim,
            hidden_dim=32,
            latent_dim=8,
        )

        x = torch.randn(5, encoder_config.input_dim)
        count_pred, dist_pred, count_var, dist_var, z = model(x)

        assert count_pred.shape == (5,)
        assert count_var.shape == (5,)
        assert (count_var > 0).all()
        assert (dist_var > 0).all()

    def test_scalability_benchmark_graph_generation(self):
        """ScalabilityBenchmark should generate graphs of specified size."""
        from scripts.analysis import ScalabilityBenchmark

        with TemporaryDirectory() as tmp_dir:
            benchmark = ScalabilityBenchmark(Path(tmp_dir))
            G = benchmark.generate_scalable_graph(1000, avg_degree=5, num_types=3)

            assert G.number_of_nodes() == 1000
            assert G.number_of_edges() == 2500
            assert len(set(nx.get_node_attributes(G, "node_type").values())) == 3

    def test_cross_graph_generalization(self, sample_graph, encoder_config):
        """CrossGraphGeneralization should create graph variants."""
        from scripts.analysis import CrossGraphGeneralization

        with TemporaryDirectory() as tmp_dir:
            cross_graph = CrossGraphGeneralization(Path(tmp_dir))
            variants = cross_graph.create_graph_variants(sample_graph, num_variants=2)

            assert len(variants) >= 2
            for name, G in variants:
                assert isinstance(G, nx.Graph)

    def test_incremental_update_analyzer(self, sample_graph):
        """IncrementalUpdateAnalyzer should simulate graph evolution."""
        from scripts.analysis import IncrementalUpdateAnalyzer

        with TemporaryDirectory() as tmp_dir:
            analyzer = IncrementalUpdateAnalyzer(Path(tmp_dir))
            evolutions = analyzer.simulate_graph_evolution(
                sample_graph,
                change_fractions=[0.1, 0.2],
            )

            assert len(evolutions) == 2
            for name, G_mod, info in evolutions:
                assert "fraction" in info
                assert G_mod.number_of_nodes() == sample_graph.number_of_nodes()


# ==================== Query Optimizer Tests (Gap 6) ====================

class TestQueryOptimizer:
    """Tests for query optimizer integration."""

    def test_optimizer_initialization(self, sample_graph, encoder_config):
        """GraphSurrogateOptimizer should initialize correctly."""
        from scripts.optimizer_integration import GraphSurrogateOptimizer

        model = GraphSurrogate(input_dim=encoder_config.input_dim)
        encoder = QueryEncoder(encoder_config)

        optimizer = GraphSurrogateOptimizer(model, encoder, sample_graph)

        assert optimizer.num_nodes == sample_graph.number_of_nodes()
        assert optimizer.avg_degree > 0
        assert len(optimizer.type_distribution) > 0

    def test_estimate_cardinality(self, sample_graph, encoder_config):
        """Optimizer should estimate query cardinality."""
        from scripts.optimizer_integration import GraphSurrogateOptimizer

        model = GraphSurrogate(input_dim=encoder_config.input_dim)
        encoder = QueryEncoder(encoder_config)
        optimizer = GraphSurrogateOptimizer(model, encoder, sample_graph)

        query = Query(QueryType.COUNT, 0, "A", radius=1)
        cardinality = optimizer.estimate_cardinality(query)

        assert cardinality >= 0

    def test_optimize_traversal_order(self, sample_graph, encoder_config):
        """Optimizer should produce traversal plan."""
        from scripts.optimizer_integration import GraphSurrogateOptimizer

        model = GraphSurrogate(input_dim=encoder_config.input_dim)
        encoder = QueryEncoder(encoder_config)
        optimizer = GraphSurrogateOptimizer(model, encoder, sample_graph)

        plan = optimizer.optimize_traversal_order(["A", "B"], radius=2)

        assert len(plan.operations) > 0
        assert plan.total_cost >= 0
        assert len(plan.notes) > 0

    def test_recommend_indexes(self, sample_graph, encoder_config):
        """Optimizer should recommend indexes."""
        from scripts.optimizer_integration import GraphSurrogateOptimizer

        model = GraphSurrogate(input_dim=encoder_config.input_dim)
        encoder = QueryEncoder(encoder_config)
        optimizer = GraphSurrogateOptimizer(model, encoder, sample_graph)

        queries = [
            Query(QueryType.COUNT, 0, "A", radius=1),
            Query(QueryType.COUNT, 1, "A", radius=2),
            Query(QueryType.COUNT, 2, "B", radius=1),
        ] * 10

        recommendations = optimizer.recommend_indexes(queries)

        assert isinstance(recommendations, list)

    def test_parallel_execution_planning(self, sample_graph, encoder_config):
        """Optimizer should plan parallel execution."""
        from scripts.optimizer_integration import GraphSurrogateOptimizer

        model = GraphSurrogate(input_dim=encoder_config.input_dim)
        encoder = QueryEncoder(encoder_config)
        optimizer = GraphSurrogateOptimizer(model, encoder, sample_graph)

        queries = [Query(QueryType.COUNT, i % 20, "A", radius=1) for i in range(20)]
        plan = optimizer.plan_parallel_execution(queries, max_parallelism=4)

        assert "parallel_groups" in plan
        assert len(plan["parallel_groups"]) == 4
        assert plan["speedup"] >= 1.0

    def test_cost_based_router(self, sample_graph, encoder_config):
        """CostBasedQueryRouter should route queries correctly."""
        from scripts.optimizer_integration import (
            GraphSurrogateOptimizer,
            CostBasedQueryRouter,
        )

        model = GraphSurrogate(input_dim=encoder_config.input_dim)
        encoder = QueryEncoder(encoder_config)
        optimizer = GraphSurrogateOptimizer(model, encoder, sample_graph)
        router = CostBasedQueryRouter(optimizer)

        query = Query(QueryType.COUNT, 0, "A", radius=1)
        routing = router.route_query(query)

        assert "strategy" in routing
        assert routing["strategy"] in [
            "direct_traversal",
            "indexed_traversal",
            "surrogate_approximation",
            "cached",
        ]


# ==================== Ablation Study Tests (Gap 8) ====================

class TestAblationStudies:
    """Tests for ablation study framework."""

    def test_encoding_ablation_configs(self, encoder_config):
        """AblationStudyRunner should create ablation configs."""
        from scripts.analysis import AblationStudyRunner

        with TemporaryDirectory() as tmp_dir:
            G = nx.gnm_random_graph(50, 100, seed=42)
            for node in G.nodes():
                G.nodes[node]["node_type"] = "A" if node < 25 else "B"

            runner = AblationStudyRunner(G, {}, Path(tmp_dir))

            no_degree = runner._remove_degree_bins(encoder_config)
            assert len(no_degree.degree_bins) == 1

            no_radius = runner._remove_radius(encoder_config)
            assert len(no_radius.radii) == 1

            no_attr = runner._remove_attr_filter(encoder_config)
            assert len(no_attr.attribute_names) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
