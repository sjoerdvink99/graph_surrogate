import random
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional

import networkx as nx
import numpy as np


INF_DISTANCE = 999


class QueryType(str, Enum):
    COUNT = "count"
    DISTANCE = "distance"


@dataclass
class AttributeFilter:
    name: Optional[str] = None
    value: Optional[str] = None


@dataclass
class Query:
    query_type: QueryType
    start_node_id: int
    start_node_type: str
    start_degree_bin: int
    radius: Optional[int] = None
    attribute_filter: Optional[AttributeFilter] = None
    target_type: Optional[str] = None
    max_hops: Optional[int] = None

    def __post_init__(self):
        if self.attribute_filter is None:
            self.attribute_filter = AttributeFilter()


def execute_query(G: nx.Graph, query: Query) -> int:
    if query.query_type == QueryType.COUNT:
        return execute_count_query(G, query)
    return execute_distance_query(G, query)


def execute_count_query(G: nx.Graph, query: Query) -> int:
    if query.start_node_id not in G:
        return 0

    visited = {query.start_node_id}
    current_layer = {query.start_node_id}

    for _ in range(query.radius):
        next_layer = set()
        for node in current_layer:
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_layer.add(neighbor)
        current_layer = next_layer
        if not current_layer:
            break

    if query.attribute_filter.name is None:
        return len(visited) - 1

    count = 0
    for node in visited:
        if node == query.start_node_id:
            continue
        node_data = G.nodes[node]
        if node_data.get(query.attribute_filter.name) == query.attribute_filter.value:
            count += 1
    return count


def execute_distance_query(G: nx.Graph, query: Query) -> int:
    if query.start_node_id not in G:
        return INF_DISTANCE

    visited = {query.start_node_id}
    current_layer = {query.start_node_id}

    for dist in range(1, query.max_hops + 1):
        next_layer = set()
        for node in current_layer:
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_layer.add(neighbor)

                    node_data = G.nodes[neighbor]
                    if node_data.get("node_type") == query.target_type:
                        return dist

        current_layer = next_layer
        if not current_layer:
            break

    return INF_DISTANCE


DEGREE_STRATA = {
    'very_low': (0, 2),
    'low': (3, 10),
    'medium': (11, 50),
    'high': (51, 200),
    'very_high': (201, float('inf')),
}


class SamplingStrategy(str, Enum):
    UNIFORM = "uniform"
    STRATIFIED = "stratified"
    DEGREE_WEIGHTED = "degree_weighted"


@dataclass
class SamplingConfig:
    strategy: SamplingStrategy = SamplingStrategy.STRATIFIED
    count_ratio: float = 0.5
    radii: list[int] = None
    max_hops_options: list[int] = None
    attribute_filter_prob: float = 0.5

    def __post_init__(self):
        if self.radii is None:
            self.radii = [1, 2, 3]
        if self.max_hops_options is None:
            self.max_hops_options = [3, 4, 5, 6]


class QuerySampler:
    def __init__(self, G: nx.Graph, seed: int = 42, node_types: Optional[list[str]] = None):
        self.G = G
        self.nodes = list(G.nodes())
        self.node_types = node_types or self._get_node_types()
        self.rng = random.Random(seed)
        self.attr_options = self._build_attr_options()

    def _get_node_types(self) -> list[str]:
        types = set()
        for _, data in self.G.nodes(data=True):
            types.add(data.get("node_type", "default"))
        return sorted(types) if types else ["default"]

    def _build_attr_options(self) -> list[tuple[Optional[str], list]]:
        options = [(None, [None])]
        attr_values = {}
        for _, data in self.G.nodes(data=True):
            for key, val in data.items():
                if key not in ("node_type", "degree_bin") and val is not None:
                    if key not in attr_values:
                        attr_values[key] = set()
                    attr_values[key].add(val)
        options.append(("node_type", self.node_types))
        for attr, values in attr_values.items():
            if values and len(values) <= 100:
                options.append((attr, sorted(values, key=str)))
        return options

    def _get_degree_bin(self, node_id) -> int:
        degree = self.G.degree(node_id)
        if degree <= 2:
            return 0
        elif degree <= 5:
            return 1
        elif degree <= 10:
            return 2
        elif degree <= 20:
            return 3
        elif degree <= 50:
            return 4
        elif degree <= 100:
            return 5
        elif degree <= 500:
            return 6
        elif degree <= 1000:
            return 7
        return 8

    def sample_count_query(self) -> Query:
        node_id = self.rng.choice(self.nodes)
        node_type = self.G.nodes[node_id].get("node_type", "default")
        degree_bin = self._get_degree_bin(node_id)
        radius = self.rng.choice([1, 2, 3])
        return Query(
            query_type=QueryType.COUNT,
            start_node_id=node_id,
            start_node_type=node_type,
            start_degree_bin=degree_bin,
            radius=radius,
            attribute_filter=AttributeFilter(),
        )

    def sample_distance_query(self) -> Query:
        node_id = self.rng.choice(self.nodes)
        node_type = self.G.nodes[node_id].get("node_type", "default")
        degree_bin = self._get_degree_bin(node_id)
        target_type = self.rng.choice(self.node_types)
        max_hops = self.rng.choice([3, 4, 5, 6])
        return Query(
            query_type=QueryType.DISTANCE,
            start_node_id=node_id,
            start_node_type=node_type,
            start_degree_bin=degree_bin,
            target_type=target_type,
            max_hops=max_hops,
        )

    def sample_query(self) -> Query:
        if self.rng.random() < 0.5:
            return self.sample_count_query()
        return self.sample_distance_query()

    def generate_dataset(self, num_samples: int, show_progress: bool = False) -> list[tuple[Query, int]]:
        dataset = []
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(num_samples), desc="Generating queries")
        else:
            iterator = range(num_samples)
        for _ in iterator:
            query = self.sample_query()
            result = execute_query(self.G, query)
            dataset.append((query, result))
        return dataset


class StratifiedQuerySampler:
    def __init__(
        self,
        G: nx.Graph,
        seed: int = 42,
        node_types: Optional[list[str]] = None,
        config: Optional[SamplingConfig] = None,
    ):
        self.G = G
        self.nodes = list(G.nodes())
        self.node_types = node_types or self._get_node_types()
        self.config = config or SamplingConfig()
        self._build_strata()
        self.attr_options = self._build_attr_options()
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def _get_node_types(self) -> list[str]:
        types = set()
        for _, data in self.G.nodes(data=True):
            types.add(data.get("node_type", "default"))
        return sorted(types) if types else ["default"]

    def _build_strata(self):
        self.strata = defaultdict(list)
        self.node_degrees = {}
        for node in self.G.nodes():
            degree = self.G.degree(node)
            self.node_degrees[node] = degree
            for stratum_name, (low, high) in DEGREE_STRATA.items():
                if low <= degree <= high:
                    self.strata[stratum_name].append(node)
                    break
        self.stratum_weights = {}
        total_inverse = sum(1.0 / max(len(nodes), 1) for nodes in self.strata.values())
        for name, nodes in self.strata.items():
            if nodes:
                self.stratum_weights[name] = (1.0 / len(nodes)) / total_inverse
            else:
                self.stratum_weights[name] = 0.0
        self.stratum_names = list(self.strata.keys())
        self.stratum_probs = [self.stratum_weights[name] for name in self.stratum_names]

    def _build_attr_options(self) -> list[tuple[Optional[str], list]]:
        options = [(None, [None])]
        attr_values = {}
        for _, data in self.G.nodes(data=True):
            for key, val in data.items():
                if key not in ("node_type", "degree_bin") and val is not None:
                    if key not in attr_values:
                        attr_values[key] = set()
                    attr_values[key].add(val)
        options.append(("node_type", self.node_types))
        for attr, values in attr_values.items():
            if values and len(values) <= 100:
                options.append((attr, sorted(values, key=str)))
        return options

    def _get_degree_bin(self, node_id) -> int:
        degree = self.node_degrees.get(node_id, self.G.degree(node_id))
        if degree <= 2:
            return 0
        elif degree <= 5:
            return 1
        elif degree <= 10:
            return 2
        elif degree <= 20:
            return 3
        elif degree <= 50:
            return 4
        elif degree <= 100:
            return 5
        elif degree <= 500:
            return 6
        elif degree <= 1000:
            return 7
        return 8

    def sample_node(self) -> int:
        if self.config.strategy == SamplingStrategy.UNIFORM:
            return self.rng.choice(self.nodes)
        elif self.config.strategy == SamplingStrategy.STRATIFIED:
            stratum = self.np_rng.choice(self.stratum_names, p=self.stratum_probs)
            nodes = self.strata[stratum]
            if nodes:
                return self.rng.choice(nodes)
            return self.rng.choice(self.nodes)
        elif self.config.strategy == SamplingStrategy.DEGREE_WEIGHTED:
            total_degree = sum(self.node_degrees.values())
            probs = [self.node_degrees[n] / total_degree for n in self.nodes]
            return self.np_rng.choice(self.nodes, p=probs)
        else:
            return self.rng.choice(self.nodes)

    def sample_count_query(self) -> Query:
        node_id = self.sample_node()
        node_type = self.G.nodes[node_id].get("node_type", "default")
        degree_bin = self._get_degree_bin(node_id)
        radius = self.rng.choice(self.config.radii)
        if self.rng.random() < self.config.attribute_filter_prob and len(self.attr_options) > 1:
            attr_name, attr_values = self.rng.choice(self.attr_options[1:])
            attr_filter = AttributeFilter(name=attr_name, value=self.rng.choice(attr_values))
        else:
            attr_filter = AttributeFilter()
        return Query(
            query_type=QueryType.COUNT,
            start_node_id=node_id,
            start_node_type=node_type,
            start_degree_bin=degree_bin,
            radius=radius,
            attribute_filter=attr_filter,
        )

    def sample_distance_query(self) -> Query:
        node_id = self.sample_node()
        node_type = self.G.nodes[node_id].get("node_type", "default")
        degree_bin = self._get_degree_bin(node_id)
        target_type = self.rng.choice(self.node_types)
        max_hops = self.rng.choice(self.config.max_hops_options)
        return Query(
            query_type=QueryType.DISTANCE,
            start_node_id=node_id,
            start_node_type=node_type,
            start_degree_bin=degree_bin,
            target_type=target_type,
            max_hops=max_hops,
        )

    def sample_query(self) -> Query:
        if self.rng.random() < self.config.count_ratio:
            return self.sample_count_query()
        return self.sample_distance_query()

    def generate_dataset(self, num_samples: int, show_progress: bool = False) -> list[tuple[Query, int]]:
        dataset = []
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(num_samples), desc="Generating queries")
        else:
            iterator = range(num_samples)
        for _ in iterator:
            query = self.sample_query()
            result = execute_query(self.G, query)
            dataset.append((query, result))
        return dataset

    def generate_stratified_dataset(self, num_samples: int, show_progress: bool = False) -> list[tuple[Query, int]]:
        samples_per_stratum = num_samples // len(self.stratum_names)
        dataset = []
        if show_progress:
            from tqdm import tqdm
        for stratum_name in self.stratum_names:
            nodes = self.strata[stratum_name]
            if not nodes:
                continue
            stratum_samples = min(samples_per_stratum, len(nodes) * 10)
            if show_progress:
                iterator = tqdm(range(stratum_samples), desc=f"Stratum: {stratum_name}")
            else:
                iterator = range(stratum_samples)
            for _ in iterator:
                node_id = self.rng.choice(nodes)
                node_type = self.G.nodes[node_id].get("node_type", "default")
                degree_bin = self._get_degree_bin(node_id)
                if self.rng.random() < self.config.count_ratio:
                    query = Query(
                        query_type=QueryType.COUNT,
                        start_node_id=node_id,
                        start_node_type=node_type,
                        start_degree_bin=degree_bin,
                        radius=self.rng.choice(self.config.radii),
                        attribute_filter=AttributeFilter(),
                    )
                else:
                    query = Query(
                        query_type=QueryType.DISTANCE,
                        start_node_id=node_id,
                        start_node_type=node_type,
                        start_degree_bin=degree_bin,
                        target_type=self.rng.choice(self.node_types),
                        max_hops=self.rng.choice(self.config.max_hops_options),
                    )
                result = execute_query(self.G, query)
                dataset.append((query, result))
        self.rng.shuffle(dataset)
        return dataset[:num_samples]

    def get_stratum_stats(self) -> dict:
        stats = {}
        for name, nodes in self.strata.items():
            if nodes:
                degrees = [self.node_degrees[n] for n in nodes]
                stats[name] = {
                    'count': len(nodes),
                    'fraction': len(nodes) / len(self.nodes),
                    'min_degree': min(degrees),
                    'max_degree': max(degrees),
                    'mean_degree': np.mean(degrees),
                    'sampling_weight': self.stratum_weights[name],
                }
        return stats


def analyze_query_difficulty(G: nx.Graph, queries: list[Query], results: list[int]) -> dict:
    count_results = []
    dist_results = []
    for query, result in zip(queries, results):
        if query.query_type == QueryType.COUNT:
            count_results.append(result)
        else:
            dist_results.append(result)
    analysis = {}
    if count_results:
        count_results = np.array(count_results)
        analysis['count'] = {
            'num_queries': len(count_results),
            'mean': float(np.mean(count_results)),
            'std': float(np.std(count_results)),
            'median': float(np.median(count_results)),
            'min': int(np.min(count_results)),
            'max': int(np.max(count_results)),
            'p10': float(np.percentile(count_results, 10)),
            'p90': float(np.percentile(count_results, 90)),
            'p99': float(np.percentile(count_results, 99)),
            'zeros': int(np.sum(count_results == 0)),
        }
    if dist_results:
        dist_results = np.array(dist_results)
        unique, counts = np.unique(dist_results, return_counts=True)
        analysis['distance'] = {
            'num_queries': len(dist_results),
            'mean': float(np.mean(dist_results)),
            'std': float(np.std(dist_results)),
            'distribution': {int(u): int(c) for u, c in zip(unique, counts)},
            'unreachable_ratio': float(np.mean(dist_results >= INF_DISTANCE)),
        }
    return analysis
