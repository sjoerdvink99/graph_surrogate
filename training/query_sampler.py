import random
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional

import networkx as nx


class QueryType(str, Enum):
    COUNT = "count"
    DISTANCE = "distance"


@dataclass
class AttributeFilter:
    name: Optional[str] = None
    value: Optional[str | int] = None


@dataclass
class Query:
    query_type: QueryType
    start_node_id: int | str
    start_node_type: str
    start_degree_bin: int = 0
    radius: int = 1
    attribute_filter: AttributeFilter = None
    target_type: Optional[str] = None
    max_hops: int = 6

    def __post_init__(self):
        if self.attribute_filter is None:
            self.attribute_filter = AttributeFilter()

    def to_dict(self) -> dict:
        return {
            "query_type": self.query_type.value,
            "start_node_id": self.start_node_id,
            "start_node_type": self.start_node_type,
            "start_degree_bin": self.start_degree_bin,
            "radius": self.radius,
            "attribute_filter": {
                "name": self.attribute_filter.name,
                "value": self.attribute_filter.value,
            },
            "target_type": self.target_type,
            "max_hops": self.max_hops,
        }


INF_DISTANCE = 7


def get_k_hop_neighbors(G: nx.Graph, node: int | str, k: int) -> set:
    if k == 0:
        return set()

    neighbors = set()
    current_level = {node}

    for _ in range(k):
        next_level = set()
        for n in current_level:
            next_level.update(G.neighbors(n))
        neighbors.update(next_level)
        current_level = next_level - {node}

    neighbors.discard(node)
    return neighbors


def count_neighborhood(G: nx.Graph, start_node: int | str, radius: int, attr_filter: AttributeFilter) -> int:
    neighbors = get_k_hop_neighbors(G, start_node, radius)

    if attr_filter.name is None:
        return len(neighbors)

    count = 0
    for neighbor in neighbors:
        node_data = G.nodes[neighbor]
        if attr_filter.name in node_data:
            if node_data[attr_filter.name] == attr_filter.value:
                count += 1

    return count


def compute_shortest_distance(G: nx.Graph, start_node: int | str, target_type: str, max_hops: int) -> int:
    try:
        lengths = nx.single_source_shortest_path_length(G, start_node, cutoff=max_hops)
        min_dist = INF_DISTANCE
        for node, dist in lengths.items():
            if G.nodes[node].get("node_type") == target_type and dist > 0:
                min_dist = min(min_dist, dist)
        return min_dist
    except nx.NodeNotFound:
        return INF_DISTANCE


def execute_query(G: nx.Graph, query: Query) -> int:
    if query.query_type == QueryType.DISTANCE:
        return compute_shortest_distance(G, query.start_node_id, query.target_type, query.max_hops)
    return count_neighborhood(G, query.start_node_id, query.radius, query.attribute_filter)


class QuerySampler:
    def __init__(
        self,
        G: nx.Graph,
        seed: int = 42,
        node_types: Optional[list[str]] = None,
        radii: Optional[list[int]] = None,
        max_hops_options: Optional[list[int]] = None,
    ):
        self.G = G
        self.nodes = list(G.nodes())
        self.node_types = node_types or self._get_node_types()
        self.radii = radii or [1, 2, 3]
        self.max_hops_options = max_hops_options or [3, 4, 5, 6]

        self.attr_options = self._build_attr_options()

        random.seed(seed)

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

    def _get_degree_bin(self, node_id: int | str) -> int:
        degree = self.G.degree(node_id)
        if degree <= 2:
            return 0
        elif degree <= 5:
            return 1
        elif degree <= 10:
            return 2
        elif degree <= 50:
            return 3
        return 4

    def sample_count_query(self) -> Query:
        node_id = random.choice(self.nodes)
        node_type = self.G.nodes[node_id].get("node_type", "default")
        degree_bin = self._get_degree_bin(node_id)

        radius = random.choice(self.radii)

        attr_name, attr_values = random.choice(self.attr_options)
        if attr_name is None:
            attr_filter = AttributeFilter()
        else:
            attr_filter = AttributeFilter(name=attr_name, value=random.choice(attr_values))

        return Query(
            query_type=QueryType.COUNT,
            start_node_id=node_id,
            start_node_type=node_type,
            start_degree_bin=degree_bin,
            radius=radius,
            attribute_filter=attr_filter,
        )

    def sample_distance_query(self) -> Query:
        node_id = random.choice(self.nodes)
        node_type = self.G.nodes[node_id].get("node_type", "default")
        degree_bin = self._get_degree_bin(node_id)

        target_type = random.choice(self.node_types)
        max_hops = random.choice(self.max_hops_options)

        return Query(
            query_type=QueryType.DISTANCE,
            start_node_id=node_id,
            start_node_type=node_type,
            start_degree_bin=degree_bin,
            target_type=target_type,
            max_hops=max_hops,
        )

    def sample_query(self) -> Query:
        if random.random() < 0.5:
            return self.sample_count_query()
        return self.sample_distance_query()

    def generate_dataset(self, num_samples: int) -> list[tuple[Query, int]]:
        dataset = []
        for _ in range(num_samples):
            query = self.sample_query()
            result = execute_query(self.G, query)
            dataset.append((query, result))
        return dataset

    def generate_dataset_streaming(
        self,
        num_samples: int,
        chunk_size: int = 1000,
    ) -> Iterator[list[tuple[Query, int]]]:
        remaining = num_samples
        while remaining > 0:
            batch_size = min(chunk_size, remaining)
            batch = []
            for _ in range(batch_size):
                query = self.sample_query()
                result = execute_query(self.G, query)
                batch.append((query, result))
            yield batch
            remaining -= batch_size


class StreamingQueryDataset:
    def __init__(
        self,
        G: nx.Graph,
        num_samples: int,
        seed: int = 42,
        node_types: Optional[list[str]] = None,
    ):
        self.sampler = QuerySampler(G, seed=seed, node_types=node_types)
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[tuple[Query, int]]:
        random.seed(self.seed)
        for _ in range(self.num_samples):
            query = self.sampler.sample_query()
            result = execute_query(self.sampler.G, query)
            yield query, result


class ChunkedQueryGenerator:
    def __init__(
        self,
        G: nx.Graph,
        total_samples: int,
        chunk_size: int = 10000,
        seed: int = 42,
        node_types: Optional[list[str]] = None,
    ):
        self.G = G
        self.total_samples = total_samples
        self.chunk_size = chunk_size
        self.seed = seed
        self.node_types = node_types
        self._sampler = None

    @property
    def sampler(self) -> QuerySampler:
        if self._sampler is None:
            self._sampler = QuerySampler(self.G, seed=self.seed, node_types=self.node_types)
        return self._sampler

    def __len__(self) -> int:
        return (self.total_samples + self.chunk_size - 1) // self.chunk_size

    def __iter__(self) -> Iterator[list[tuple[Query, int]]]:
        random.seed(self.seed)
        remaining = self.total_samples

        while remaining > 0:
            batch_size = min(self.chunk_size, remaining)
            chunk = []

            for _ in range(batch_size):
                query = self.sampler.sample_query()
                result = execute_query(self.G, query)
                chunk.append((query, result))

            yield chunk
            remaining -= batch_size

    def get_attribute_options(self) -> list[tuple[Optional[str], list]]:
        return self.sampler.attr_options

    def get_node_types(self) -> list[str]:
        return self.sampler.node_types
