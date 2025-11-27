import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm


@dataclass
class StructuralFeatureConfig:
    compute_clustering: bool = True
    compute_pagerank: bool = True
    compute_core_number: bool = True
    compute_neighbor_degree: bool = True
    compute_2hop_size: bool = True
    compute_triangles: bool = True
    pagerank_alpha: float = 0.85
    max_2hop_sample: int = 10000

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StructuralFeatureConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class StructuralFeatureComputer:
    FEATURE_NAMES = [
        'degree',
        'log_degree',
        'clustering_coefficient',
        'core_number',
        'pagerank',
        'avg_neighbor_degree',
        'two_hop_size',
        'log_two_hop_size',
    ]

    def __init__(self, config: Optional[StructuralFeatureConfig] = None):
        self.config = config or StructuralFeatureConfig()
        self.features: Optional[dict] = None
        self.feature_tensor: Optional[torch.Tensor] = None
        self.node_to_idx: Optional[dict] = None
        self.stats: Optional[dict] = None

    @property
    def num_features(self) -> int:
        return len(self.FEATURE_NAMES)

    def compute(self, G: nx.Graph, show_progress: bool = True) -> dict:
        nodes = list(G.nodes())
        n = len(nodes)
        self.node_to_idx = {node: i for i, node in enumerate(nodes)}

        features = {name: np.zeros(n) for name in self.FEATURE_NAMES}

        if show_progress:
            print("Computing degree features...")
        degrees = dict(G.degree())
        for node, idx in self.node_to_idx.items():
            features['degree'][idx] = degrees[node]
            features['log_degree'][idx] = np.log1p(degrees[node])

        if self.config.compute_clustering:
            if show_progress:
                print("Computing clustering coefficients...")
            clustering = nx.clustering(G)
            for node, idx in self.node_to_idx.items():
                features['clustering_coefficient'][idx] = clustering[node]

        if self.config.compute_core_number:
            if show_progress:
                print("Computing core numbers...")
            try:
                core_numbers = nx.core_number(G)
                for node, idx in self.node_to_idx.items():
                    features['core_number'][idx] = core_numbers[node]
            except nx.NetworkXError:
                pass

        if self.config.compute_pagerank:
            if show_progress:
                print("Computing PageRank...")
            try:
                pagerank = nx.pagerank(G, alpha=self.config.pagerank_alpha, max_iter=100)
                for node, idx in self.node_to_idx.items():
                    features['pagerank'][idx] = pagerank[node]
            except nx.PowerIterationFailedConvergence:
                total_edges = G.number_of_edges() * 2
                for node, idx in self.node_to_idx.items():
                    features['pagerank'][idx] = degrees[node] / max(total_edges, 1)

        if self.config.compute_neighbor_degree:
            if show_progress:
                print("Computing average neighbor degrees...")
            avg_neighbor_deg = nx.average_neighbor_degree(G)
            for node, idx in self.node_to_idx.items():
                features['avg_neighbor_degree'][idx] = avg_neighbor_deg.get(node, 0)

        if self.config.compute_2hop_size:
            if show_progress:
                print("Computing 2-hop neighborhood sizes...")
            iterator = tqdm(self.node_to_idx.items(), desc="2-hop sizes") if show_progress else self.node_to_idx.items()
            for node, idx in iterator:
                two_hop_size = self._compute_2hop_size(G, node)
                features['two_hop_size'][idx] = two_hop_size
                features['log_two_hop_size'][idx] = np.log1p(two_hop_size)

        self.features = features
        self._compute_stats()
        self._create_tensor()

        return features

    def _compute_2hop_size(self, G: nx.Graph, node) -> int:
        try:
            neighbors_1 = set(G.neighbors(node))
            neighbors_2 = set()
            for n1 in neighbors_1:
                neighbors_2.update(G.neighbors(n1))
            neighbors_2.discard(node)
            neighbors_2 -= neighbors_1
            return len(neighbors_1) + len(neighbors_2)
        except Exception:
            return 0

    def _compute_stats(self):
        self.stats = {}
        for name, values in self.features.items():
            self.stats[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values) + 1e-8),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }

    def _create_tensor(self):
        n = len(self.node_to_idx)
        tensor = np.zeros((n, len(self.FEATURE_NAMES)))

        for i, name in enumerate(self.FEATURE_NAMES):
            values = self.features[name]
            mean = self.stats[name]['mean']
            std = self.stats[name]['std']
            tensor[:, i] = (values - mean) / std

        self.feature_tensor = torch.tensor(tensor, dtype=torch.float32)

    def get_features(self, node) -> torch.Tensor:
        if self.feature_tensor is None:
            raise ValueError("Features not computed. Call compute() first.")
        idx = self.node_to_idx.get(node)
        if idx is None:
            return torch.zeros(len(self.FEATURE_NAMES))
        return self.feature_tensor[idx]

    def get_features_batch(self, nodes: list) -> torch.Tensor:
        if self.feature_tensor is None:
            raise ValueError("Features not computed. Call compute() first.")
        indices = [self.node_to_idx.get(n, 0) for n in nodes]
        return self.feature_tensor[indices]

    def save(self, path: Path):
        path = Path(path)
        torch.save(self.feature_tensor, path / "structural_features.pt")

        with open(path / "structural_node_mapping.json", "w") as f:
            mapping = {str(k): v for k, v in self.node_to_idx.items()}
            json.dump(mapping, f)

        with open(path / "structural_feature_stats.json", "w") as f:
            json.dump({
                'stats': self.stats,
                'config': self.config.to_dict(),
                'feature_names': self.FEATURE_NAMES,
            }, f, indent=2)

    def load(self, path: Path, G: Optional[nx.Graph] = None):
        path = Path(path)
        self.feature_tensor = torch.load(path / "structural_features.pt", weights_only=True)

        with open(path / "structural_node_mapping.json") as f:
            mapping = json.load(f)
            self.node_to_idx = {}
            for k, v in mapping.items():
                try:
                    self.node_to_idx[int(k)] = v
                except ValueError:
                    self.node_to_idx[k] = v

        with open(path / "structural_feature_stats.json") as f:
            data = json.load(f)
            self.stats = data['stats']
            self.config = StructuralFeatureConfig.from_dict(data.get('config', {}))

    def exists(self, path: Path) -> bool:
        path = Path(path)
        return (
            (path / "structural_features.pt").exists() and
            (path / "structural_node_mapping.json").exists() and
            (path / "structural_feature_stats.json").exists()
        )


def compute_or_load_features(
    G: nx.Graph,
    cache_dir: Path,
    config: Optional[StructuralFeatureConfig] = None,
    force_recompute: bool = False,
    show_progress: bool = True,
) -> StructuralFeatureComputer:
    computer = StructuralFeatureComputer(config)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not force_recompute and computer.exists(cache_dir):
        if show_progress:
            print("Loading cached structural features...")
        computer.load(cache_dir, G)
    else:
        if show_progress:
            print("Computing structural features...")
        computer.compute(G, show_progress=show_progress)
        computer.save(cache_dir)
        if show_progress:
            print(f"Structural features saved to {cache_dir}")

    return computer
