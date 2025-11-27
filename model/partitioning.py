"""
Multilevel Graph Partitioning for Hierarchical Road Network Embedding (RNE).

This module implements tree-structured graph partitioning using recursive bisection,
which creates a hierarchy where each node has ancestors at multiple levels.
The partitioning enables parameter sharing via the tree structure, making the
embedding more efficient than flat approaches.

Reference: Hierarchical Road Network Embedding paper methodology.
"""

import math
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import networkx as nx
import numpy as np
import torch


@dataclass
class PartitionNode:
    """A node in the partition tree."""
    level: int
    partition_id: int
    node_ids: list[int]
    children: list['PartitionNode'] = field(default_factory=list)
    parent: Optional['PartitionNode'] = None
    centroid: Optional[np.ndarray] = None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def size(self) -> int:
        return len(self.node_ids)


@dataclass
class PartitionTree:
    """Complete partition tree for hierarchical embedding."""
    root: PartitionNode
    num_levels: int
    level_sizes: list[int]
    node_to_ancestors: dict[int, list[tuple[int, int]]]  # node_id -> [(level, partition_id), ...]
    level_partitions: list[list[PartitionNode]]

    def get_ancestors(self, node_id: int) -> list[tuple[int, int]]:
        """Get list of (level, partition_id) for all ancestors of a node."""
        return self.node_to_ancestors.get(node_id, [])

    def get_partition_at_level(self, level: int, partition_id: int) -> Optional[PartitionNode]:
        """Get a specific partition node."""
        if level < len(self.level_partitions):
            for p in self.level_partitions[level]:
                if p.partition_id == partition_id:
                    return p
        return None


class GraphPartitioner:
    """
    Multilevel graph partitioner using recursive bisection.

    Creates a tree-structured hierarchy where each level l has partitions,
    and each node belongs to exactly one partition at each level.
    The partition hierarchy enables efficient parameter sharing in the
    Hierarchical RNE model.
    """

    def __init__(
        self,
        num_levels: int = 4,
        min_partition_size: int = 10,
        balance_factor: float = 1.5,
        use_coordinates: bool = True,
        seed: int = 42,
    ):
        """
        Initialize the graph partitioner.

        Args:
            num_levels: Number of hierarchy levels (depth of partition tree)
            min_partition_size: Minimum nodes per partition before stopping recursion
            balance_factor: Maximum ratio between largest and smallest partition
            use_coordinates: If True, use node coordinates for partitioning (road networks)
            seed: Random seed for reproducibility
        """
        self.num_levels = num_levels
        self.min_partition_size = min_partition_size
        self.balance_factor = balance_factor
        self.use_coordinates = use_coordinates
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def partition(self, G: nx.Graph) -> PartitionTree:
        """
        Create a multilevel partition tree for the graph.

        Args:
            G: NetworkX graph (undirected)

        Returns:
            PartitionTree containing the complete hierarchy
        """
        nodes = list(G.nodes())

        # Get node coordinates if available (for road networks)
        coords = self._get_coordinates(G, nodes)

        # Create root partition containing all nodes
        root = PartitionNode(
            level=0,
            partition_id=0,
            node_ids=nodes,
            centroid=coords.mean(axis=0) if coords is not None else None,
        )

        # Build partition tree recursively
        level_partitions = [[root]]
        partition_counter = [1]  # Counter for partition IDs at each level

        for level in range(1, self.num_levels):
            current_level = []
            partition_counter.append(0)

            for parent in level_partitions[level - 1]:
                if parent.size <= self.min_partition_size:
                    # Too small to partition further, this becomes a leaf
                    continue

                # Partition this node into children
                children = self._bisect_partition(
                    G, parent, coords, level, partition_counter[level]
                )

                for child in children:
                    child.parent = parent
                    parent.children.append(child)
                    current_level.append(child)
                    partition_counter[level] += 1

            if not current_level:
                # No more partitions created at this level
                break

            level_partitions.append(current_level)

        # Build ancestor mapping for each node
        node_to_ancestors = self._build_ancestor_map(level_partitions)

        # Compute level sizes (number of partitions at each level)
        level_sizes = [len(level) for level in level_partitions]

        return PartitionTree(
            root=root,
            num_levels=len(level_partitions),
            level_sizes=level_sizes,
            node_to_ancestors=node_to_ancestors,
            level_partitions=level_partitions,
        )

    def _get_coordinates(
        self, G: nx.Graph, nodes: list[int]
    ) -> Optional[np.ndarray]:
        """Extract node coordinates if available."""
        if not self.use_coordinates:
            return None

        # Check for common coordinate attributes
        coord_attrs = [('x', 'y'), ('lon', 'lat'), ('lng', 'lat'), ('pos',)]

        for attrs in coord_attrs:
            try:
                if len(attrs) == 1:
                    # Position stored as tuple/list
                    coords = []
                    for n in nodes:
                        pos = G.nodes[n].get(attrs[0])
                        if pos is not None:
                            coords.append(pos)
                        else:
                            return None
                    return np.array(coords, dtype=np.float32)
                else:
                    # Position stored as separate x, y attributes
                    coords = []
                    for n in nodes:
                        x = G.nodes[n].get(attrs[0])
                        y = G.nodes[n].get(attrs[1])
                        if x is not None and y is not None:
                            coords.append([float(x), float(y)])
                        else:
                            return None
                    return np.array(coords, dtype=np.float32)
            except (KeyError, TypeError, ValueError):
                continue

        # Fall back to spectral coordinates
        return self._compute_spectral_coordinates(G, nodes)

    def _compute_spectral_coordinates(
        self, G: nx.Graph, nodes: list[int], dim: int = 2
    ) -> np.ndarray:
        """Compute spectral coordinates for nodes without geometric positions."""
        subgraph = G.subgraph(nodes)
        n = len(nodes)

        if n <= dim + 1:
            # Too small, use random positions
            return self.rng.randn(n, dim).astype(np.float32)

        try:
            # Compute Laplacian eigenvectors
            L = nx.laplacian_matrix(subgraph).astype(np.float64)

            # Use sparse eigendecomposition for efficiency
            from scipy.sparse.linalg import eigsh
            eigenvalues, eigenvectors = eigsh(L, k=min(dim + 1, n - 1), which='SM')

            # Skip the first eigenvector (constant) and use the next `dim` eigenvectors
            coords = eigenvectors[:, 1:dim + 1].astype(np.float32)

            # Handle case where we have fewer eigenvectors than dimensions
            if coords.shape[1] < dim:
                padding = self.rng.randn(n, dim - coords.shape[1]).astype(np.float32) * 0.1
                coords = np.concatenate([coords, padding], axis=1)

            return coords
        except Exception:
            # Fallback to random coordinates
            return self.rng.randn(n, dim).astype(np.float32)

    def _bisect_partition(
        self,
        G: nx.Graph,
        parent: PartitionNode,
        coords: Optional[np.ndarray],
        level: int,
        start_id: int,
    ) -> list[PartitionNode]:
        """
        Bisect a partition into two children using coordinate-based or spectral methods.
        """
        nodes = parent.node_ids
        n = len(nodes)

        if n <= self.min_partition_size:
            return []

        # Get coordinates for this partition's nodes
        if coords is not None:
            node_to_idx = {node: i for i, node in enumerate(list(G.nodes()))}
            part_coords = np.array([coords[node_to_idx[n]] for n in nodes])
        else:
            part_coords = self._compute_spectral_coordinates(G, nodes)

        # Find the principal axis for bisection
        centroid = part_coords.mean(axis=0)
        centered = part_coords - centroid

        # Use SVD to find principal direction
        if centered.shape[0] > 1:
            try:
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
                principal_dir = Vt[0]
            except np.linalg.LinAlgError:
                # Fallback: use first coordinate axis
                principal_dir = np.zeros(centered.shape[1])
                principal_dir[0] = 1.0
        else:
            principal_dir = np.zeros(centered.shape[1])
            principal_dir[0] = 1.0

        # Project points onto principal axis
        projections = centered @ principal_dir

        # Find median split point
        median_idx = np.argsort(projections)[n // 2]
        split_value = projections[median_idx]

        # Assign nodes to partitions
        left_nodes = []
        right_nodes = []
        left_coords = []
        right_coords = []

        for i, (node, proj) in enumerate(zip(nodes, projections)):
            if proj <= split_value:
                left_nodes.append(node)
                left_coords.append(part_coords[i])
            else:
                right_nodes.append(node)
                right_coords.append(part_coords[i])

        # Ensure both partitions have nodes
        if len(left_nodes) == 0 or len(right_nodes) == 0:
            # Fallback: random split
            self.rng.shuffle(nodes)
            mid = n // 2
            left_nodes = nodes[:mid]
            right_nodes = nodes[mid:]
            left_coords = part_coords[:mid]
            right_coords = part_coords[mid:]

        children = []

        if len(left_nodes) > 0:
            left_centroid = np.mean(left_coords, axis=0) if left_coords else None
            children.append(PartitionNode(
                level=level,
                partition_id=start_id,
                node_ids=left_nodes,
                centroid=left_centroid,
            ))

        if len(right_nodes) > 0:
            right_centroid = np.mean(right_coords, axis=0) if right_coords else None
            children.append(PartitionNode(
                level=level,
                partition_id=start_id + 1 if len(left_nodes) > 0 else start_id,
                node_ids=right_nodes,
                centroid=right_centroid,
            ))

        return children

    def _build_ancestor_map(
        self, level_partitions: list[list[PartitionNode]]
    ) -> dict[int, list[tuple[int, int]]]:
        """Build mapping from each node to its ancestor partitions at each level."""
        node_to_ancestors = defaultdict(list)

        # For each level, map nodes to their partition
        for level, partitions in enumerate(level_partitions):
            for partition in partitions:
                for node_id in partition.node_ids:
                    node_to_ancestors[node_id].append((level, partition.partition_id))

        return dict(node_to_ancestors)


class MetisPartitioner(GraphPartitioner):
    """
    Graph partitioner using METIS library for higher quality partitions.
    Falls back to coordinate-based bisection if METIS is not available.
    """

    def __init__(
        self,
        num_levels: int = 4,
        min_partition_size: int = 10,
        balance_factor: float = 1.03,
        seed: int = 42,
    ):
        super().__init__(
            num_levels=num_levels,
            min_partition_size=min_partition_size,
            balance_factor=balance_factor,
            use_coordinates=True,
            seed=seed,
        )
        self._metis_available = self._check_metis()

    def _check_metis(self) -> bool:
        """Check if METIS/pymetis is available."""
        try:
            import pymetis
            return True
        except ImportError:
            return False

    def _bisect_partition(
        self,
        G: nx.Graph,
        parent: PartitionNode,
        coords: Optional[np.ndarray],
        level: int,
        start_id: int,
    ) -> list[PartitionNode]:
        """Bisect using METIS if available, otherwise use parent method."""
        if not self._metis_available or len(parent.node_ids) < 20:
            return super()._bisect_partition(G, parent, coords, level, start_id)

        try:
            import pymetis

            nodes = parent.node_ids
            subgraph = G.subgraph(nodes)

            # Build adjacency list for METIS
            node_to_local = {n: i for i, n in enumerate(nodes)}
            adjacency = []
            for node in nodes:
                neighbors = [node_to_local[n] for n in subgraph.neighbors(node) if n in node_to_local]
                adjacency.append(np.array(neighbors, dtype=np.int32))

            # Run METIS bisection
            n_cuts, membership = pymetis.part_graph(2, adjacency=adjacency)

            # Split nodes based on membership
            left_nodes = [nodes[i] for i, m in enumerate(membership) if m == 0]
            right_nodes = [nodes[i] for i, m in enumerate(membership) if m == 1]

            if len(left_nodes) == 0 or len(right_nodes) == 0:
                return super()._bisect_partition(G, parent, coords, level, start_id)

            children = []
            children.append(PartitionNode(
                level=level,
                partition_id=start_id,
                node_ids=left_nodes,
            ))
            children.append(PartitionNode(
                level=level,
                partition_id=start_id + 1,
                node_ids=right_nodes,
            ))

            return children

        except Exception:
            return super()._bisect_partition(G, parent, coords, level, start_id)


def create_partition_index(
    partition_tree: PartitionTree,
    embed_dim: int,
) -> dict:
    """
    Create index structures for efficient embedding lookup.

    Returns a dictionary containing:
    - level_embed_sizes: Number of partitions at each level
    - node_ancestor_indices: Tensor mapping node_id to ancestor indices
    - ancestor_levels: Tensor mapping node_id to ancestor levels
    """
    max_nodes = max(partition_tree.node_to_ancestors.keys()) + 1
    max_ancestors = partition_tree.num_levels

    # Create tensors for ancestor indices
    node_ancestor_indices = torch.zeros(max_nodes, max_ancestors, dtype=torch.long)
    ancestor_masks = torch.zeros(max_nodes, max_ancestors, dtype=torch.bool)

    for node_id, ancestors in partition_tree.node_to_ancestors.items():
        for i, (level, partition_id) in enumerate(ancestors):
            node_ancestor_indices[node_id, i] = partition_id
            ancestor_masks[node_id, i] = True

    return {
        'num_levels': partition_tree.num_levels,
        'level_sizes': partition_tree.level_sizes,
        'node_ancestor_indices': node_ancestor_indices,
        'ancestor_masks': ancestor_masks,
    }
