"""
Hierarchical Road Network Embedding (RNE) for Shortest-Path Distance Approximation.

This module implements the hierarchical embedding approach where each node's global
embedding is the sum of its ancestors' local embeddings in the partition tree.
The L1 distance metric is used for path linearity compatibility.

Key components:
1. Local embedding matrices at each hierarchy level
2. Ancestral embedding summation: v(i) = sum_{v_l,j in anc(v_i)} M_l[j]
3. L1 distance metric for approximation
4. Active fine-tuning with distance bucketing

Reference: Hierarchical RNE for road network distance approximation.
"""

import math
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .partitioning import PartitionTree, GraphPartitioner, create_partition_index


@dataclass
class HierarchicalRNEConfig:
    """Configuration for Hierarchical RNE model."""
    embed_dim: int = 128  # Dimension of node embeddings
    num_levels: int = 4   # Number of hierarchy levels
    level_dims: Optional[list[int]] = None  # Optional per-level dimensions
    distance_metric: Literal['l1', 'l2'] = 'l1'  # L1 recommended for path linearity
    use_level_weights: bool = True  # Learnable weights per level
    use_bias: bool = True  # Include bias terms
    dropout: float = 0.1
    max_nodes: int = 100000

    def __post_init__(self):
        if self.level_dims is None:
            self.level_dims = [self.embed_dim] * self.num_levels


class LevelEmbedding(nn.Module):
    """
    Local embedding matrix for a single hierarchy level.

    Each partition at this level has its own embedding vector.
    The embedding is learned during training and summed with
    ancestor embeddings to form the global node embedding.
    """

    def __init__(
        self,
        num_partitions: int,
        embed_dim: int,
        use_bias: bool = True,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.num_partitions = num_partitions
        self.embed_dim = embed_dim

        # Local embedding matrix M_l
        self.embedding = nn.Embedding(num_partitions, embed_dim)

        # Optional bias
        self.bias = nn.Parameter(torch.zeros(embed_dim)) if use_bias else None

        # Initialize
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_scale)

    def forward(self, partition_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for partition IDs.

        Args:
            partition_ids: (batch_size,) tensor of partition indices

        Returns:
            (batch_size, embed_dim) tensor of embeddings
        """
        emb = self.embedding(partition_ids)
        if self.bias is not None:
            emb = emb + self.bias
        return emb


class HierarchicalEmbedding(nn.Module):
    """
    Hierarchical node embedding using partition tree structure.

    The global embedding for node v_i is computed as:
        v(i) = sum_{(l, j) in ancestors(v_i)} M_l[j]

    where M_l is the local embedding matrix at level l and j is the
    partition ID within that level.
    """

    def __init__(
        self,
        config: HierarchicalRNEConfig,
        partition_tree: PartitionTree,
    ):
        super().__init__()
        self.config = config
        self.num_levels = partition_tree.num_levels

        # Create partition index for efficient lookup
        self.partition_index = create_partition_index(partition_tree, config.embed_dim)

        # Register buffers for ancestor indices
        self.register_buffer(
            'node_ancestor_indices',
            self.partition_index['node_ancestor_indices']
        )
        self.register_buffer(
            'ancestor_masks',
            self.partition_index['ancestor_masks']
        )

        # Create level embedding modules
        level_sizes = self.partition_index['level_sizes']
        self.level_embeddings = nn.ModuleList([
            LevelEmbedding(
                num_partitions=max(size, 1),
                embed_dim=config.level_dims[min(i, len(config.level_dims) - 1)],
                use_bias=config.use_bias,
            )
            for i, size in enumerate(level_sizes)
        ])

        # Learnable level weights (importance of each level)
        if config.use_level_weights:
            self.level_weights = nn.Parameter(torch.ones(self.num_levels))
        else:
            self.register_buffer('level_weights', torch.ones(self.num_levels))

        # Optional projection if level dims vary
        if len(set(config.level_dims)) > 1:
            self.level_projections = nn.ModuleList([
                nn.Linear(dim, config.embed_dim) if dim != config.embed_dim else nn.Identity()
                for dim in config.level_dims[:self.num_levels]
            ])
        else:
            self.level_projections = None

        # Final normalization
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute global embeddings for nodes by summing ancestral embeddings.

        Args:
            node_ids: (batch_size,) tensor of node indices

        Returns:
            (batch_size, embed_dim) tensor of global embeddings
        """
        batch_size = node_ids.shape[0]
        device = node_ids.device

        # Get ancestor indices for all nodes in batch
        ancestor_indices = self.node_ancestor_indices[node_ids]  # (batch, num_levels)
        masks = self.ancestor_masks[node_ids]  # (batch, num_levels)

        # Compute weighted sum of ancestral embeddings
        global_embedding = torch.zeros(batch_size, self.config.embed_dim, device=device)

        for level in range(self.num_levels):
            if level < len(self.level_embeddings):
                # Get partition IDs at this level
                partition_ids = ancestor_indices[:, level]
                level_mask = masks[:, level].float().unsqueeze(-1)

                # Get level embedding
                level_emb = self.level_embeddings[level](partition_ids)

                # Project if needed
                if self.level_projections is not None:
                    level_emb = self.level_projections[level](level_emb)

                # Weight and accumulate
                weight = F.softplus(self.level_weights[level])
                global_embedding = global_embedding + weight * level_emb * level_mask

        # Normalize and dropout
        global_embedding = self.norm(global_embedding)
        global_embedding = self.dropout(global_embedding)

        return global_embedding

    def get_level_embedding(self, node_ids: torch.Tensor, level: int) -> torch.Tensor:
        """Get embedding from a specific level only."""
        ancestor_indices = self.node_ancestor_indices[node_ids]
        partition_ids = ancestor_indices[:, level]
        return self.level_embeddings[level](partition_ids)


class HierarchicalRNE(nn.Module):
    """
    Hierarchical Road Network Embedding model for shortest-path distance approximation.

    Architecture:
    1. Hierarchical embedding layer computes global node embeddings
    2. L1 distance metric between source and target embeddings
    3. Optional MLP refinement for improved accuracy

    The model approximates shortest-path distance as:
        d_approx(s, t) = ||v(s) - v(t)||_1

    where v(s) and v(t) are the global embeddings of source and target nodes.
    """

    def __init__(
        self,
        config: HierarchicalRNEConfig,
        partition_tree: PartitionTree,
        use_mlp_refinement: bool = True,
        mlp_hidden_dim: int = 256,
    ):
        super().__init__()
        self.config = config
        self.use_mlp_refinement = use_mlp_refinement

        # Hierarchical embedding
        self.embedding = HierarchicalEmbedding(config, partition_tree)

        # Optional MLP for distance refinement
        if use_mlp_refinement:
            self.distance_mlp = nn.Sequential(
                nn.Linear(config.embed_dim * 2 + 1, mlp_hidden_dim),
                nn.LayerNorm(mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(mlp_hidden_dim // 2, 1),
                nn.Softplus(),  # Ensure non-negative distances
            )
        else:
            self.distance_mlp = None

        # Learnable scaling factor for L1 distance
        self.distance_scale = nn.Parameter(torch.ones(1))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training stability."""
        if self.distance_mlp is not None:
            for module in self.distance_mlp.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def compute_embedding(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Compute global embeddings for nodes."""
        return self.embedding(node_ids)

    def compute_distance(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
        source_emb: Optional[torch.Tensor] = None,
        target_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute approximate shortest-path distances.

        Args:
            source_ids: (batch_size,) source node IDs
            target_ids: (batch_size,) target node IDs
            source_emb: Optional pre-computed source embeddings
            target_emb: Optional pre-computed target embeddings

        Returns:
            (batch_size,) tensor of approximate distances
        """
        # Get embeddings if not provided
        if source_emb is None:
            source_emb = self.embedding(source_ids)
        if target_emb is None:
            target_emb = self.embedding(target_ids)

        # Compute L1 distance (or L2 based on config)
        diff = source_emb - target_emb

        if self.config.distance_metric == 'l1':
            raw_distance = torch.abs(diff).sum(dim=-1)
        else:  # l2
            raw_distance = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)

        # Scale the raw distance
        scaled_distance = raw_distance * F.softplus(self.distance_scale)

        # Optional MLP refinement
        if self.use_mlp_refinement and self.distance_mlp is not None:
            # Concatenate embeddings and raw distance for refinement
            mlp_input = torch.cat([
                source_emb,
                target_emb,
                scaled_distance.unsqueeze(-1),
            ], dim=-1)
            refined_distance = self.distance_mlp(mlp_input).squeeze(-1)
            return refined_distance

        return scaled_distance

    def forward(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass computing distances and embeddings.

        Args:
            source_ids: (batch_size,) source node IDs
            target_ids: (batch_size,) target node IDs

        Returns:
            distances: (batch_size,) approximate distances
            source_emb: (batch_size, embed_dim) source embeddings
            target_emb: (batch_size, embed_dim) target embeddings
        """
        source_emb = self.embedding(source_ids)
        target_emb = self.embedding(target_ids)
        distances = self.compute_distance(
            source_ids, target_ids, source_emb, target_emb
        )
        return distances, source_emb, target_emb

    def batch_query(
        self,
        source_id: int,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Efficient batch query from single source to multiple targets.

        Args:
            source_id: Single source node ID
            target_ids: (num_targets,) target node IDs

        Returns:
            (num_targets,) approximate distances
        """
        device = target_ids.device

        # Compute source embedding once
        source_ids = torch.tensor([source_id], device=device)
        source_emb = self.embedding(source_ids)  # (1, embed_dim)

        # Compute target embeddings
        target_emb = self.embedding(target_ids)  # (num_targets, embed_dim)

        # Broadcast source embedding
        source_emb_expanded = source_emb.expand(len(target_ids), -1)

        return self.compute_distance(
            source_ids.expand(len(target_ids)),
            target_ids,
            source_emb_expanded,
            target_emb,
        )


class RNELoss(nn.Module):
    """
    Loss function for Hierarchical RNE training.

    Combines:
    1. Distance regression loss (Smooth L1)
    2. Optional embedding regularization (norm penalty)
    3. Optional contrastive loss for embedding quality
    """

    def __init__(
        self,
        beta: float = 1.0,
        use_relative_error: bool = True,
        embed_regularization: float = 0.01,
        use_contrastive: bool = False,
        contrastive_weight: float = 0.1,
    ):
        super().__init__()
        self.beta = beta
        self.use_relative_error = use_relative_error
        self.embed_regularization = embed_regularization
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight

    def forward(
        self,
        pred_distances: torch.Tensor,
        true_distances: torch.Tensor,
        source_emb: Optional[torch.Tensor] = None,
        target_emb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss.

        Args:
            pred_distances: (batch_size,) predicted distances
            true_distances: (batch_size,) ground truth distances
            source_emb: Optional source embeddings for regularization
            target_emb: Optional target embeddings for regularization

        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}

        # Distance loss
        if self.use_relative_error:
            # Relative error: |pred - true| / (true + epsilon)
            epsilon = 1.0
            relative_error = torch.abs(pred_distances - true_distances) / (true_distances + epsilon)
            distance_loss = F.smooth_l1_loss(relative_error, torch.zeros_like(relative_error), beta=self.beta)
        else:
            distance_loss = F.smooth_l1_loss(pred_distances, true_distances, beta=self.beta)

        loss_dict['distance_loss'] = distance_loss.item()
        total_loss = distance_loss

        # Embedding regularization
        if self.embed_regularization > 0 and source_emb is not None and target_emb is not None:
            # L2 regularization on embedding norms
            source_norm = torch.norm(source_emb, p=2, dim=-1).mean()
            target_norm = torch.norm(target_emb, p=2, dim=-1).mean()
            reg_loss = self.embed_regularization * (source_norm + target_norm)
            loss_dict['reg_loss'] = reg_loss.item()
            total_loss = total_loss + reg_loss

        # Contrastive loss for embedding quality
        if self.use_contrastive and source_emb is not None and target_emb is not None:
            # Pairs with larger true distance should have larger embedding distance
            batch_size = len(true_distances)
            if batch_size > 1:
                # Sample random pairs within batch
                idx1 = torch.randperm(batch_size, device=pred_distances.device)[:batch_size // 2]
                idx2 = torch.randperm(batch_size, device=pred_distances.device)[:batch_size // 2]

                # Get distances for both pairs
                dist1 = true_distances[idx1]
                dist2 = true_distances[idx2]
                pred1 = pred_distances[idx1]
                pred2 = pred_distances[idx2]

                # Contrastive: if dist1 > dist2, then pred1 should be > pred2
                margin = 0.5
                should_be_larger = (dist1 > dist2 + margin).float()
                ordering_loss = should_be_larger * F.relu(pred2 - pred1 + margin)
                ordering_loss = ordering_loss + (1 - should_be_larger) * F.relu(pred1 - pred2 + margin)

                contrastive_loss = self.contrastive_weight * ordering_loss.mean()
                loss_dict['contrastive_loss'] = contrastive_loss.item()
                total_loss = total_loss + contrastive_loss

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict


def compute_mean_relative_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1.0,
) -> float:
    """Compute Mean Relative Error (MRE)."""
    with torch.no_grad():
        relative_error = torch.abs(pred - target) / (target + epsilon)
        return relative_error.mean().item()


def compute_absolute_error_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """Compute Absolute Error statistics."""
    with torch.no_grad():
        abs_error = torch.abs(pred - target)
        return {
            'mae': abs_error.mean().item(),
            'median_ae': abs_error.median().item(),
            'max_ae': abs_error.max().item(),
            'p95_ae': torch.quantile(abs_error, 0.95).item(),
            'p99_ae': torch.quantile(abs_error, 0.99).item(),
        }


def compute_error_distribution(
    pred: torch.Tensor,
    target: torch.Tensor,
    thresholds: list[float] = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20],
) -> dict:
    """Compute CDF of relative errors at various thresholds."""
    with torch.no_grad():
        relative_error = torch.abs(pred - target) / (target + 1.0)
        result = {}
        for thresh in thresholds:
            pct = (relative_error <= thresh).float().mean().item() * 100
            result[f'within_{int(thresh*100)}pct'] = pct
        return result
