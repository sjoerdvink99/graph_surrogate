import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mdn import MixtureDensityHead


@dataclass
class ModelConfig:
    embed_dim: int = 64
    hidden_dim: int = 256
    latent_dim: int = 64
    num_layers: int = 6
    expansion: int = 4
    dropout: float = 0.1
    num_node_types: int = 10
    num_degree_bins: int = 11
    num_radii: int = 3
    num_attr_names: int = 5
    num_attr_values: int = 50
    num_max_hops: int = 4
    num_structural_features: int = 8
    use_log_transform: bool = True
    use_mdn: bool = False
    num_mixtures: int = 5
    use_structural_features: bool = True

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PreNormResidualBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class PredictionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        output_activation: str = "none",
    ):
        super().__init__()
        self.output_activation = output_activation
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x).squeeze(-1)
        if self.output_activation == "softplus":
            out = F.softplus(out)
        elif self.output_activation == "exp":
            out = torch.exp(torch.clamp(out, max=10))
        return out


class GraphSurrogate(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.node_type_embed = nn.Embedding(config.num_node_types + 1, config.embed_dim)
        self.degree_bin_embed = nn.Embedding(config.num_degree_bins + 1, config.embed_dim)
        self.radius_embed = nn.Embedding(config.num_radii + 1, config.embed_dim)
        self.attr_name_embed = nn.Embedding(config.num_attr_names + 2, config.embed_dim)
        self.attr_value_embed = nn.Embedding(config.num_attr_values + 2, config.embed_dim)
        self.target_type_embed = nn.Embedding(config.num_node_types + 1, config.embed_dim)
        self.max_hops_embed = nn.Embedding(config.num_max_hops + 1, config.embed_dim)
        self.query_type_embed = nn.Embedding(2, config.embed_dim)

        if config.use_structural_features:
            self.structural_proj = nn.Sequential(
                nn.Linear(config.num_structural_features, config.embed_dim),
                nn.LayerNorm(config.embed_dim),
                nn.GELU(),
            )

        num_embeddings = 8
        if config.use_structural_features:
            num_embeddings += 1
        encoder_input_dim = config.embed_dim * num_embeddings

        self.input_proj = nn.Sequential(
            nn.Linear(encoder_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.encoder = nn.ModuleList([
            PreNormResidualBlock(config.hidden_dim, config.expansion, config.dropout)
            for _ in range(config.num_layers)
        ])

        self.pre_latent_norm = nn.LayerNorm(config.hidden_dim)
        self.latent_proj = nn.Linear(config.hidden_dim, config.latent_dim)
        self.latent_norm = nn.LayerNorm(config.latent_dim)

        if config.use_mdn:
            self.count_head = MixtureDensityHead(
                config.latent_dim, config.num_mixtures, config.hidden_dim // 2
            )
            self.distance_head = MixtureDensityHead(
                config.latent_dim, config.num_mixtures, config.hidden_dim // 2
            )
        else:
            count_activation = "none" if config.use_log_transform else "softplus"
            self.count_head = PredictionHead(
                config.latent_dim, config.hidden_dim, config.dropout, count_activation
            )
            self.distance_head = PredictionHead(
                config.latent_dim, config.hidden_dim, config.dropout, "softplus"
            )

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'input_proj' in name or 'latent_proj' in name:
                    nn.init.xavier_uniform_(module.weight)
                else:
                    std = 0.02 / math.sqrt(2 * self.config.num_layers)
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode_query(
        self,
        node_type_idx: torch.Tensor,
        degree_bin_idx: torch.Tensor,
        radius_idx: torch.Tensor,
        attr_name_idx: torch.Tensor,
        attr_value_idx: torch.Tensor,
        target_type_idx: torch.Tensor,
        max_hops_idx: torch.Tensor,
        query_type_idx: torch.Tensor,
        structural_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embeddings = [
            self.node_type_embed(node_type_idx),
            self.degree_bin_embed(degree_bin_idx),
            self.radius_embed(radius_idx),
            self.attr_name_embed(attr_name_idx),
            self.attr_value_embed(attr_value_idx),
            self.target_type_embed(target_type_idx),
            self.max_hops_embed(max_hops_idx),
            self.query_type_embed(query_type_idx),
        ]

        if self.config.use_structural_features and structural_features is not None:
            embeddings.append(self.structural_proj(structural_features))

        return torch.cat(embeddings, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        structural_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        for block in self.encoder:
            h = block(h)

        h = self.pre_latent_norm(h)
        z = self.latent_proj(h)
        z = self.latent_norm(z)

        if self.config.use_mdn:
            count_pred = self.count_head.predict(z)
            distance_pred = self.distance_head.predict(z)
        else:
            count_pred = self.count_head(z)
            distance_pred = self.distance_head(z)

        return count_pred, distance_pred, z

    def predict_count(
        self,
        x: torch.Tensor,
        structural_features: Optional[torch.Tensor] = None,
        return_original_scale: bool = True,
    ) -> torch.Tensor:
        count_pred, _, _ = self.forward(x, structural_features)

        if self.config.use_log_transform and return_original_scale:
            count_pred = torch.expm1(count_pred)
            count_pred = torch.clamp(count_pred, min=0)

        return count_pred

    def predict_distance(
        self,
        x: torch.Tensor,
        structural_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, distance_pred, _ = self.forward(x, structural_features)
        return distance_pred

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        structural_features: Optional[torch.Tensor] = None,
        num_samples: int = 30,
    ) -> dict[str, torch.Tensor]:
        was_training = self.training
        self.train()

        count_samples = []
        dist_samples = []

        with torch.no_grad():
            for _ in range(num_samples):
                count_pred, dist_pred, _ = self.forward(x, structural_features)
                count_samples.append(count_pred)
                dist_samples.append(dist_pred)

        if not was_training:
            self.eval()

        count_samples = torch.stack(count_samples)
        dist_samples = torch.stack(dist_samples)

        result = {
            'count_mean': count_samples.mean(dim=0),
            'count_std': count_samples.std(dim=0),
            'distance_mean': dist_samples.mean(dim=0),
            'distance_std': dist_samples.std(dim=0),
        }

        if self.config.use_log_transform:
            result['count_mean_original'] = torch.clamp(torch.expm1(result['count_mean']), min=0)
            result['count_std_original'] = result['count_std'] * result['count_mean_original']

        return result

    def get_latent(
        self,
        x: torch.Tensor,
        structural_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, _, z = self.forward(x, structural_features)
        return z


class ImprovedLoss(nn.Module):
    def __init__(
        self,
        use_log_transform: bool = True,
        count_beta: float = 0.5,
        distance_beta: float = 1.0,
        count_weight: float = 1.0,
        distance_weight: float = 1.0,
        use_mdn: bool = False,
    ):
        super().__init__()
        self.use_log_transform = use_log_transform
        self.count_beta = count_beta
        self.distance_beta = distance_beta
        self.count_weight = count_weight
        self.distance_weight = distance_weight
        self.use_mdn = use_mdn

    def forward(
        self,
        count_pred: torch.Tensor,
        count_target: torch.Tensor,
        distance_pred: torch.Tensor,
        distance_target: torch.Tensor,
        query_types: torch.Tensor,
        count_head: Optional[nn.Module] = None,
        distance_head: Optional[nn.Module] = None,
        latent: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count_mask = query_types == 0
        distance_mask = query_types == 1

        if count_mask.any():
            if self.use_mdn and count_head is not None and latent is not None:
                count_loss = count_head.nll_loss(latent[count_mask], count_target[count_mask])
            else:
                if self.use_log_transform:
                    log_target = torch.log1p(count_target[count_mask])
                    count_loss = F.smooth_l1_loss(
                        count_pred[count_mask], log_target, beta=self.count_beta
                    )
                else:
                    count_loss = F.smooth_l1_loss(
                        count_pred[count_mask], count_target[count_mask], beta=10.0
                    )
            count_loss = count_loss * self.count_weight
        else:
            count_loss = torch.tensor(0.0, device=count_pred.device)

        if distance_mask.any():
            if self.use_mdn and distance_head is not None and latent is not None:
                distance_loss = distance_head.nll_loss(latent[distance_mask], distance_target[distance_mask])
            else:
                distance_loss = F.smooth_l1_loss(
                    distance_pred[distance_mask], distance_target[distance_mask], beta=self.distance_beta
                )
            distance_loss = distance_loss * self.distance_weight
        else:
            distance_loss = torch.tensor(0.0, device=distance_pred.device)

        total_loss = count_loss + distance_loss
        return total_loss, count_loss, distance_loss


def create_model_from_encoder_config(encoder_config, model_config: Optional[ModelConfig] = None) -> GraphSurrogate:
    if model_config is None:
        model_config = ModelConfig()

    model_config.num_node_types = len(encoder_config.node_types)
    model_config.num_degree_bins = len(encoder_config.degree_bins)
    model_config.num_radii = len(encoder_config.radii)
    model_config.num_attr_names = len(encoder_config.attribute_names)
    model_config.num_attr_values = sum(len(v) for v in encoder_config.attribute_values.values())
    model_config.num_max_hops = len(encoder_config.max_hops_options)

    return GraphSurrogate(model_config)
