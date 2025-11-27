import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        use_softplus: bool = False,
    ):
        super().__init__()
        self.use_softplus = use_softplus
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x).squeeze(-1)
        if self.use_softplus:
            out = F.softplus(out)
        return out


class GraphSurrogate(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 3,
        expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.encoder = nn.Sequential(*[
            PreNormResidualBlock(hidden_dim, expansion=expansion, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.pre_latent_norm = nn.LayerNorm(hidden_dim)

        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        self.latent_norm = nn.LayerNorm(latent_dim)

        self.count_head = PredictionHead(
            latent_dim, hidden_dim, dropout=dropout, use_softplus=True
        )
        self.distance_head = PredictionHead(
            latent_dim, hidden_dim, dropout=dropout, use_softplus=True
        )

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'input_embed' in name or 'latent_proj' in name:
                    nn.init.xavier_uniform_(module.weight)
                else:
                    std = 0.02 / math.sqrt(2 * self.num_layers)
                    nn.init.normal_(module.weight, mean=0.0, std=std)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_embed(x)

        h = self.encoder(h)
        h = self.pre_latent_norm(h)

        z = self.latent_proj(h)
        z = self.latent_norm(z)

        count_pred = self.count_head(z)
        distance_pred = self.distance_head(z)

        return count_pred, distance_pred, z

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_embed(x)
        h = self.encoder(h)
        h = self.pre_latent_norm(h)
        z = self.latent_proj(h)
        return self.latent_norm(z)

    @torch.jit.export
    def predict_count(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_embed(x)
        h = self.encoder(h)
        h = self.pre_latent_norm(h)
        z = self.latent_proj(h)
        z = self.latent_norm(z)
        return self.count_head(z)

    @torch.jit.export
    def predict_distance(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_embed(x)
        h = self.encoder(h)
        h = self.pre_latent_norm(h)
        z = self.latent_proj(h)
        z = self.latent_norm(z)
        return self.distance_head(z)


class TwoHeadLoss(nn.Module):
    def __init__(
        self,
        count_delta: float = 10.0,
        distance_delta: float = 1.0,
        count_weight: float = 1.0,
        distance_weight: float = 1.0,
        learn_weights: bool = False,
    ):
        super().__init__()
        self.count_delta = count_delta
        self.distance_delta = distance_delta

        if learn_weights:
            self.log_var_count = nn.Parameter(torch.zeros(1))
            self.log_var_dist = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('log_var_count', torch.log(torch.tensor(count_weight)))
            self.register_buffer('log_var_dist', torch.log(torch.tensor(distance_weight)))

        self.learn_weights = learn_weights

    def forward(
        self,
        count_pred: torch.Tensor,
        count_target: torch.Tensor,
        distance_pred: torch.Tensor,
        distance_target: torch.Tensor,
        query_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count_mask = query_types == 0
        distance_mask = query_types == 1

        if count_mask.any():
            count_errors = F.smooth_l1_loss(
                count_pred[count_mask],
                count_target[count_mask],
                beta=self.count_delta,
                reduction='none'
            )
            if self.learn_weights:
                precision = torch.exp(-self.log_var_count)
                loss_count = precision * count_errors.mean() + self.log_var_count
            else:
                loss_count = count_errors.mean() * torch.exp(self.log_var_count)
        else:
            loss_count = torch.tensor(0.0, device=count_pred.device)

        if distance_mask.any():
            dist_errors = F.smooth_l1_loss(
                distance_pred[distance_mask],
                distance_target[distance_mask],
                beta=self.distance_delta,
                reduction='none'
            )
            if self.learn_weights:
                precision = torch.exp(-self.log_var_dist)
                loss_distance = precision * dist_errors.mean() + self.log_var_dist
            else:
                loss_distance = dist_errors.mean() * torch.exp(self.log_var_dist)
        else:
            loss_distance = torch.tensor(0.0, device=distance_pred.device)

        total_loss = loss_count + loss_distance

        return total_loss, loss_count, loss_distance
