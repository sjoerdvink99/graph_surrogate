import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureDensityHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_mixtures: int = 5,
        hidden_dim: Optional[int] = None,
        min_sigma: float = 1e-4,
        max_sigma: float = 100.0,
    ):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        hidden_dim = hidden_dim or input_dim

        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )

        self.pi_head = nn.Linear(hidden_dim, num_mixtures)
        self.mu_head = nn.Linear(hidden_dim, num_mixtures)
        self.sigma_head = nn.Linear(hidden_dim, num_mixtures)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.xavier_uniform_(self.sigma_head.weight)
        nn.init.zeros_(self.sigma_head.bias)
        nn.init.zeros_(self.pi_head.weight)
        nn.init.zeros_(self.pi_head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        pi = F.softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h)
        sigma_raw = self.sigma_head(h)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * torch.sigmoid(sigma_raw)
        return pi, mu, sigma

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pi, mu, _ = self.forward(x)
        return (pi * mu).sum(dim=-1)

    def predict_with_uncertainty(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pi, mu, sigma = self.forward(x)
        mean = (pi * mu).sum(dim=-1)
        second_moment = (pi * (sigma ** 2 + mu ** 2)).sum(dim=-1)
        variance = second_moment - mean ** 2
        std = torch.sqrt(torch.clamp(variance, min=1e-8))
        return mean, std

    def sample(self, x: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
        pi, mu, sigma = self.forward(x)
        component_indices = torch.multinomial(pi, num_samples, replacement=True)
        mu_selected = mu.gather(1, component_indices)
        sigma_selected = sigma.gather(1, component_indices)
        samples = torch.normal(mu_selected, sigma_selected)
        return samples

    def nll_loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pi, mu, sigma = self.forward(x)
        target = target.unsqueeze(-1)
        log_component_prob = (
            -0.5 * math.log(2 * math.pi)
            - torch.log(sigma)
            - 0.5 * ((target - mu) / sigma) ** 2
        )
        log_mix_prob = torch.logsumexp(torch.log(pi + 1e-8) + log_component_prob, dim=-1)
        return -log_mix_prob.mean()
