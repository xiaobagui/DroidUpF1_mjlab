"""Least-squares AMP discriminator."""

from __future__ import annotations

import torch
from torch import autograd, nn


class Discriminator(nn.Module):
  def __init__(self, state_dim: int, hidden_dims: tuple[int, ...]):
    super().__init__()
    layers: list[nn.Module] = []
    width = 2 * state_dim
    for hidden in hidden_dims:
      layers.extend((nn.Linear(width, hidden), nn.ReLU()))
      width = hidden
    self.trunk = nn.Sequential(*layers)
    self.head = nn.Linear(width, 1)

  def forward(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    return self.head(self.trunk(torch.cat((state, next_state), dim=1)))

  def style_reward(
    self,
    state: torch.Tensor,
    next_state: torch.Tensor,
    coefficient: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = self(state, next_state).squeeze(1)
    reward = coefficient * torch.clamp(
      1.0 - 0.25 * torch.square(prediction - 1.0), min=0.0
    )
    return reward, prediction

  def gradient_penalty(
    self,
    expert_state: torch.Tensor,
    expert_next_state: torch.Tensor,
    coefficient: float = 10.0,
  ) -> torch.Tensor:
    state = expert_state.detach().requires_grad_(True)
    next_state = expert_next_state.detach().requires_grad_(True)
    prediction = self(state, next_state)
    gradient = autograd.grad(
      prediction,
      (state, next_state),
      grad_outputs=torch.ones_like(prediction),
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )
    joined = torch.cat(gradient, dim=1)
    return coefficient * joined.norm(2, dim=1).square().mean()

