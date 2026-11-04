import math

import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(q.size(-1))

        weights = torch.softmax(scores, dim=-1)
        output = weights @ v

        return output, weights