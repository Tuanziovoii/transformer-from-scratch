import math

import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(q.size(-1))

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        output = weights @ v

        return output, weights