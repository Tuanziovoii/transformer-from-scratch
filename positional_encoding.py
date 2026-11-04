import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        if d_model <= 0 or d_model % 2 != 0:
            raise ValueError("当前实现要求 d_model 是正偶数")

        self.dropout = nn.Dropout(dropout)

        # [max_len, 1]：每一行对应一个位置。
        position = torch.arange(
            max_len, dtype=torch.float32
        ).unsqueeze(1)

        # 对应公式中的 1 / 10000^(2i / d_model)。
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [1, max_len, d_model]，便于加到整个 batch 上。
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 的形状：[batch_size, seq_len, d_model]
        seq_len = x.size(1)

        if seq_len > self.pe.size(1):
            raise ValueError("序列长度超过了 max_len")

        position_encoding = self.pe[:, :seq_len].to(dtype=x.dtype)
        return self.dropout(x + position_encoding)