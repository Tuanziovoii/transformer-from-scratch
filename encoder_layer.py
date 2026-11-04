import torch
from torch import nn

from feed_forward import PositionwiseFeedForward
from multi_head_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
        )

        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_output, _ = self.self_attention(
            x, x, x, mask=mask
        )
        x = self.norm1(
            x + self.attention_dropout(attention_output)
        )

        ffn_output = self.feed_forward(x)
        x = self.norm2(x + self.ffn_dropout(ffn_output))

        return x