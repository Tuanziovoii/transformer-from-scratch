import torch
from torch import nn

from feed_forward import PositionwiseFeedForward
from multi_head_attention import MultiHeadAttention


class DecoderLayer(nn.Module):
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
        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_output, _ = self.self_attention(
            x, x, x, mask=target_mask
        )
        x = self.norm1(x + self.dropout1(self_output))

        cross_output, _ = self.cross_attention(
            x, memory, memory, mask=memory_mask
        )
        x = self.norm2(x + self.dropout2(cross_output))

        ffn_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ffn_output))

        return x