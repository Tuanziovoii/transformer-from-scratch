import torch
from torch import nn

from attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model 和 num_heads 必须为正数")
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # [batch, length, d_model]
        # → [batch, length, heads, head_dim]
        # → [batch, heads, length, head_dim]
        return x.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.split_heads(self.q_proj(query))
        k = self.split_heads(self.k_proj(key))
        v = self.split_heads(self.v_proj(value))

        output, weights = self.attention(q, k, v, mask=mask)

        batch_size = output.size(0)
        query_len = output.size(2)

        # 把各个头放回同一个 token 下，再拼接。
        output = output.transpose(1, 2).contiguous()
        output = output.view(
            batch_size,
            query_len,
            self.num_heads * self.head_dim,
        )

        return self.out_proj(output), weights