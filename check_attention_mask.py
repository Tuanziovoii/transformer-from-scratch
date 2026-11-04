import torch

from attention import ScaledDotProductAttention


torch.manual_seed(42)
attention = ScaledDotProductAttention()

# [batch_size, heads, seq_len, head_dim]
q = torch.randn(1, 1, 4, 8)
k = torch.randn(1, 1, 4, 8)
v = torch.randn(1, 1, 4, 8)

# 下三角为 True：允许看自己和过去。
mask = torch.ones(4, 4, dtype=torch.bool).tril()

output, weights = attention(q, k, v, mask=mask)

assert torch.isfinite(output).all().item()

# 所有未来位置的权重必须为 0。
future_weights = weights[0, 0][~mask]
torch.testing.assert_close(
    future_weights,
    torch.zeros_like(future_weights),
)

# 屏蔽后，每行权重仍应归一化。
torch.testing.assert_close(
    weights.sum(dim=-1),
    torch.ones(1, 1, 4),
)

# 第 0 个位置只能关注自己，因此输出等于第 0 个 V。
torch.testing.assert_close(output[0, 0, 0], v[0, 0, 0])

# 修改最后一个位置的 K、V。
# 前三个位置看不到它，所以它们的输出不应改变。
changed_k = k.clone()
changed_v = v.clone()
changed_k[:, :, 3, :] += 100.0
changed_v[:, :, 3, :] += 100.0

changed_output, _ = attention(
    q, changed_k, changed_v, mask=mask
)

torch.testing.assert_close(
    output[:, :, :3, :],
    changed_output[:, :, :3, :],
)

print("允许关注的位置:")
print(mask.int())
print("注意力权重:")
print(weights[0, 0])
print("全部检查通过")