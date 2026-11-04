import torch

from attention import ScaledDotProductAttention


attention = ScaledDotProductAttention()

# 先验证刚才手算过的例子。
q = torch.tensor([[1.0, 0.0]])
k = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
])
v = torch.tensor([
    [10.0, 0.0],
    [0.0, 20.0],
])

output, weights = attention(q, k, v)

torch.testing.assert_close(
    weights,
    torch.tensor([[0.66976154, 0.33023846]]),
)
torch.testing.assert_close(
    output,
    torch.tensor([[6.6976154, 6.6047692]]),
)

# 再验证多个 batch、多个注意力头的计算。
torch.manual_seed(42)

q = torch.randn(2, 3, 4, 8, requires_grad=True)
k = torch.randn(2, 3, 5, 8, requires_grad=True)
v = torch.randn(2, 3, 5, 6, requires_grad=True)

output, weights = attention(q, k, v)

assert weights.shape == (2, 3, 4, 5)
assert output.shape == (2, 3, 4, 6)

# 每个 query 对全部 key 的权重之和为 1。
torch.testing.assert_close(
    weights.sum(dim=-1),
    torch.ones(2, 3, 4),
)
assert (weights >= 0).all().item()

# 检查梯度可以传回 Q、K、V。
output.square().mean().backward()

for tensor in (q, k, v):
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all().item()
    assert tensor.grad.abs().sum().item() > 0

print("注意力权重形状:", weights.shape)
print("输出形状:", output.shape)
print("全部检查通过")