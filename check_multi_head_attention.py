import torch
from torch import nn

from multi_head_attention import MultiHeadAttention


torch.manual_seed(42)

model = MultiHeadAttention(d_model=8, num_heads=2)
x = torch.randn(2, 4, 8)
mask = torch.ones(4, 4, dtype=torch.bool).tril()

output, weights = model(x, x, x, mask=mask)

assert output.shape == (2, 4, 8)
assert weights.shape == (2, 2, 4, 4)

# 每个头都必须屏蔽未来位置。
future_weights = weights.masked_select(
    (~mask)[None, None, :, :]
)
torch.testing.assert_close(
    future_weights,
    torch.zeros_like(future_weights),
)

# 与 PyTorch 自带的多头注意力对照。
# 它只用于检查，我们的模型仍使用自己写的模块。
reference = nn.MultiheadAttention(
    embed_dim=8,
    num_heads=2,
    dropout=0.0,
    batch_first=True,
)

with torch.no_grad():
    reference.in_proj_weight.copy_(torch.cat([
        model.q_proj.weight,
        model.k_proj.weight,
        model.v_proj.weight,
    ]))
    reference.in_proj_bias.copy_(torch.cat([
        model.q_proj.bias,
        model.k_proj.bias,
        model.v_proj.bias,
    ]))
    reference.out_proj.weight.copy_(model.out_proj.weight)
    reference.out_proj.bias.copy_(model.out_proj.bias)

# 注意：PyTorch 这个接口用 True 表示屏蔽，
# 与我们约定的 True 表示允许相反，所以传入 ~mask。
expected_output, expected_weights = reference(
    x, x, x,
    attn_mask=~mask,
    average_attn_weights=False,
)

torch.testing.assert_close(output, expected_output)
torch.testing.assert_close(weights, expected_weights)

# 查询长度与 key/value 长度不同，也应正常工作。
query = torch.randn(2, 3, 8)
memory = torch.randn(2, 5, 8)
cross_output, cross_weights = model(query, memory, memory)

assert cross_output.shape == (2, 3, 8)
assert cross_weights.shape == (2, 2, 3, 5)

# 所有投影层都应该能收到梯度。
output.square().mean().backward()

for name, parameter in model.named_parameters():
    assert parameter.grad is not None, name
    assert torch.isfinite(parameter.grad).all().item(), name

print("自注意力输出形状:", output.shape)
print("各头权重形状:", weights.shape)
print("与 PyTorch 对照一致")
print("全部检查通过")