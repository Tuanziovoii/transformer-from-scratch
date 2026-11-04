import torch

from encoder_layer import EncoderLayer


torch.manual_seed(42)

model = EncoderLayer(
    d_model=8,
    num_heads=2,
    d_ff=32,
    dropout=0.1,
)

# 验证时关闭 dropout，使结果可重复比较。
model.eval()

x = torch.randn(2, 4, 8)
output = model(x)

assert output.shape == x.shape
assert torch.isfinite(output).all().item()

# 手动按 Post-LayerNorm 顺序计算，对照 forward。
with torch.no_grad():
    attention_output, _ = model.self_attention(x, x, x)
    after_attention = model.norm1(x + attention_output)
    expected = model.norm2(
        after_attention + model.feed_forward(after_attention)
    )

torch.testing.assert_close(output, expected)

# 模拟最后一个位置是 padding：
# 每个 query 都只能关注前 3 个 key。
mask = torch.tensor(
    [[[[True, True, True, False]]]]
)

masked_output = model(x, mask=mask)

# 改变被屏蔽位置的输入，不应影响前三个位置。
changed_x = x.clone()
changed_x[:, 3, :] += 100.0
changed_output = model(changed_x, mask=mask)

torch.testing.assert_close(
    masked_output[:, :3, :],
    changed_output[:, :3, :],
)

# eval() 关闭 dropout，但不关闭梯度计算。
# 使用非均匀目标检查梯度。
target = torch.randn_like(output)
loss = (output - target).square().mean()
loss.backward()

for name, parameter in model.named_parameters():
    assert parameter.grad is not None, name
    assert torch.isfinite(parameter.grad).all().item(), name

print("输入形状:", x.shape)
print("输出形状:", output.shape)
print("Post-LayerNorm 顺序检查通过")
print("Padding mask 检查通过")
print("梯度检查通过")