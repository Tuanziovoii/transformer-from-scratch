import torch

from decoder_layer import DecoderLayer


torch.manual_seed(42)

model = DecoderLayer(
    d_model=8,
    num_heads=2,
    d_ff=32,
    dropout=0.1,
)
model.eval()

# 目标长度为 4，源长度为 5。
x = torch.randn(2, 4, 8)
memory = torch.randn(2, 5, 8, requires_grad=True)

target_mask = torch.ones(4, 4, dtype=torch.bool).tril()

# 屏蔽源序列的最后一个位置。
memory_mask = torch.tensor(
    [[[[True, True, True, True, False]]]]
)

output = model(
    x,
    memory,
    target_mask=target_mask,
    memory_mask=memory_mask,
)

assert output.shape == (2, 4, 8)
assert torch.isfinite(output).all().item()

# 检查三个子层的顺序以及 Post-LayerNorm。
with torch.no_grad():
    self_output, _ = model.self_attention(
        x, x, x, mask=target_mask
    )
    after_self = model.norm1(x + self_output)

    cross_output, _ = model.cross_attention(
        after_self, memory, memory, mask=memory_mask
    )
    after_cross = model.norm2(after_self + cross_output)

    expected = model.norm3(
        after_cross + model.feed_forward(after_cross)
    )

torch.testing.assert_close(output, expected)

# 改变目标序列最后一个位置，前三个输出不能改变。
changed_x = x.clone()
changed_x[:, 3, :] += torch.arange(8, dtype=x.dtype) * 10

changed_output = model(
    changed_x,
    memory,
    target_mask=target_mask,
    memory_mask=memory_mask,
)

torch.testing.assert_close(
    output[:, :3, :],
    changed_output[:, :3, :],
)

# 改变被屏蔽的源位置，所有输出都不应改变。
changed_memory = memory.detach().clone()
changed_memory[:, 4, :] += 100.0

changed_output = model(
    x,
    changed_memory,
    target_mask=target_mask,
    memory_mask=memory_mask,
)

torch.testing.assert_close(output, changed_output)

# 检查 decoder 参数和 encoder 输出方向的梯度。
target = torch.randn_like(output)
loss = (output - target).square().mean()
loss.backward()

for name, parameter in model.named_parameters():
    assert parameter.grad is not None, name
    assert torch.isfinite(parameter.grad).all().item(), name

assert memory.grad is not None
assert torch.isfinite(memory.grad).all().item()
assert memory.grad[:, :4, :].abs().sum().item() > 0

# 被屏蔽的源位置不应接收交叉注意力的梯度。
torch.testing.assert_close(
    memory.grad[:, 4, :],
    torch.zeros_like(memory.grad[:, 4, :]),
)

print("目标输入形状:", x.shape)
print("Encoder 输出形状:", memory.shape)
print("Decoder 输出形状:", output.shape)
print("子层顺序、因果 mask、源 padding mask 检查通过")
print("梯度检查通过")