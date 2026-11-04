import torch
from torch import nn

from feed_forward import PositionwiseFeedForward


torch.manual_seed(42)

model = PositionwiseFeedForward(d_model=8, d_ff=32)
x = torch.randn(2, 4, 8)

output = model(x)
assert output.shape == (2, 4, 8)

# 用相同参数检查“两层线性变换 + ReLU”的计算。
reference = nn.Sequential(
    model.linear1,
    nn.ReLU(),
    model.linear2,
)
torch.testing.assert_close(output, reference(x))

# 相同输入出现在不同位置，应得到相同输出。
repeated_x = x.clone()
repeated_x[0, 1] = repeated_x[0, 0]
repeated_output = model(repeated_x)

torch.testing.assert_close(
    repeated_output[0, 0],
    repeated_output[0, 1],
)

# 只修改一个位置，不应影响其他位置。
changed_x = x.clone()
changed_x[0, 0] += 10.0
changed_output = model(changed_x)

torch.testing.assert_close(output[0, 1:], changed_output[0, 1:])
torch.testing.assert_close(output[1], changed_output[1])

# 检查两个线性层的梯度。
output.square().mean().backward()

for name, parameter in model.named_parameters():
    assert parameter.grad is not None, name
    assert torch.isfinite(parameter.grad).all().item(), name
    assert parameter.grad.abs().sum().item() > 0, name

print("输入形状:", x.shape)
print("隐藏层形状:", model.linear1(x).shape)
print("输出形状:", output.shape)
print("全部检查通过")