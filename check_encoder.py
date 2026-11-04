import torch

from encoder import Encoder


torch.manual_seed(42)

model = Encoder(
    num_layers=2,
    d_model=8,
    num_heads=2,
    d_ff=32,
    dropout=0.1,
)
model.eval()

x = torch.randn(2, 4, 8)

# 屏蔽每条序列的最后一个 key。
mask = torch.tensor([[[[True, True, True, False]]]])

output = model(x, mask=mask)

assert output.shape == x.shape
assert torch.isfinite(output).all().item()

# 两层参数不能是同一个对象或共用同一块存储。
params0 = dict(model.layers[0].named_parameters())
params1 = dict(model.layers[1].named_parameters())

for name in params0:
    assert params0[name] is not params1[name], name
    assert params0[name].data_ptr() != params1[name].data_ptr(), name

# 确认 forward 确实按顺序经过了两层。
with torch.no_grad():
    expected = model.layers[0](x, mask=mask)
    expected = model.layers[1](expected, mask=mask)

torch.testing.assert_close(output, expected)

# mask 必须在所有层中生效。
changed_x = x.clone()
changed_x[:, 3, :] += 100.0
changed_output = model(changed_x, mask=mask)

torch.testing.assert_close(
    output[:, :3, :],
    changed_output[:, :3, :],
)

# 检查每层参数都接入了计算图，且梯度有限。
target = torch.randn_like(output)
loss = (output - target).square().mean()
loss.backward()

for name, parameter in model.named_parameters():
    assert parameter.grad is not None, name
    assert torch.isfinite(parameter.grad).all().item(), name

print("Encoder 层数:", len(model.layers))
print("输出形状:", output.shape)
print("各层参数独立")
print("堆叠顺序、mask 和梯度检查通过")