import math

import torch


# 一个查询，向量维度为 2。
q = torch.tensor([[1.0, 0.0]])

# 两个可供查询的位置。
k = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
])

# 两个位置各自携带的信息。
v = torch.tensor([
    [10.0, 0.0],
    [0.0, 20.0],
])

# 1. 计算查询与每个 key 的匹配分数。
scores = q @ k.transpose(-2, -1)

# 2. 除以 key 向量维度的平方根。
scaled_scores = scores / math.sqrt(k.size(-1))

# 3. 将分数转换成权重，对所有 key 的权重求和为 1。
weights = torch.softmax(scaled_scores, dim=-1)

# 4. 用权重对 value 加权求和。
output = weights @ v

print("原始分数:", scores)
print("缩放后分数:", scaled_scores)
print("注意力权重:", weights)
print("输出:", output)

torch.testing.assert_close(
    weights.sum(dim=-1),
    torch.ones(1),
)

torch.testing.assert_close(
    output[0],
    weights[0, 0] * v[0] + weights[0, 1] * v[1],
)

print("检查通过")