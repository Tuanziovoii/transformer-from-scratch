import math

import torch

from embeddings import TokenEmbedding
from positional_encoding import PositionalEncoding


torch.manual_seed(42)

embedding = TokenEmbedding(vocab_size=10, d_model=8)
position_encoding = PositionalEncoding(
    d_model=8,
    dropout=0.0,
    max_len=20,
)

# 相同 token 出现在不同位置、不同序列中。
token_ids = torch.tensor([
    [2, 2, 2, 2],
    [2, 2, 2, 2],
], dtype=torch.long)

x = embedding(token_ids)
output = position_encoding(x)

assert output.shape == (2, 4, 8)

# 加入位置编码前，同一个 token 的向量相同。
torch.testing.assert_close(x[0, 0], x[0, 1])

# 加入后，同一个 token 在不同位置的向量不同。
assert not torch.allclose(output[0, 0], output[0, 1])

# 两条序列内容相同，同一位置的结果也应相同。
torch.testing.assert_close(output[0], output[1])

# 位置 0：sin(0)=0，cos(0)=1。
torch.testing.assert_close(
    position_encoding.pe[0, 0],
    torch.tensor([0., 1., 0., 1., 0., 1., 0., 1.]),
)

# 检查位置 1 的第 0、1 维。
torch.testing.assert_close(
    position_encoding.pe[0, 1, :2],
    torch.tensor([math.sin(1.0), math.cos(1.0)]),
)

# 位置编码不是可训练参数。
assert not position_encoding.pe.requires_grad
assert len(list(position_encoding.parameters())) == 0

# 梯度仍能穿过位置编码，传回 embedding。
output.sum().backward()
assert embedding.embedding.weight.grad is not None
assert embedding.embedding.weight.grad[2].abs().sum().item() > 0

print("Embedding 输出形状:", x.shape)
print("加入位置编码后的形状:", output.shape)
print("位置 0 的编码:", position_encoding.pe[0, 0])
print("全部检查通过")