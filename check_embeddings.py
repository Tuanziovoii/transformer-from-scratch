import torch

from embeddings import TokenEmbedding


torch.manual_seed(42)

model = TokenEmbedding(vocab_size=10, d_model=8)

token_ids = torch.tensor([
    [1, 2, 3, 2],
    [4, 5, 6, 0],
], dtype=torch.long)

output = model(token_ids)

print("输入形状:", token_ids.shape)
print("嵌入表形状:", model.embedding.weight.shape)
print("输出形状:", output.shape)

assert output.shape == (2, 4, 8)

# 同一个 token 编号，应该查出相同的向量。
torch.testing.assert_close(output[0, 1], output[0, 3])

# 检查输出确实使用了论文中的缩放。
torch.testing.assert_close(
    output[0, 0],
    model.embedding.weight[1] * (8 ** 0.5),
)

# 检查梯度能传回嵌入表，后续才能训练。
output.sum().backward()
assert model.embedding.weight.grad is not None
assert model.embedding.weight.grad[1].abs().sum().item() > 0

print("全部检查通过")