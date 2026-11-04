import sys
import torch

print("Python 路径:", sys.executable)
print("PyTorch 版本:", torch.__version__)
print("MPS 可用:", torch.backends.mps.is_available())

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

x = torch.tensor([1.0, 2.0, 3.0], device=device)
print("计算设备:", device)
print("计算结果:", x * 2)