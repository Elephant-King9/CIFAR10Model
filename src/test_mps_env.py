# 用于测试mps是否可以运行
import torch.backends.mps

print(torch.backends.mps.is_available())  # True代表MacOS版本支持
print(torch.backends.mps.is_built())  # True代表mps支持
