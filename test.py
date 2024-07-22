import torch
import torch.nn.functional as F

x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
# y = torch.tensor([5, 6], dtype=torch.float32)
x.requires_grad = True
# y.requires_grad = True
# result = y - x
result = x.sum(dim=1)
result.sum().backward()

print(result)
print(x.grad)
# print(y.grad)
