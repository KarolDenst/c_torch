import torch

t1 = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
t2 = torch.tensor([[2, 4, 6, 8]], dtype=torch.float32)
t1.requires_grad = True
t2.requires_grad = True

result = t1 / t2
result.sum().backward()
print(result)
print(t1.grad)
print(t2.grad)
