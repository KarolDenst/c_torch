import torch

x0 = torch.tensor([-2], dtype=torch.float32)
w1 = torch.tensor([-3], dtype=torch.float32)
w2 = torch.tensor([4], dtype=torch.float32)
x0.requires_grad = True
w1.requires_grad = True
w2.requires_grad = True
x1 = x0 * w1
x1.retain_grad()
x1w1 = x1 * w1
x1w1.retain_grad()
x1w2 = x1 * w2
x1w2.retain_grad()
x1w1_x1w2 = x1w1 + x1w2
x1w1_x1w2.retain_grad()

x1w1_x1w2.sum().backward()
print(x1w1_x1w2.grad)
print(x1w2.grad)
print(x1w1.grad)
print(w2.grad)
print(w1.grad)
print(x1.grad)
print(x0.grad)
