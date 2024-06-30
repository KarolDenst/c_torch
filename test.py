import torch

t1 = torch.tensor([[1.0, 2.0]])
t2 = torch.tensor([[8.0, 6.0], [4.0, 2.0]])
t1.requires_grad = True
t2.requires_grad = True

sum = t1 @ t2
print(sum)
# sum.backward()
print(t1.grad)
print(t2.grad)
