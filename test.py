import torch
import torch.nn.functional as F


x = torch.tensor([-1,-2,3,4],dtype=torch.float32, requires_grad=True)


out = x.relu()
out.sum().backward()

print(out)
print(x.grad)
