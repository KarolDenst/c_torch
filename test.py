import torch
import torch.nn.functional as F

# auto w = Tensor({1.0, 2.0, 3.0, 4.0}, {2, 2});
# auto b = Tensor({1.0, 2.0}, {2});
# auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {3, 2});
# auto y = Tensor({-1.0, 2.0, -3.0, 4.0, -5.0, 6.0}, {3, 2});
#
# // act
# auto wx = x & w;
# auto wx_b = wx + b;
# auto o = tanh(&wx_b);
# auto loss = nn::functional::mse_loss(o, y);
w = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
b = torch.tensor([0.1, 0.2], dtype=torch.float32)
x = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32)
y = torch.tensor([[-0.1, 0.2], [-0.3, 0.4], [-0.5, 0.6]], dtype=torch.float32)
w.requires_grad = True
b.requires_grad = True

wx = x @ w
wx_b = wx + b
o = F.tanh(wx_b)
loss = F.mse_loss(o, y, reduction="sum")
loss.backward()

print("w.grad:", w.grad)
print("b.grad:", b.grad)
print("wx:", wx)
print("wx_b:", wx_b)
print("o:", o)
print("loss:", loss)
