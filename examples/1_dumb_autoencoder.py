import torch
import torch.nn.functional as F

model = torch.nn.Sequential(
    torch.nn.Linear(10, 5),
    torch.nn.Tanh(),
    torch.nn.Linear(5, 10),
)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for i in range(100000):
    random_digits = torch.randint(0, 10, (10,))
    one_hot_encoded = F.one_hot(random_digits, num_classes=10)
    one_hot_encoded = one_hot_encoded.float()
    data = one_hot_encoded.clone()
    expected = one_hot_encoded.clone()

    optimizer.zero_grad()
    result = model(data)
    loss = torch.nn.functional.mse_loss(result, expected, reduction="sum")
    if i % 10000 == 0:
        print(f"Iteration {i} Loss: {loss.item()}")
    loss.backward()
    optimizer.step()
