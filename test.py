import torch
import torch.nn.functional as F
import time


start = time.time()

x = torch.randn([1000, 1000])
y = torch.randn([1000, 1000])
for i in range(10):
    result = x + y
    result = x - y
    result = x * y
    result = x / y
    result = x @ y

end = time.time()

print(1000 * (end - start), "ms")
