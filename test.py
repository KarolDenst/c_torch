import torch
import torch.nn.functional as F
import time


start = time.time()

x = torch.randn([1000, 1000])
y = torch.randn([1000, 1000])
for i in range(10):
    result = x + y
    result = x + y
    result = x + y
    result = x + y
    result = x + y

end = time.time()

print(1000 * (end - start), "ms")

# time python matmul - 1470 ms
# time cpp matmul - 8909 ms
# time python others - 72 ms
# time cpp others - 723 ms
